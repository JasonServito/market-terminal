//! Earnings for US-listed companies: when results came out, what they were, and how they
//! compared with what analysts expected.
//!
//! - Release dates come from SEC EDGAR. Companies furnish each earnings release on Form 8-K
//!   under Item 2.02 (Results of Operations and Financial Condition), so the report dates of
//!   those 8-Ks are the days results came out. Companies that file no such 8-K fall back to
//!   the filing dates of their quarterly and annual reports (10-Q, 10-K).
//! - GAAP diluted EPS and revenue per quarter come from the XBRL financial data in those
//!   reports (`data.sec.gov/api/xbrl/companyfacts`).
//! - Analyst EPS estimates, and the EPS figure they're measured against, come from Yahoo
//!   Finance: its earnings calendar page goes back years, and `quoteSummary` has the last
//!   four quarters and the next report with its revenue estimate. Revenue estimates for past
//!   quarters aren't available from a free source, so only the next report has one.
//!
//! The SEC sources are free and keyless: `www.sec.gov/files/company_tickers.json` maps tickers
//! to CIKs, and `data.sec.gov/submissions/CIK##########.json` lists a company's filings. The
//! SEC asks automated clients to declare themselves in the User-Agent; `TR_SEC_USER_AGENT`
//! replaces the default one.

use std::cmp::Reverse;
use std::collections::{BTreeMap, HashMap};
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use chrono::{Datelike, Days, Months, NaiveDate};
use serde_json::Value;
#[cfg(not(target_arch = "wasm32"))]
use tokio::sync::Mutex;

#[cfg(not(target_arch = "wasm32"))]
const TICKERS: &str = "https://www.sec.gov/files/company_tickers.json";
#[cfg(not(target_arch = "wasm32"))]
const SUBMISSIONS: &str = "https://data.sec.gov/submissions";
#[cfg(not(target_arch = "wasm32"))]
const COMPANY_FACTS: &str = "https://data.sec.gov/api/xbrl/companyfacts";
/// Older filings pages fetched beyond the recent list, which large companies fill in a couple
/// of years with insider reports.
#[cfg(not(target_arch = "wasm32"))]
const MAX_PAGES: usize = 4;
/// Earnings come out four times a year; a few hours old is fresh enough, and a new report
/// shows up the same day.
#[cfg(not(target_arch = "wasm32"))]
const TTL: Duration = Duration::from_secs(3 * 3600);
/// How soon to try again when a source couldn't be reached.
#[cfg(not(target_arch = "wasm32"))]
const RETRY: Duration = Duration::from_secs(20 * 60);

/// Diluted EPS concepts in order of preference.
const EPS_CONCEPTS: [&str; 3] = ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic"];
/// Total revenue concepts in order of preference; companies move between them over the years.
const REVENUE_CONCEPTS: [&str; 5] = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
];

/// One earnings report: a release that came out, or the next one.
#[derive(Debug, Clone, PartialEq)]
pub struct Earning {
    /// The day results came out, or are expected.
    pub date: NaiveDate,
    /// End of the reported quarter (estimated for the next report).
    pub period: Option<NaiveDate>,
    /// The form the date was found on: `8-K`, `10-Q` or `10-K`; empty when only Yahoo
    /// Finance lists it.
    pub form: String,
    /// Diluted EPS under GAAP, from the company's filings.
    pub eps_gaap: Option<f64>,
    /// EPS as analysts count it (often adjusted), which the estimate is compared with.
    pub eps_actual: Option<f64>,
    /// Consensus EPS estimate.
    pub eps_estimate: Option<f64>,
    /// Revenue for the quarter, from the company's filings.
    pub revenue: Option<f64>,
    /// Consensus revenue estimate (next report only).
    pub revenue_estimate: Option<f64>,
    /// Results haven't come out yet.
    pub upcoming: bool,
}

/// How a report compared with the EPS analysts expected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Beat,
    Miss,
    InLine,
    Upcoming,
    /// No estimate or no comparable EPS to judge by.
    Unknown,
}

/// A reported figure less its estimate, and that as a percentage of the estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Surprise {
    pub diff: f64,
    pub pct: Option<f64>,
}

impl Earning {
    pub fn new(date: NaiveDate, period: Option<NaiveDate>, form: impl Into<String>) -> Self {
        Self {
            date,
            period,
            form: form.into(),
            eps_gaap: None,
            eps_actual: None,
            eps_estimate: None,
            revenue: None,
            revenue_estimate: None,
            upcoming: false,
        }
    }

    pub fn eps_surprise(&self) -> Option<Surprise> {
        surprise(self.eps_actual?, self.eps_estimate?)
    }

    pub fn revenue_surprise(&self) -> Option<Surprise> {
        surprise(self.revenue?, self.revenue_estimate?)
    }

    /// Beat or miss on EPS; within half a cent counts as in line.
    pub fn outcome(&self) -> Outcome {
        if self.upcoming {
            return Outcome::Upcoming;
        }
        match self.eps_surprise() {
            Some(s) if s.diff.abs() < 0.005 => Outcome::InLine,
            Some(s) if s.diff > 0.0 => Outcome::Beat,
            Some(_) => Outcome::Miss,
            None => Outcome::Unknown,
        }
    }
}

fn surprise(actual: f64, estimate: f64) -> Option<Surprise> {
    let diff = actual - estimate;
    Some(Surprise { diff, pct: (estimate.abs() > 1e-9).then(|| diff / estimate.abs() * 100.0) })
}

/// Ticker -> CIK from `company_tickers.json` (`{"0":{"cik_str":320193,"ticker":"AAPL",…},…}`).
pub fn parse_tickers(json: &str) -> Result<HashMap<String, u64>> {
    let v: Value = serde_json::from_str(json).context("company_tickers.json")?;
    let rows = v.as_object().ok_or_else(|| anyhow!("company_tickers.json is not an object"))?;
    Ok(rows
        .values()
        .filter_map(|r| {
            let ticker = r.get("ticker")?.as_str()?.to_ascii_uppercase();
            let cik = r.get("cik_str").and_then(|c| c.as_u64().or_else(|| c.as_str()?.parse().ok()))?;
            Some((ticker, cik))
        })
        .collect())
}

/// EDGAR spells share classes with a dash (`BRK-B`); market keys may use a dot or a slash.
pub fn edgar_ticker(ticker: &str) -> String {
    ticker.trim().to_ascii_uppercase().replace(['.', '/'], "-")
}

/// One page of filings: the columns object (`form`, `filingDate`, `reportDate`, `items`, …),
/// either `filings.recent` of a submissions file or the whole of an older page.
fn filings_of(page: &Value) -> Vec<Filing> {
    let col = |name: &str| page.get(name).and_then(Value::as_array).cloned().unwrap_or_default();
    let (forms, filed, report, items) = (col("form"), col("filingDate"), col("reportDate"), col("items"));
    let date =
        |v: Option<&Value>| v.and_then(Value::as_str).and_then(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());
    forms
        .iter()
        .enumerate()
        .filter_map(|(i, f)| {
            Some(Filing {
                form: f.as_str()?.to_string(),
                filed: date(filed.get(i))?,
                report: date(report.get(i)),
                items: items.get(i).and_then(Value::as_str).unwrap_or_default().to_string(),
            })
        })
        .collect()
}

#[derive(Debug, Clone)]
struct Filing {
    form: String,
    filed: NaiveDate,
    report: Option<NaiveDate>,
    items: String,
}

/// Older filings pages named by a submissions file (`filings.files`), newest first, with the
/// last date each covers.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
fn older_pages(submissions: &Value) -> Vec<(String, Option<NaiveDate>)> {
    let files = submissions.pointer("/filings/files").and_then(Value::as_array).cloned().unwrap_or_default();
    files
        .iter()
        .filter_map(|f| {
            let name = f.get("name")?.as_str()?.to_string();
            let to =
                f.get("filingTo").and_then(Value::as_str).and_then(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok());
            Some((name, to))
        })
        .collect()
}

/// Earnings releases in some filings pages, newest first.
fn earnings_from(filings: &[Filing]) -> Vec<Earning> {
    let releases: Vec<Earning> = filings
        .iter()
        .filter(|f| f.form == "8-K" && f.items.split(',').any(|i| i.trim() == "2.02"))
        .map(|f| {
            // The 8-K's date of report is the release day; it's filed that day or soon after.
            let date = f.report.filter(|r| *r <= f.filed && (f.filed - *r).num_days() <= 4).unwrap_or(f.filed);
            Earning::new(date, None, f.form.clone())
        })
        .collect();
    let mut out = if releases.is_empty() {
        filings
            .iter()
            .filter(|f| f.form == "10-Q" || f.form == "10-K")
            .map(|f| Earning::new(f.filed, f.report, f.form.clone()))
            .collect()
    } else {
        releases
    };
    out.sort_by_key(|e| Reverse(e.date));
    // One marker per release: amendments and second filings within a few days fold into it.
    out.dedup_by(|later, kept| (kept.date - later.date).num_days().abs() <= 5);
    out
}

/// Earnings releases from a submissions file and any older pages, newest first.
pub fn parse_submissions(submissions: &str, older: &[&str]) -> Result<Vec<Earning>> {
    let v: Value = serde_json::from_str(submissions).context("submissions JSON")?;
    let recent = v.pointer("/filings/recent").ok_or_else(|| anyhow!("submissions JSON has no filings.recent"))?;
    let mut filings = filings_of(recent);
    for page in older {
        let p: Value = serde_json::from_str(page).context("older filings page")?;
        filings.extend(filings_of(&p));
    }
    Ok(earnings_from(&filings))
}

fn ymd(v: Option<&Value>) -> Option<NaiveDate> {
    v.and_then(Value::as_str).and_then(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d").ok())
}

/// A number, or Yahoo's `{"raw": 1.23, "fmt": "1.23"}` form of one.
fn num(v: Option<&Value>) -> Option<f64> {
    let v = v?;
    v.as_f64().or_else(|| v.get("raw")?.as_f64()).filter(|x| x.is_finite())
}

/// Quarterly GAAP figures from a company's XBRL financial data, by period end.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Fundamentals {
    pub eps: BTreeMap<NaiveDate, f64>,
    pub revenue: BTreeMap<NaiveDate, f64>,
}

/// The value for the period ending within `days` of `end`.
fn near(map: &BTreeMap<NaiveDate, f64>, end: NaiveDate, days: u64) -> Option<f64> {
    let (lo, hi) = (end - Days::new(days), end + Days::new(days));
    map.range(lo..=hi).min_by_key(|(d, _)| (**d - end).num_days().abs()).map(|(_, v)| *v)
}

/// Quarterly diluted EPS and revenue from `companyfacts/CIK##########.json`.
pub fn parse_companyfacts(json: &str) -> Result<Fundamentals> {
    let v: Value = serde_json::from_str(json).context("companyfacts JSON")?;
    let Some(gaap) = v.pointer("/facts/us-gaap") else { return Ok(Fundamentals::default()) };
    Ok(Fundamentals { eps: merged(gaap, &EPS_CONCEPTS), revenue: merged(gaap, &REVENUE_CONCEPTS) })
}

/// Quarterly values, each period from the first concept in `concepts` that has it.
fn merged(gaap: &Value, concepts: &[&str]) -> BTreeMap<NaiveDate, f64> {
    let mut out = BTreeMap::new();
    for c in concepts {
        let Some(units) = gaap.get(*c).and_then(|f| f.get("units")).and_then(Value::as_object) else { continue };
        for facts in units.values() {
            for (end, val) in quarterly(facts) {
                if near(&out, end, 7).is_none() {
                    out.insert(end, val);
                }
            }
        }
    }
    out
}

/// Three-month values from one concept's facts. A company often reports only the year in
/// its 10-K, so a fourth quarter is the year less its first three. Each period keeps its
/// latest filing, which carries restatements and split adjustments.
fn quarterly(facts: &Value) -> Vec<(NaiveDate, f64)> {
    // (end, is_year) -> (filed, start, value)
    let mut latest: HashMap<(NaiveDate, bool), (NaiveDate, NaiveDate, f64)> = HashMap::new();
    for f in facts.as_array().into_iter().flatten() {
        let (Some(start), Some(end), Some(val)) = (ymd(f.get("start")), ymd(f.get("end")), num(f.get("val"))) else {
            continue;
        };
        let filed = ymd(f.get("filed")).unwrap_or(end);
        let year = match (end - start).num_days() {
            80..=100 => false,
            350..=380 => true,
            _ => continue,
        };
        let slot = latest.entry((end, year)).or_insert((filed, start, val));
        if filed > slot.0 {
            *slot = (filed, start, val);
        }
    }
    let quarters: Vec<(NaiveDate, NaiveDate, f64)> =
        latest.iter().filter(|((_, y), _)| !y).map(|((end, _), (_, start, v))| (*start, *end, *v)).collect();
    let mut out: Vec<(NaiveDate, f64)> = quarters.iter().map(|(_, end, v)| (*end, *v)).collect();
    for ((year_end, _), (_, year_start, total)) in latest.iter().filter(|((_, y), _)| *y) {
        if quarters.iter().any(|(_, end, _)| (*end - *year_end).num_days().abs() <= 7) {
            continue;
        }
        let earlier: Vec<f64> = quarters
            .iter()
            .filter(|(start, end, _)| *start >= *year_start - Days::new(7) && *end <= *year_end - Days::new(30))
            .map(|(_, _, v)| *v)
            .collect();
        if earlier.len() == 3 {
            out.push((*year_end, total - earlier.iter().sum::<f64>()));
        }
    }
    out
}

/// EPS against its estimate for one quarter or report.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpsRow {
    /// Period end (`quoteSummary`) or report date (earnings calendar).
    pub date: NaiveDate,
    pub actual: Option<f64>,
    pub estimate: Option<f64>,
}

/// The next report Yahoo Finance expects.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Next {
    pub date: NaiveDate,
    pub eps_estimate: Option<f64>,
    pub revenue_estimate: Option<f64>,
}

/// The US Eastern calendar day of a Unix time, near enough for report times (Yahoo stamps
/// them at the release hour, or midnight Eastern when the hour isn't known).
fn eastern_day(ts: f64) -> Option<NaiveDate> {
    chrono::DateTime::from_timestamp(ts as i64 - 5 * 3600, 0).map(|t| t.date_naive())
}

/// The last four quarters (`earningsHistory`) and the next report (`calendarEvents`) from a
/// `quoteSummary` response.
pub fn parse_quote_summary(json: &str) -> Result<(Vec<EpsRow>, Option<Next>)> {
    let v: Value = serde_json::from_str(json).context("quoteSummary JSON")?;
    let r = v.pointer("/quoteSummary/result/0").ok_or_else(|| anyhow!("quoteSummary has no result"))?;
    let quarters = r
        .pointer("/earningsHistory/history")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|q| {
            let quarter = q.get("quarter")?;
            let date = num(Some(quarter))
                .and_then(|ts| chrono::DateTime::from_timestamp(ts as i64, 0))
                .map(|t| t.date_naive())
                .or_else(|| ymd(quarter.get("fmt")))?;
            Some(EpsRow { date, actual: num(q.get("epsActual")), estimate: num(q.get("epsEstimate")) })
        })
        .collect();
    let next = r.pointer("/calendarEvents/earnings").and_then(|e| {
        let first = e.get("earningsDate")?.as_array()?.first()?;
        Some(Next {
            date: eastern_day(num(Some(first))?)?,
            eps_estimate: num(e.get("earningsAverage")),
            revenue_estimate: num(e.get("revenueAverage")),
        })
    });
    Ok((quarters, next))
}

/// "Oct 30, 2025, 4 PM EDT" or "October 30, 2025 at 4 PM EDT" -> 2025-10-30.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
fn calendar_date(s: &str) -> Option<NaiveDate> {
    let mut words = Vec::new();
    for w in s.split_whitespace() {
        words.push(w);
        let w = w.trim_end_matches(',');
        if w.len() == 4 && w.bytes().all(|b| b.is_ascii_digit()) {
            let day = words.join(" ");
            let day = day.trim_end_matches(',');
            return ["%b %d, %Y", "%B %d, %Y", "%b %d %Y", "%B %d %Y"]
                .iter()
                .find_map(|f| NaiveDate::parse_from_str(day, f).ok());
        }
    }
    None
}

/// "1.64", "+0.04", "-" (none).
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
fn figure(s: &str) -> Option<f64> {
    let s = s.trim().trim_start_matches('+').replace(',', "");
    s.parse::<f64>().ok().filter(|x| x.is_finite())
}

/// Report dates with EPS estimates and reported EPS from Yahoo's earnings calendar page
/// (`finance.yahoo.com/calendar/earnings?symbol=…`), newest first. Columns are found by
/// their headings; a page without the table gives nothing.
#[cfg(not(target_arch = "wasm32"))]
pub fn parse_calendar(html: &str) -> Vec<EpsRow> {
    use scraper::{ElementRef, Html, Selector};
    let sel = |s: &str| Selector::parse(s).expect("static selector");
    let (table, tr, cell) = (sel("table"), sel("tr"), sel("th, td"));
    let text = |e: ElementRef| e.text().collect::<Vec<_>>().join(" ").split_whitespace().collect::<Vec<_>>().join(" ");
    let doc = Html::parse_document(html);
    for t in doc.select(&table) {
        let mut rows = t.select(&tr).map(|r| r.select(&cell).map(text).collect::<Vec<String>>());
        let Some(head) = rows.next() else { continue };
        let col = |want: &str| head.iter().position(|h| h.to_ascii_lowercase().contains(want));
        let (Some(dc), Some(ec), Some(rc)) = (col("date"), col("estimate"), col("reported")) else { continue };
        return rows
            .filter_map(|c| {
                let date = calendar_date(c.get(dc)?)?;
                Some(EpsRow { date, actual: figure(c.get(rc)?), estimate: figure(c.get(ec)?) })
            })
            .collect();
    }
    Vec::new()
}

/// The release within `days` of `date`, closest first.
fn closest(releases: &mut [Earning], date: NaiveDate, days: i64) -> Option<&mut Earning> {
    releases
        .iter_mut()
        .filter(|e| (e.date - date).num_days().abs() <= days)
        .min_by_key(|e| (e.date - date).num_days().abs())
}

/// The quarter after one ending on `p`: month-end quarters stay on month ends, 52/53-week
/// years move 13 weeks.
fn next_quarter_end(p: NaiveDate) -> NaiveDate {
    let month_end = p.succ_opt().is_some_and(|d| d.day() == 1);
    let by_month = p.with_day(1).and_then(|d| d.checked_add_months(Months::new(4))).and_then(|d| d.pred_opt());
    match by_month {
        Some(d) if month_end => d,
        _ => p + Days::new(91),
    }
}

/// Lines the sources up into one list, newest first, with the next report on top.
///
/// SEC releases are the backbone. Each takes the last quarter that ended within 100 days
/// before it, and that quarter's GAAP EPS and revenue. Yahoo's calendar rows join releases
/// within five days of them (or stand alone for companies the SEC doesn't list), and its
/// last four quarters join the release for that quarter.
pub fn combine(
    mut releases: Vec<Earning>,
    facts: &Fundamentals,
    quarters: &[EpsRow],
    reports: &[EpsRow],
    next: Option<Next>,
    today: NaiveDate,
) -> Vec<Earning> {
    releases.retain(|e| e.date <= today);
    let mut listed: Option<Next> = None;
    for r in reports {
        if r.date > today {
            if r.actual.is_none() && listed.is_none_or(|n| r.date < n.date) {
                listed = Some(Next { date: r.date, eps_estimate: r.estimate, revenue_estimate: None });
            }
            continue;
        }
        if r.actual.is_none() && r.estimate.is_none() {
            continue;
        }
        match closest(&mut releases, r.date, 5) {
            Some(e) => {
                e.eps_actual = e.eps_actual.or(r.actual);
                e.eps_estimate = e.eps_estimate.or(r.estimate);
            }
            None => releases.push(Earning {
                eps_actual: r.actual,
                eps_estimate: r.estimate,
                ..Earning::new(r.date, None, "")
            }),
        }
    }
    releases.sort_by_key(|e| Reverse(e.date));

    // Quarter ends: the filings' own, then Yahoo's (calendar quarters) where filings have none.
    let mut ends: Vec<NaiveDate> = facts.eps.keys().chain(facts.revenue.keys()).copied().collect();
    ends.sort();
    ends.dedup_by(|b, a| (*b - *a).num_days().abs() <= 7);
    for q in quarters {
        if !ends.iter().any(|e| (*e - q.date).num_days().abs() <= 10) {
            ends.push(q.date);
        }
    }
    ends.sort();
    for e in releases.iter_mut().filter(|e| e.period.is_none()) {
        e.period = ends.iter().rev().find(|p| **p < e.date && (e.date - **p).num_days() <= 100).copied();
    }
    for e in &mut releases {
        if let Some(p) = e.period {
            e.eps_gaap = near(&facts.eps, p, 7);
            e.revenue = near(&facts.revenue, p, 7);
        }
    }

    for q in quarters {
        let same_period = releases.iter().position(|e| e.period.is_some_and(|p| (p - q.date).num_days().abs() <= 10));
        let after = || {
            releases.iter().rposition(|e| e.period.is_none() && e.date > q.date && (e.date - q.date).num_days() <= 100)
        };
        if let Some(i) = same_period.or_else(after) {
            let e = &mut releases[i];
            e.period = e.period.or(Some(q.date));
            e.eps_actual = q.actual.or(e.eps_actual);
            e.eps_estimate = q.estimate.or(e.eps_estimate);
        }
    }

    let latest = releases.first().map(|e| e.date);
    let next = match (next, listed) {
        (Some(n), Some(l)) if (n.date - l.date).num_days().abs() <= 7 => {
            Some(Next { eps_estimate: n.eps_estimate.or(l.eps_estimate), ..n })
        }
        (n, l) => n.filter(|n| n.date >= today).or(l),
    }
    // Yahoo can be slow to roll past a report that's already out.
    .filter(|n| n.date >= today && latest.is_none_or(|l| (n.date - l).num_days() > 20));
    if let Some(n) = next {
        let period = releases.iter().find_map(|e| e.period).map(next_quarter_end);
        releases.insert(
            0,
            Earning {
                eps_estimate: n.eps_estimate,
                revenue_estimate: n.revenue_estimate,
                upcoming: true,
                ..Earning::new(n.date, period, "")
            },
        );
    }
    releases
}

/// Fetches and caches earnings per ticker. Desktop only: EDGAR and Yahoo don't allow
/// cross-origin requests from a browser page.
#[cfg(not(target_arch = "wasm32"))]
pub struct Earnings {
    client: reqwest::Client,
    yahoo: crate::yahoo::YahooProvider,
    ciks: Mutex<Option<HashMap<String, u64>>>,
    cache: Mutex<HashMap<String, Cached>>,
}

/// A ticker's reports, when they were fetched and how long they stay fresh.
#[cfg(not(target_arch = "wasm32"))]
type Cached = (Instant, Duration, Vec<Earning>);

#[cfg(not(target_arch = "wasm32"))]
impl Earnings {
    pub fn new() -> Result<Self> {
        let agent = std::env::var("TR_SEC_USER_AGENT").unwrap_or_else(|_| {
            concat!("terminal/", env!("CARGO_PKG_VERSION"), " (personal market research; earnings dates)").into()
        });
        // Large companies' XBRL data runs to several megabytes.
        let client = reqwest::Client::builder().user_agent(agent).timeout(Duration::from_secs(60)).build()?;
        Ok(Self {
            client,
            yahoo: crate::yahoo::YahooProvider::new()?,
            ciks: Mutex::new(None),
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// The body of `url`; `None` when it doesn't exist.
    async fn get(&self, url: &str) -> Result<Option<String>> {
        let r = self.client.get(url).send().await.with_context(|| format!("GET {url}"))?;
        let status = r.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !status.is_success() {
            return Err(anyhow!("GET {url}: HTTP {status}"));
        }
        Ok(Some(r.text().await?))
    }

    async fn cik(&self, ticker: &str) -> Result<Option<u64>> {
        let mut ciks = self.ciks.lock().await;
        if ciks.is_none() {
            let body = self.get(TICKERS).await?.ok_or_else(|| anyhow!("{TICKERS}: HTTP 404"))?;
            *ciks = Some(parse_tickers(&body)?);
        }
        Ok(ciks.as_ref().and_then(|m| m.get(&edgar_ticker(ticker)).copied()))
    }

    /// Release dates from the company's filings, newest first.
    async fn releases(&self, key: &str, cik: u64) -> Result<Vec<Earning>> {
        let Some(body) = self.get(&format!("{SUBMISSIONS}/CIK{cik:010}.json")).await? else { return Ok(Vec::new()) };
        let v: Value = serde_json::from_str(&body).context("submissions JSON")?;
        // Pages back about ten years, enough for a daily chart's usual history.
        let cutoff = chrono::Utc::now().date_naive() - Days::new(3660);
        let mut older = Vec::new();
        for (name, to) in older_pages(&v).into_iter().take(MAX_PAGES) {
            if to.is_some_and(|t| t < cutoff) {
                break;
            }
            match self.get(&format!("{SUBMISSIONS}/{name}")).await {
                Ok(Some(p)) => older.push(p),
                Ok(None) => break,
                Err(e) => {
                    tracing::warn!("earnings {key}: {e:#}");
                    break;
                }
            }
        }
        let refs: Vec<&str> = older.iter().map(String::as_str).collect();
        parse_submissions(&body, &refs)
    }

    async fn facts(&self, cik: u64) -> Result<Fundamentals> {
        match self.get(&format!("{COMPANY_FACTS}/CIK{cik:010}.json")).await? {
            Some(body) => parse_companyfacts(&body),
            None => Ok(Fundamentals::default()),
        }
    }

    /// Earnings reports for `ticker` (`yahoo` is its Yahoo symbol), next report first, then
    /// newest first; empty for tickers no source lists (ETFs and funds report no earnings).
    /// Each source that can't be reached is left out and tried again a little later.
    pub async fn for_ticker(&self, ticker: &str, yahoo: &str) -> Result<Vec<Earning>> {
        let key = edgar_ticker(ticker);
        if let Some((at, ttl, e)) = self.cache.lock().await.get(&key)
            && at.elapsed() < *ttl
        {
            return Ok(e.clone());
        }
        let mut complete = true;
        let mut keep = |what: &str, e: anyhow::Error| {
            tracing::warn!("earnings {key} ({what}): {e:#}");
            complete = false;
        };
        let cik = self.cik(&key).await.unwrap_or_else(|e| {
            keep("SEC tickers", e);
            None
        });
        let sec = async {
            match cik {
                Some(c) => tokio::join!(self.releases(&key, c), self.facts(c)),
                None => (Ok(Vec::new()), Ok(Fundamentals::default())),
            }
        };
        let ((releases, facts), summary, calendar) = tokio::join!(
            sec,
            self.yahoo.quote_summary(yahoo, "earningsHistory,calendarEvents"),
            self.yahoo.earnings_calendar(yahoo),
        );
        let releases = releases.unwrap_or_else(|e| {
            keep("SEC filings", e);
            Vec::new()
        });
        let facts = facts.unwrap_or_else(|e| {
            keep("SEC financial data", e);
            Fundamentals::default()
        });
        let (quarters, next) = summary.and_then(|b| parse_quote_summary(&b)).unwrap_or_else(|e| {
            keep("Yahoo estimates", e);
            (Vec::new(), None)
        });
        let reports = calendar.map(|h| parse_calendar(&h)).unwrap_or_else(|e| {
            keep("Yahoo earnings calendar", e);
            Vec::new()
        });
        let today = chrono::Utc::now().date_naive();
        let earnings = combine(releases, &facts, &quarters, &reports, next, today);
        let ttl = if complete { TTL } else { RETRY };
        self.cache.lock().await.insert(key, (Instant::now(), ttl, earnings.clone()));
        Ok(earnings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(s: &str) -> NaiveDate {
        NaiveDate::parse_from_str(s, "%Y-%m-%d").unwrap()
    }

    #[test]
    fn maps_tickers_to_ciks() {
        let json = r#"{"0":{"cik_str":320193,"ticker":"AAPL","title":"Apple Inc."},
                       "1":{"cik_str":1067983,"ticker":"BRK-B","title":"BERKSHIRE HATHAWAY INC"}}"#;
        let m = parse_tickers(json).unwrap();
        assert_eq!(m.get("AAPL"), Some(&320193));
        assert_eq!(m.get(&edgar_ticker("brk.b")), Some(&1067983));
        assert_eq!(edgar_ticker("BF/B"), "BF-B");
    }

    #[test]
    fn earnings_come_from_item_2_02_eight_ks() {
        let json = r#"{"cik":"320193","filings":{"recent":{
            "form":      ["4",          "8-K",        "10-Q",       "8-K",        "8-K",        "8-K/A",      "8-K"],
            "filingDate":["2025-08-02", "2025-07-31", "2025-08-01", "2025-06-10", "2025-05-02", "2025-05-03", "2025-05-01"],
            "reportDate":["2025-07-30", "2025-07-31", "2025-06-28", "2025-06-09", "2025-05-01", "2025-05-01", "2025-05-01"],
            "items":     ["",           "2.02,9.01",  "",           "5.02",       "2.02,9.01",  "2.02",       "2.02"]
        },"files":[]}}"#;
        let e = parse_submissions(json, &[]).unwrap();
        let dates: Vec<NaiveDate> = e.iter().map(|x| x.date).collect();
        assert_eq!(dates, [d("2025-07-31"), d("2025-05-01")], "the 5.02 8-K and the 8-K/A are not releases");
        assert!(e.iter().all(|x| x.form == "8-K"));
    }

    #[test]
    fn older_pages_add_history_and_quarterly_reports_stand_in() {
        let recent = r#"{"filings":{"recent":{
            "form":["10-Q","10-K","S-8"],"filingDate":["2025-08-05","2025-03-01","2025-01-15"],
            "reportDate":["2025-06-30","2024-12-31",""],"items":["","",""]},
            "files":[{"name":"CIK0000000001-submissions-001.json","filingFrom":"2015-01-01","filingTo":"2024-12-31"}]}}"#;
        let older = r#"{"form":["10-Q"],"filingDate":["2024-11-04"],"reportDate":["2024-09-30"],"items":[""]}"#;
        let v: Value = serde_json::from_str(recent).unwrap();
        assert_eq!(older_pages(&v), [("CIK0000000001-submissions-001.json".to_string(), Some(d("2024-12-31")))]);
        let e = parse_submissions(recent, &[older]).unwrap();
        let got: Vec<(NaiveDate, Option<NaiveDate>)> = e.iter().map(|x| (x.date, x.period)).collect();
        assert_eq!(
            got,
            [
                (d("2025-08-05"), Some(d("2025-06-30"))),
                (d("2025-03-01"), Some(d("2024-12-31"))),
                (d("2024-11-04"), Some(d("2024-09-30")))
            ]
        );
    }
    #[test]
    fn quarterly_figures_come_from_xbrl_with_fourth_quarters_worked_out() {
        let json = r#"{"facts":{"us-gaap":{
          "EarningsPerShareDiluted":{"units":{"USD/shares":[
            {"start":"2023-10-01","end":"2023-12-30","val":2.18,"filed":"2024-02-02"},
            {"start":"2023-12-31","end":"2024-03-30","val":1.53,"filed":"2024-05-03"},
            {"start":"2023-12-31","end":"2024-03-30","val":1.52,"filed":"2025-05-02"},
            {"start":"2023-10-01","end":"2024-03-30","val":3.71,"filed":"2024-05-03"},
            {"start":"2024-03-31","end":"2024-06-29","val":1.40,"filed":"2024-08-02"},
            {"start":"2023-10-01","end":"2024-09-28","val":6.08,"filed":"2024-11-01"}
          ]}},
          "SalesRevenueNet":{"units":{"USD":[
            {"start":"2017-07-02","end":"2017-09-30","val":52579000000,"filed":"2017-11-03"}
          ]}},
          "RevenueFromContractWithCustomerExcludingAssessedTax":{"units":{"USD":[
            {"start":"2024-06-30","end":"2024-09-28","val":94930000000,"filed":"2024-11-01"}
          ]}}
        }}}"#;
        let f = parse_companyfacts(json).unwrap();
        let eps: Vec<(NaiveDate, f64)> = f.eps.iter().map(|(d, v)| (*d, (v * 100.0).round() / 100.0)).collect();
        assert_eq!(
            eps,
            [(d("2023-12-30"), 2.18), (d("2024-03-30"), 1.52), (d("2024-06-29"), 1.40), (d("2024-09-28"), 0.98)],
            "the six-month figure is skipped, the restated Q2 wins and Q4 is the year less Q1-Q3"
        );
        assert_eq!(f.revenue.get(&d("2017-09-30")), Some(&52_579_000_000.0));
        assert_eq!(f.revenue.get(&d("2024-09-28")), Some(&94_930_000_000.0));
        assert_eq!(near(&f.revenue, d("2024-09-30"), 7), Some(94_930_000_000.0));
    }

    #[test]
    fn quote_summary_gives_recent_quarters_and_the_next_report() {
        let json = r#"{"quoteSummary":{"result":[{
          "earningsHistory":{"history":[
            {"epsActual":{"raw":1.64,"fmt":"1.64"},"epsEstimate":{"raw":1.6,"fmt":"1.60"},"quarter":{"raw":1727654400,"fmt":"2024-09-30"}},
            {"epsActual":1.4,"epsEstimate":1.35,"quarter":1719705600}
          ]},
          "calendarEvents":{"earnings":{"earningsDate":[1738184400],"earningsAverage":2.35,"revenueAverage":124000000000}}
        }],"error":null}}"#;
        let (quarters, next) = parse_quote_summary(json).unwrap();
        assert_eq!(
            quarters,
            [
                EpsRow { date: d("2024-09-30"), actual: Some(1.64), estimate: Some(1.6) },
                EpsRow { date: d("2024-06-30"), actual: Some(1.4), estimate: Some(1.35) }
            ]
        );
        assert_eq!(
            next,
            Some(Next { date: d("2025-01-29"), eps_estimate: Some(2.35), revenue_estimate: Some(124e9) }),
            "4 PM Eastern is still the 29th"
        );
    }

    #[test]
    fn earnings_calendar_table_is_read_by_its_headings() {
        let html = r#"<html><body><table>
          <thead><tr><th>Symbol</th><th>Company</th><th>Earnings Date</th><th>EPS Estimate</th><th>Reported EPS</th><th>Surprise (%)</th></tr></thead>
          <tbody>
            <tr><td><a href="/q">AAPL</a></td><td>Apple Inc.</td><td>Jan 29, 2026, 4 PM EST</td><td>2.67</td><td>-</td><td>-</td></tr>
            <tr><td>AAPL</td><td>Apple Inc.</td><td>October 30, 2025 at 4 PM EDT</td><td>1.77</td><td>1.85</td><td>+4.52</td></tr>
            <tr><td>AAPL</td><td>Apple Inc.</td><td>Jul 31, 2025, 4 PM EDT</td><td>1.43</td><td>1.57</td><td>+9.79</td></tr>
          </tbody></table></body></html>"#;
        assert_eq!(
            parse_calendar(html),
            [
                EpsRow { date: d("2026-01-29"), actual: None, estimate: Some(2.67) },
                EpsRow { date: d("2025-10-30"), actual: Some(1.85), estimate: Some(1.77) },
                EpsRow { date: d("2025-07-31"), actual: Some(1.57), estimate: Some(1.43) }
            ]
        );
        assert!(parse_calendar("<html><body><p>Consent required</p></body></html>").is_empty());
        assert_eq!(calendar_date("Sep 4, 2025"), Some(d("2025-09-04")));
    }

    #[test]
    fn sources_line_up_on_each_release() {
        let releases = vec![
            Earning::new(d("2025-10-30"), None, "8-K"),
            Earning::new(d("2025-07-31"), None, "8-K"),
            Earning::new(d("2025-05-01"), None, "8-K"),
        ];
        let facts = Fundamentals {
            eps: [(d("2025-09-27"), 1.85), (d("2025-06-28"), 1.57), (d("2025-03-29"), 1.65)].into(),
            revenue: [(d("2025-09-27"), 102.5e9), (d("2025-06-28"), 94.0e9)].into(),
        };
        // Yahoo's quarters are calendar quarters; the filings' fiscal ones win.
        let quarters = [
            EpsRow { date: d("2025-09-30"), actual: Some(1.85), estimate: Some(1.77) },
            EpsRow { date: d("2025-03-31"), actual: Some(1.65), estimate: Some(1.66) },
        ];
        let reports = [
            EpsRow { date: d("2026-01-29"), actual: None, estimate: Some(2.60) },
            EpsRow { date: d("2025-07-31"), actual: Some(1.57), estimate: Some(1.43) },
            EpsRow { date: d("2025-05-01"), actual: Some(9.0), estimate: Some(9.0) },
            // A report the SEC didn't list stands on its own.
            EpsRow { date: d("2025-01-30"), actual: Some(2.40), estimate: Some(2.35) },
        ];
        let next = Some(Next { date: d("2026-01-29"), eps_estimate: Some(2.67), revenue_estimate: Some(138e9) });
        let e = combine(releases, &facts, &quarters, &reports, next, d("2025-11-02"));

        let dates: Vec<NaiveDate> = e.iter().map(|x| x.date).collect();
        assert_eq!(dates, [d("2026-01-29"), d("2025-10-30"), d("2025-07-31"), d("2025-05-01"), d("2025-01-30")]);
        let next = &e[0];
        assert!(next.upcoming);
        assert_eq!(
            (next.period, next.eps_estimate, next.revenue_estimate),
            (Some(d("2025-12-27")), Some(2.67), Some(138e9))
        );
        assert_eq!(next.outcome(), Outcome::Upcoming);

        assert_eq!(e[1].period, Some(d("2025-09-27")));
        assert_eq!((e[1].eps_gaap, e[1].revenue), (Some(1.85), Some(102.5e9)));
        assert_eq!(e[1].outcome(), Outcome::Beat);
        assert_eq!(e[2].outcome(), Outcome::Beat, "estimate from the calendar page");
        assert_eq!(e[3].eps_actual, Some(1.65), "quoteSummary wins over the calendar page");
        assert_eq!(e[3].outcome(), Outcome::Miss);
        assert_eq!((e[4].form.as_str(), e[4].outcome()), ("", Outcome::Beat));
        let s = e[1].eps_surprise().unwrap();
        assert!((s.diff - 0.08).abs() < 1e-9 && (s.pct.unwrap() - 4.519_774).abs() < 1e-3);
    }

    #[test]
    fn next_report_is_dropped_once_results_are_out() {
        let releases = vec![Earning::new(d("2025-10-30"), Some(d("2025-09-27")), "8-K")];
        let stale = Some(Next { date: d("2025-10-30"), eps_estimate: Some(1.77), revenue_estimate: None });
        let e = combine(releases, &Fundamentals::default(), &[], &[], stale, d("2025-10-30"));
        assert_eq!(e.len(), 1);
        assert!(!e[0].upcoming);
        assert_eq!(e[0].outcome(), Outcome::Unknown);
    }

    #[test]
    fn next_quarter_ends() {
        assert_eq!(next_quarter_end(d("2025-09-30")), d("2025-12-31"));
        assert_eq!(next_quarter_end(d("2025-11-30")), d("2026-02-28"));
        assert_eq!(next_quarter_end(d("2025-09-27")), d("2025-12-27"));
    }
}
