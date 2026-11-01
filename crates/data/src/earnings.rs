//! Earnings dates for US-listed companies from SEC EDGAR. Companies furnish each earnings
//! release on Form 8-K under Item 2.02 (Results of Operations and Financial Condition), so the
//! report dates of those 8-Ks are the days results came out. Companies that file no such 8-K
//! fall back to the filing dates of their quarterly and annual reports (10-Q, 10-K).
//!
//! Sources, both free and keyless: `www.sec.gov/files/company_tickers.json` maps tickers to
//! CIKs, and `data.sec.gov/submissions/CIK##########.json` lists a company's filings. The SEC
//! asks automated clients to declare themselves in the User-Agent; `TR_SEC_USER_AGENT`
//! replaces the default one.

use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use chrono::NaiveDate;
use serde_json::Value;
#[cfg(not(target_arch = "wasm32"))]
use tokio::sync::Mutex;

#[cfg(not(target_arch = "wasm32"))]
const TICKERS: &str = "https://www.sec.gov/files/company_tickers.json";
#[cfg(not(target_arch = "wasm32"))]
const SUBMISSIONS: &str = "https://data.sec.gov/submissions";
/// Older filings pages fetched beyond the recent list, which large companies fill in a couple
/// of years with insider reports.
#[cfg(not(target_arch = "wasm32"))]
const MAX_PAGES: usize = 4;
/// Earnings come out four times a year, so a day-old list is fresh enough.
#[cfg(not(target_arch = "wasm32"))]
const TTL: Duration = Duration::from_secs(12 * 3600);

/// One earnings release.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Earning {
    /// The day results came out.
    pub date: NaiveDate,
    /// End of the reported period, when the filing gives it (10-Q, 10-K).
    pub period: Option<NaiveDate>,
    /// The form it was found on: `8-K`, `10-Q` or `10-K`.
    pub form: String,
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
        .map(|f| Earning {
            // The 8-K's date of report is the release day; it's filed that day or soon after.
            date: f.report.filter(|r| *r <= f.filed && (f.filed - *r).num_days() <= 4).unwrap_or(f.filed),
            period: None,
            form: f.form.clone(),
        })
        .collect();
    let mut out = if releases.is_empty() {
        filings
            .iter()
            .filter(|f| f.form == "10-Q" || f.form == "10-K")
            .map(|f| Earning { date: f.filed, period: f.report, form: f.form.clone() })
            .collect()
    } else {
        releases
    };
    out.sort_by_key(|e| std::cmp::Reverse(e.date));
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

/// Fetches and caches earnings dates per ticker. Desktop only: EDGAR doesn't allow
/// cross-origin requests from a browser page.
#[cfg(not(target_arch = "wasm32"))]
pub struct Earnings {
    client: reqwest::Client,
    ciks: Mutex<Option<HashMap<String, u64>>>,
    cache: Mutex<HashMap<String, (Instant, Vec<Earning>)>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl Earnings {
    pub fn new() -> Result<Self> {
        let agent = std::env::var("TR_SEC_USER_AGENT").unwrap_or_else(|_| {
            concat!("terminal/", env!("CARGO_PKG_VERSION"), " (personal market research; earnings dates)").into()
        });
        let client = reqwest::Client::builder().user_agent(agent).timeout(Duration::from_secs(30)).build()?;
        Ok(Self { client, ciks: Mutex::new(None), cache: Mutex::new(HashMap::new()) })
    }

    async fn get(&self, url: &str) -> Result<String> {
        let r = self.client.get(url).send().await.with_context(|| format!("GET {url}"))?;
        let status = r.status();
        if !status.is_success() {
            return Err(anyhow!("GET {url}: HTTP {status}"));
        }
        Ok(r.text().await?)
    }

    async fn cik(&self, ticker: &str) -> Result<Option<u64>> {
        let mut ciks = self.ciks.lock().await;
        if ciks.is_none() {
            *ciks = Some(parse_tickers(&self.get(TICKERS).await?)?);
        }
        Ok(ciks.as_ref().and_then(|m| m.get(&edgar_ticker(ticker)).copied()))
    }

    /// Earnings releases for `ticker`, newest first; empty for tickers EDGAR doesn't list
    /// (ETFs and funds file no earnings).
    pub async fn for_ticker(&self, ticker: &str) -> Result<Vec<Earning>> {
        let key = edgar_ticker(ticker);
        if let Some((at, e)) = self.cache.lock().await.get(&key)
            && at.elapsed() < TTL
        {
            return Ok(e.clone());
        }
        let Some(cik) = self.cik(&key).await? else { return Ok(Vec::new()) };
        let body = self.get(&format!("{SUBMISSIONS}/CIK{cik:010}.json")).await?;
        let v: Value = serde_json::from_str(&body).context("submissions JSON")?;
        // Pages back about ten years, enough for a daily chart's usual history.
        let cutoff = chrono::Utc::now().date_naive() - chrono::Days::new(3660);
        let mut older = Vec::new();
        for (name, to) in older_pages(&v).into_iter().take(MAX_PAGES) {
            if to.is_some_and(|t| t < cutoff) {
                break;
            }
            match self.get(&format!("{SUBMISSIONS}/{name}")).await {
                Ok(p) => older.push(p),
                Err(e) => {
                    tracing::warn!("earnings {key}: {e:#}");
                    break;
                }
            }
        }
        let refs: Vec<&str> = older.iter().map(String::as_str).collect();
        let earnings = parse_submissions(&body, &refs)?;
        self.cache.lock().await.insert(key, (Instant::now(), earnings.clone()));
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
}
