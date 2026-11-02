//! The indicator catalog. Every indicator is a [`Def`]: the code typed on the command line,
//! its name, aliases, category, parameters and placement, and a function from bars and
//! parameters to plotted [`Output`]s. Charts hold [`Spec`]s, a code with its parameters,
//! persisted as the label the user would type (`"BB 20 2"`).

use std::fmt::Write as _;
use std::sync::LazyLock;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{Bar, Timeframe};

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Category {
    MovingAverages,
    Bands,
    Momentum,
    Trend,
    Volatility,
    Volume,
    Statistics,
    Levels,
}

impl Category {
    pub const ALL: [Category; 8] = [
        Category::MovingAverages,
        Category::Bands,
        Category::Momentum,
        Category::Trend,
        Category::Volatility,
        Category::Volume,
        Category::Statistics,
        Category::Levels,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Category::MovingAverages => "Moving Averages",
            Category::Bands => "Bands & Channels",
            Category::Momentum => "Momentum",
            Category::Trend => "Trend",
            Category::Volatility => "Volatility",
            Category::Volume => "Volume",
            Category::Statistics => "Statistics",
            Category::Levels => "Levels",
        }
    }
}

/// One numeric parameter. Integer parameters are rounded; all are clamped to `min..=max`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Param {
    pub name: &'static str,
    pub default: f64,
    pub min: f64,
    pub max: f64,
    pub int: bool,
}

/// Longest window any length parameter accepts.
pub const MAX_LENGTH: f64 = 1000.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Plot {
    Line,
    /// Columns from zero, coloured by sign.
    Hist,
    /// A marker per defined value.
    Dots,
    /// A larger marker per defined value, for signals that should stand out (crosses).
    Marks,
    /// A flat dash across each bar, so a level that changes (a new session's pivot)
    /// starts afresh instead of joining the previous one.
    Steps,
    /// The area between the line and zero, filled (oscillator waves).
    Fill,
}

/// Colour role of an output; the app maps roles to its theme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tone {
    /// The indicator's own colour.
    Main,
    /// A signal or trigger line.
    Signal,
    /// A faded version of the main colour (outer bands).
    Dim,
    Up,
    Down,
    /// The n-th colour of the theme's series palette.
    Series(u8),
    /// A colour of its own, unpremultiplied RGBA, the same in every theme.
    Fixed([u8; 4]),
}

#[derive(Debug, Clone)]
pub struct Output {
    pub name: &'static str,
    /// Aligned with the bars; displaced outputs run past the last bar.
    pub values: Vec<f64>,
    pub plot: Plot,
    pub tone: Tone,
}

/// Bars an indicator runs on.
pub struct Input<'a> {
    pub bars: &'a [Bar],
    pub tf: Timeframe,
    pub d: Ohlcv,
}

impl<'a> Input<'a> {
    pub fn new(bars: &'a [Bar], tf: Timeframe) -> Self {
        Input { bars, tf, d: Ohlcv::new(bars) }
    }

    fn c(&self) -> &[f64] {
        &self.d.close
    }
}

type Compute = fn(&Input<'_>, &[f64]) -> Vec<Output>;

#[derive(Debug)]
pub struct Def {
    pub code: &'static str,
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub category: Category,
    pub params: Vec<Param>,
    /// Drawn in a sub-pane under the price chart rather than over prices.
    pub pane: bool,
    /// Fixed value range of the sub-pane (bounded oscillators).
    pub range: Option<(f64, f64)>,
    /// Reference levels drawn across the sub-pane.
    pub levels: &'static [f64],
    /// Several instances with different parameters can share a chart.
    pub multi: bool,
    compute: Compute,
}

impl Def {
    fn new(code: &'static str, name: &'static str, category: Category, pane: bool, compute: Compute) -> Self {
        Def {
            code,
            name,
            aliases: &[],
            category,
            params: Vec::new(),
            pane,
            range: None,
            levels: &[],
            multi: false,
            compute,
        }
    }

    fn alias(mut self, aliases: &'static [&'static str]) -> Self {
        self.aliases = aliases;
        self
    }

    fn param(mut self, name: &'static str, default: f64, min: f64, max: f64, int: bool) -> Self {
        self.params.push(Param { name, default, min, max, int });
        self
    }

    /// A window length.
    fn n(self, name: &'static str, default: f64) -> Self {
        self.param(name, default, 1.0, MAX_LENGTH, true)
    }

    /// An integer parameter with its own bounds.
    fn int(self, name: &'static str, default: f64, min: f64, max: f64) -> Self {
        self.param(name, default, min, max, true)
    }

    fn real(self, name: &'static str, default: f64, min: f64, max: f64) -> Self {
        self.param(name, default, min, max, false)
    }

    fn range(mut self, lo: f64, hi: f64) -> Self {
        self.range = Some((lo, hi));
        self
    }

    fn levels(mut self, levels: &'static [f64]) -> Self {
        self.levels = levels;
        self
    }

    fn multi(mut self) -> Self {
        self.multi = true;
        self
    }

    pub fn defaults(&self) -> Vec<f64> {
        self.params.iter().map(|p| p.default).collect()
    }

    /// `params` filled with defaults, clamped and rounded to this indicator's parameters.
    pub fn normalize(&self, params: &[f64]) -> Vec<f64> {
        self.params
            .iter()
            .enumerate()
            .map(|(k, p)| {
                let v = params.get(k).copied().filter(|v| v.is_finite()).unwrap_or(p.default).clamp(p.min, p.max);
                if p.int { v.round() } else { v }
            })
            .collect()
    }

    pub fn run(&self, input: &Input<'_>, params: &[f64]) -> Vec<Output> {
        (self.compute)(input, &self.normalize(params))
    }

    pub fn matches(&self, code: &str) -> bool {
        self.code.eq_ignore_ascii_case(code) || self.aliases.iter().any(|a| a.eq_ignore_ascii_case(code))
    }
}

fn overlay(code: &'static str, name: &'static str, category: Category, compute: Compute) -> Def {
    Def::new(code, name, category, false, compute)
}

fn pane(code: &'static str, name: &'static str, category: Category, compute: Compute) -> Def {
    Def::new(code, name, category, true, compute)
}

static CATALOG: LazyLock<Vec<Def>> = LazyLock::new(catalog);

/// Every indicator, grouped by category.
pub fn all() -> &'static [Def] {
    &CATALOG
}

/// The indicator with this code or alias (case-insensitive).
pub fn find(code: &str) -> Option<&'static Def> {
    all().iter().find(|d| d.matches(code))
}

/// An indicator on a chart: its code and full parameter list.
#[derive(Debug, Clone, PartialEq)]
pub struct Spec {
    pub code: String,
    pub params: Vec<f64>,
}

impl Spec {
    /// `def` with its default parameters.
    pub fn of(def: &Def) -> Self {
        Spec { code: def.code.into(), params: def.defaults() }
    }

    /// The indicator `code` (or an alias) with `params`, filled and clamped.
    pub fn new(code: &str, params: &[f64]) -> Option<Self> {
        let def = find(code)?;
        Some(Spec { code: def.code.into(), params: def.normalize(params) })
    }

    /// A command-line indicator: `head` names it and `rest` holds up to as many numbers as
    /// it has parameters (`BB 20 2.5`, `RSI`).
    pub fn parse(head: &str, rest: &[&str]) -> Option<Self> {
        let def = find(head)?;
        if rest.len() > def.params.len() {
            return None;
        }
        let params: Vec<f64> =
            rest.iter().map(|s| s.parse::<f64>().ok().filter(|v| v.is_finite())).collect::<Option<_>>()?;
        Some(Spec { code: def.code.into(), params: def.normalize(&params) })
    }

    /// Parse a stored label. Unknown codes are kept as they are, so a layout saved by a
    /// newer build survives a round trip through an older one.
    fn from_label(label: &str) -> Self {
        let mut words = label.split_whitespace();
        let head = words.next().unwrap_or_default();
        let params: Vec<f64> = words.filter_map(|w| w.parse().ok()).collect();
        Spec::new(head, &params).unwrap_or_else(|| Spec { code: head.to_ascii_uppercase(), params })
    }

    pub fn def(&self) -> Option<&'static Def> {
        find(&self.code)
    }

    /// The code and parameters as typed: `SMA 50`, `BB 20 2`.
    pub fn label(&self) -> String {
        let mut s = self.code.clone();
        for p in &self.params {
            let _ = write!(s, " {p}");
        }
        s
    }

    pub fn name(&self) -> &str {
        self.def().map_or(&self.code, |d| d.name)
    }

    pub fn is_pane(&self) -> bool {
        self.def().is_some_and(|d| d.pane)
    }

    /// Whether `other` takes this one's place on a chart: the same indicator, unless it
    /// allows several instances, in which case the same parameters too.
    pub fn same_kind(&self, other: &Spec) -> bool {
        self.code == other.code && (self.params == other.params || !self.def().is_some_and(|d| d.multi))
    }

    pub fn compute(&self, input: &Input<'_>) -> Vec<Output> {
        self.def().map(|d| d.run(input, &self.params)).unwrap_or_default()
    }
}

impl Serialize for Spec {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.label())
    }
}

impl<'de> Deserialize<'de> for Spec {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        /// The enum layouts were saved with before the catalog existed.
        #[derive(Deserialize)]
        enum Legacy {
            Sma(usize),
            Ema(usize),
            Vwap,
            Bollinger(usize, f64),
            Rsi(usize),
            Macd(usize, usize, usize),
        }
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            Label(String),
            Legacy(Legacy),
        }
        let spec = |code: &str, params: &[f64]| Spec::new(code, params).expect("legacy indicators are in the catalog");
        Ok(match Repr::deserialize(d)? {
            Repr::Label(s) => Spec::from_label(&s),
            Repr::Legacy(Legacy::Sma(n)) => spec("SMA", &[n as f64]),
            Repr::Legacy(Legacy::Ema(n)) => spec("EMA", &[n as f64]),
            Repr::Legacy(Legacy::Vwap) => spec("VWAP", &[]),
            Repr::Legacy(Legacy::Bollinger(n, k)) => spec("BB", &[n as f64, k]),
            Repr::Legacy(Legacy::Rsi(n)) => spec("RSI", &[n as f64]),
            Repr::Legacy(Legacy::Macd(a, b, c)) => spec("MACD", &[a as f64, b as f64, c as f64]),
        })
    }
}

fn n(p: &[f64], k: usize) -> usize {
    p[k] as usize
}

fn out(name: &'static str, values: Vec<f64>, plot: Plot, tone: Tone) -> Output {
    Output { name, values, plot, tone }
}

fn line(name: &'static str, values: Vec<f64>) -> Output {
    out(name, values, Plot::Line, Tone::Main)
}

fn one(name: &'static str, values: Vec<f64>) -> Vec<Output> {
    vec![line(name, values)]
}

fn signal(name: &'static str, values: Vec<f64>) -> Output {
    out(name, values, Plot::Line, Tone::Signal)
}

fn hist(name: &'static str, values: Vec<f64>) -> Vec<Output> {
    vec![out(name, values, Plot::Hist, Tone::Main)]
}

fn up(name: &'static str, values: Vec<f64>) -> Output {
    out(name, values, Plot::Line, Tone::Up)
}

fn down(name: &'static str, values: Vec<f64>) -> Output {
    out(name, values, Plot::Line, Tone::Down)
}

fn nth(name: &'static str, values: Vec<f64>, k: u8) -> Output {
    out(name, values, Plot::Line, Tone::Series(k))
}

fn bands(b: Bands, middle: &'static str) -> Vec<Output> {
    vec![
        out("Upper", b.upper, Plot::Line, Tone::Dim),
        line(middle, b.middle),
        out("Lower", b.lower, Plot::Line, Tone::Dim),
    ]
}

fn with_signal(name: &'static str, (main, sig): (Vec<f64>, Vec<f64>)) -> Vec<Output> {
    vec![line(name, main), signal("Signal", sig)]
}

/// WaveTrend drawn as filled waves (the signal line dark, the main line light over it) with
/// green and red dots where the main line crosses above or below the signal.
fn waves(d: &Ohlcv, n1: usize, n2: usize) -> Vec<Output> {
    let (wt1, wt2) = wavetrend(d, n1, n2);
    let (turn_up, turn_down) = crosses(&wt1, &wt2);
    vec![
        out("Wave 2", wt2, Plot::Fill, Tone::Fixed([36, 86, 214, 235])),
        out("Wave 1", wt1, Plot::Fill, Tone::Fixed([96, 176, 255, 150])),
        out("Turn up", turn_up, Plot::Marks, Tone::Up),
        out("Turn down", turn_down, Plot::Marks, Tone::Down),
    ]
}

/// A 0..100 oscillator moved onto WaveTrend's scale: 50 becomes 0 and the range ±100, so
/// 70 and 30 sit at ±40.
fn centred(values: Vec<f64>) -> Vec<f64> {
    values.into_iter().map(|v| (v - 50.0) * 2.0).collect()
}

fn macd_like(m: Macd, name: &'static str) -> Vec<Output> {
    vec![line(name, m.macd), signal("Signal", m.signal), out("Histogram", m.histogram, Plot::Hist, Tone::Main)]
}

fn trail(t: Trail, plot: Plot) -> Vec<Output> {
    let (u, d) = t.split();
    vec![out("Up", u, plot, Tone::Up), out("Down", d, plot, Tone::Down)]
}

fn pivot_levels(i: &Input<'_>, kind: PivotKind) -> Vec<Output> {
    pivots(&i.d, session_key(i.tf), kind)
        .into_iter()
        .map(|(name, v)| {
            let tone = match name.as_bytes()[0] {
                b'R' => Tone::Down,
                b'S' => Tone::Up,
                _ => Tone::Main,
            };
            out(name, v, Plot::Steps, tone)
        })
        .collect()
}

const GMMA: [(usize, &str); 12] = [
    (3, "EMA 3"),
    (5, "EMA 5"),
    (8, "EMA 8"),
    (10, "EMA 10"),
    (12, "EMA 12"),
    (15, "EMA 15"),
    (30, "EMA 30"),
    (35, "EMA 35"),
    (40, "EMA 40"),
    (45, "EMA 45"),
    (50, "EMA 50"),
    (60, "EMA 60"),
];

fn catalog() -> Vec<Def> {
    let ma = Category::MovingAverages;
    let bnd = Category::Bands;
    let mom = Category::Momentum;
    let trd = Category::Trend;
    let vol = Category::Volatility;
    let vlm = Category::Volume;
    let sta = Category::Statistics;
    let lvl = Category::Levels;
    vec![
        // Moving averages.
        overlay("SMA", "Simple Moving Average", ma, |i, p| one("SMA", sma(i.c(), n(p, 0)))).n("Length", 20.0).multi(),
        overlay("EMA", "Exponential Moving Average", ma, |i, p| one("EMA", ema(i.c(), n(p, 0))))
            .n("Length", 20.0)
            .multi(),
        overlay("WMA", "Weighted Moving Average", ma, |i, p| one("WMA", wma(i.c(), n(p, 0))))
            .alias(&["LWMA"])
            .n("Length", 20.0)
            .multi(),
        overlay("RMA", "Smoothed Moving Average (Wilder)", ma, |i, p| one("RMA", rma(i.c(), n(p, 0))))
            .alias(&["SMMA", "WILDERS"])
            .n("Length", 14.0)
            .multi(),
        overlay("HMA", "Hull Moving Average", ma, |i, p| one("HMA", hma(i.c(), n(p, 0))))
            .alias(&["HULL"])
            .n("Length", 9.0)
            .multi(),
        overlay("DEMA", "Double Exponential Moving Average", ma, |i, p| one("DEMA", dema(i.c(), n(p, 0))))
            .n("Length", 20.0)
            .multi(),
        overlay("TEMA", "Triple Exponential Moving Average", ma, |i, p| one("TEMA", tema(i.c(), n(p, 0))))
            .n("Length", 20.0)
            .multi(),
        overlay("TRIMA", "Triangular Moving Average", ma, |i, p| one("TRIMA", trima(i.c(), n(p, 0))))
            .alias(&["TMA"])
            .n("Length", 20.0)
            .multi(),
        overlay("KAMA", "Kaufman Adaptive Moving Average", ma, |i, p| {
            one("KAMA", kama(i.c(), n(p, 0), n(p, 1), n(p, 2)))
        })
        .alias(&["KAUFMAN"])
        .n("Length", 10.0)
        .n("Fast", 2.0)
        .n("Slow", 30.0)
        .multi(),
        overlay("ZLEMA", "Zero-Lag Exponential Moving Average", ma, |i, p| one("ZLEMA", zlema(i.c(), n(p, 0))))
            .alias(&["ZLMA"])
            .n("Length", 20.0)
            .multi(),
        overlay("ALMA", "Arnaud Legoux Moving Average", ma, |i, p| one("ALMA", alma(i.c(), n(p, 0), p[1], p[2])))
            .n("Length", 9.0)
            .real("Offset", 0.85, 0.0, 1.0)
            .real("Sigma", 6.0, 0.1, 100.0)
            .multi(),
        overlay("VWMA", "Volume-Weighted Moving Average", ma, |i, p| one("VWMA", vwma(i.c(), &i.d.volume, n(p, 0))))
            .n("Length", 20.0)
            .multi(),
        overlay("EVWMA", "Elastic Volume-Weighted Moving Average", ma, |i, p| {
            one("EVWMA", evwma(i.c(), &i.d.volume, n(p, 0)))
        })
        .n("Length", 20.0)
        .multi(),
        overlay("T3", "Tillson T3 Moving Average", ma, |i, p| one("T3", t3(i.c(), n(p, 0), p[1])))
            .alias(&["TILLSON"])
            .n("Length", 5.0)
            .real("Volume Factor", 0.7, 0.0, 1.0)
            .multi(),
        overlay("MCGINLEY", "McGinley Dynamic", ma, |i, p| one("MCGINLEY", mcginley(i.c(), n(p, 0))))
            .alias(&["MCG"])
            .n("Length", 14.0)
            .multi(),
        overlay("VIDYA", "Variable Index Dynamic Average", ma, |i, p| one("VIDYA", vidya(i.c(), n(p, 0), n(p, 1))))
            .n("Length", 14.0)
            .n("CMO Length", 9.0)
            .multi(),
        overlay("FRAMA", "Fractal Adaptive Moving Average", ma, |i, p| one("FRAMA", frama(&i.d, n(p, 0))))
            .int("Length", 16.0, 2.0, MAX_LENGTH)
            .multi(),
        overlay("SUPERSMOOTHER", "Ehlers Super Smoother", ma, |i, p| one("SSF", super_smoother(i.c(), n(p, 0))))
            .alias(&["SSF", "SUPERSMOOTH"])
            .n("Length", 10.0)
            .multi(),
        overlay("DECYCLER", "Ehlers Simple Decycler", ma, |i, p| one("Decycler", decycler(i.c(), n(p, 0))))
            .int("Cutoff", 125.0, 3.0, MAX_LENGTH)
            .multi(),
        overlay("MAMA", "MESA Adaptive Moving Average", ma, |i, p| {
            let m = mama(i.c(), p[0], p[1]);
            vec![line("MAMA", m.mama), signal("FAMA", m.fama)]
        })
        .alias(&["MESA", "FAMA"])
        .real("Fast Limit", 0.5, 0.01, 0.99)
        .real("Slow Limit", 0.05, 0.01, 0.99),
        overlay("ITREND", "Ehlers Instantaneous Trendline", ma, |i, p| with_signal("Trend", itrend(i.c(), p[0])))
            .alias(&["INSTTREND"])
            .real("Alpha", 0.07, 0.01, 0.99),
        overlay("LSMA", "Least Squares Moving Average", ma, |i, p| one("LSMA", linreg(i.c(), n(p, 0))))
            .alias(&["LINREG"])
            .int("Length", 25.0, 2.0, MAX_LENGTH)
            .multi(),
        overlay("GMMA", "Guppy Multiple Moving Average", ma, |i, _| {
            GMMA.iter()
                .enumerate()
                .map(|(k, &(len, name))| {
                    let values = ema(i.c(), len);
                    if k < 6 { up(name, values) } else { down(name, values) }
                })
                .collect()
        })
        .alias(&["GUPPY"]),
        // Bands and channels.
        overlay("BB", "Bollinger Bands", bnd, |i, p| bands(bollinger(i.c(), n(p, 0), p[1]), "Basis"))
            .alias(&["BOLL", "BOLLINGER", "BBANDS"])
            .n("Length", 20.0)
            .real("StdDev", 2.0, 0.1, 10.0),
        overlay("KC", "Keltner Channels", bnd, |i, p| bands(keltner(&i.d, n(p, 0), p[1], n(p, 2)), "Basis"))
            .alias(&["KELTNER", "KELT"])
            .n("Length", 20.0)
            .real("Multiplier", 2.0, 0.1, 10.0)
            .n("ATR Length", 10.0),
        overlay("DONCHIAN", "Donchian Channels", bnd, |i, p| bands(donchian(&i.d, n(p, 0)), "Basis"))
            .alias(&["DC", "DONCH", "PRICECHANNEL"])
            .n("Length", 20.0),
        overlay("ENV", "Moving Average Envelopes", bnd, |i, p| bands(envelopes(i.c(), n(p, 0), p[1]), "Basis"))
            .alias(&["ENVELOPE", "ENVELOPES"])
            .n("Length", 20.0)
            .real("Percent", 2.5, 0.01, 50.0),
        overlay("ACCBANDS", "Acceleration Bands", bnd, |i, p| bands(acceleration_bands(&i.d, n(p, 0), p[1]), "Basis"))
            .alias(&["ABANDS"])
            .n("Length", 20.0)
            .real("Factor", 4.0, 0.1, 20.0),
        overlay("STARC", "STARC Bands", bnd, |i, p| bands(starc(&i.d, n(p, 0), n(p, 1), p[2]), "Basis"))
            .n("SMA Length", 6.0)
            .n("ATR Length", 15.0)
            .real("Multiplier", 2.0, 0.1, 10.0),
        overlay("SEB", "Standard Error Bands", bnd, |i, p| {
            bands(standard_error_bands(i.c(), n(p, 0), n(p, 1), p[2]), "Regression")
        })
        .alias(&["STDERRBANDS"])
        .int("Length", 21.0, 3.0, MAX_LENGTH)
        .n("Smoothing", 3.0)
        .real("StdErr", 2.0, 0.1, 10.0),
        overlay("VWAPB", "VWAP Bands", bnd, |i, p| bands(vwap_bands(&i.d, session_key(i.tf), p[0]), "VWAP"))
            .alias(&["VWAPBANDS"])
            .real("StdDev", 1.0, 0.1, 10.0),
        // Momentum.
        pane("RSI", "Relative Strength Index", mom, |i, p| one("RSI", rsi(i.c(), n(p, 0))))
            .n("Length", 14.0)
            .range(0.0, 100.0)
            .levels(&[30.0, 70.0]),
        pane("MACD", "Moving Average Convergence Divergence", mom, |i, p| {
            macd_like(macd(i.c(), n(p, 0), n(p, 1), n(p, 2)), "MACD")
        })
        .n("Fast", 12.0)
        .n("Slow", 26.0)
        .n("Signal", 9.0)
        .levels(&[0.0]),
        pane("STOCH", "Stochastic", mom, |i, p| {
            let s = stochastic(&i.d, n(p, 0), n(p, 1), n(p, 2));
            vec![line("%K", s.k), signal("%D", s.d)]
        })
        .alias(&["STO", "STOCHASTIC", "KD"])
        .n("%K Length", 14.0)
        .n("%K Smoothing", 1.0)
        .n("%D Smoothing", 3.0)
        .range(0.0, 100.0)
        .levels(&[20.0, 80.0]),
        pane("STOCHRSI", "Stochastic RSI", mom, |i, p| {
            let s = stoch_rsi(i.c(), n(p, 0), n(p, 1), n(p, 2), n(p, 3));
            vec![line("%K", s.k), signal("%D", s.d)]
        })
        .alias(&["SRSI"])
        .n("RSI Length", 14.0)
        .n("Stoch Length", 14.0)
        .n("%K Smoothing", 3.0)
        .n("%D Smoothing", 3.0)
        .range(0.0, 100.0)
        .levels(&[20.0, 80.0]),
        pane("CCI", "Commodity Channel Index", mom, |i, p| one("CCI", cci(&i.d.hlc3(), n(p, 0))))
            .n("Length", 20.0)
            .levels(&[-100.0, 100.0]),
        pane("WILLR", "Williams %R", mom, |i, p| one("%R", williams_r(&i.d, n(p, 0))))
            .alias(&["WPR", "%R", "WR", "WILLIAMS"])
            .n("Length", 14.0)
            .range(-100.0, 0.0)
            .levels(&[-80.0, -20.0]),
        pane("UO", "Ultimate Oscillator", mom, |i, p| one("UO", ultimate(&i.d, n(p, 0), n(p, 1), n(p, 2))))
            .alias(&["ULT", "ULTOSC", "ULTIMATE"])
            .n("Fast", 7.0)
            .n("Middle", 14.0)
            .n("Slow", 28.0)
            .range(0.0, 100.0)
            .levels(&[30.0, 70.0]),
        pane("AO", "Awesome Oscillator", mom, |i, p| hist("AO", awesome(&i.d, n(p, 0), n(p, 1))))
            .alias(&["AWESOME"])
            .n("Fast", 5.0)
            .n("Slow", 34.0)
            .levels(&[0.0]),
        pane("AC", "Accelerator Oscillator", mom, |i, p| hist("AC", accelerator(&i.d, n(p, 0), n(p, 1), n(p, 2))))
            .alias(&["ACCEL", "ACCELERATOR"])
            .n("Fast", 5.0)
            .n("Slow", 34.0)
            .n("Signal", 5.0)
            .levels(&[0.0]),
        pane("MOM", "Momentum", mom, |i, p| one("MOM", momentum(i.c(), n(p, 0))))
            .alias(&["MOMENTUM"])
            .n("Length", 10.0)
            .levels(&[0.0]),
        pane("ROC", "Rate of Change", mom, |i, p| one("ROC", roc(i.c(), n(p, 0)))).n("Length", 9.0).levels(&[0.0]),
        pane("PPO", "Percentage Price Oscillator", mom, |i, p| macd_like(ppo(i.c(), n(p, 0), n(p, 1), n(p, 2)), "PPO"))
            .n("Fast", 12.0)
            .n("Slow", 26.0)
            .n("Signal", 9.0)
            .levels(&[0.0]),
        pane("APO", "Absolute Price Oscillator", mom, |i, p| one("APO", apo(i.c(), n(p, 0), n(p, 1))))
            .alias(&["PO"])
            .n("Fast", 12.0)
            .n("Slow", 26.0)
            .levels(&[0.0]),
        pane("TSI", "True Strength Index", mom, |i, p| {
            let t = tsi(i.c(), n(p, 0), n(p, 1));
            let s = ema(&t, n(p, 2));
            vec![line("TSI", t), signal("Signal", s)]
        })
        .n("Long", 25.0)
        .n("Short", 13.0)
        .n("Signal", 13.0)
        .levels(&[0.0]),
        pane("TRIX", "TRIX", mom, |i, p| {
            let t = trix(i.c(), n(p, 0));
            let s = ema(&t, n(p, 1));
            vec![line("TRIX", t), signal("Signal", s)]
        })
        .n("Length", 18.0)
        .n("Signal", 9.0)
        .levels(&[0.0]),
        pane("KST", "Know Sure Thing", mom, |i, p| {
            let k = kst(i.c(), [n(p, 0), n(p, 1), n(p, 2), n(p, 3)], [n(p, 4), n(p, 5), n(p, 6), n(p, 7)], n(p, 8));
            vec![line("KST", k.macd), signal("Signal", k.signal)]
        })
        .alias(&["PRING"])
        .n("ROC 1", 10.0)
        .n("ROC 2", 15.0)
        .n("ROC 3", 20.0)
        .n("ROC 4", 30.0)
        .n("SMA 1", 10.0)
        .n("SMA 2", 10.0)
        .n("SMA 3", 10.0)
        .n("SMA 4", 15.0)
        .n("Signal", 9.0)
        .levels(&[0.0]),
        pane("CMO", "Chande Momentum Oscillator", mom, |i, p| one("CMO", cmo(i.c(), n(p, 0))))
            .alias(&["CHANDE"])
            .n("Length", 9.0)
            .range(-100.0, 100.0)
            .levels(&[-50.0, 50.0]),
        pane("DPO", "Detrended Price Oscillator", mom, |i, p| one("DPO", dpo(i.c(), n(p, 0))))
            .n("Length", 21.0)
            .levels(&[0.0]),
        pane("COPPOCK", "Coppock Curve", mom, |i, p| one("Coppock", coppock(i.c(), n(p, 0), n(p, 1), n(p, 2))))
            .alias(&["COPP"])
            .n("WMA Length", 10.0)
            .n("Long ROC", 14.0)
            .n("Short ROC", 11.0)
            .levels(&[0.0]),
        pane("RVGI", "Relative Vigor Index", mom, |i, p| {
            let r = rvgi(&i.d, n(p, 0));
            vec![line("RVGI", r.macd), signal("Signal", r.signal)]
        })
        .alias(&["VIGOR"])
        .n("Length", 10.0)
        .levels(&[0.0]),
        pane("FISHER", "Fisher Transform", mom, |i, p| with_signal("Fisher", fisher(&i.d, n(p, 0))))
            .alias(&["FISH"])
            .n("Length", 9.0)
            .levels(&[-1.5, 0.0, 1.5]),
        pane("STOCHMI", "Stochastic Momentum Index", mom, |i, p| {
            let s = smi(&i.d, n(p, 0), n(p, 1), n(p, 2));
            vec![line("SMI", s.macd), signal("Signal", s.signal)]
        })
        .alias(&["STOCHMOMENTUM"])
        .n("%K Length", 10.0)
        .n("%D Length", 3.0)
        .n("Signal", 10.0)
        .range(-100.0, 100.0)
        .levels(&[-40.0, 40.0]),
        pane("SMIERGO", "SMI Ergodic", mom, |i, p| macd_like(smi_ergodic(i.c(), n(p, 0), n(p, 1), n(p, 2)), "SMI"))
            .alias(&["SMII", "ERGODIC"])
            .n("Short", 5.0)
            .n("Long", 20.0)
            .n("Signal", 5.0)
            .levels(&[0.0]),
        pane("CRSI", "Connors RSI", mom, |i, p| one("CRSI", connors_rsi(i.c(), n(p, 0), n(p, 1), n(p, 2))))
            .alias(&["CONNORS", "CONNORSRSI"])
            .n("RSI Length", 3.0)
            .n("Streak Length", 2.0)
            .n("Rank Length", 100.0)
            .range(0.0, 100.0)
            .levels(&[10.0, 90.0]),
        pane("LRSI", "Laguerre RSI", mom, |i, p| one("LRSI", laguerre_rsi(i.c(), p[0])))
            .alias(&["LAGUERRE"])
            .real("Gamma", 0.5, 0.0, 0.99)
            .range(0.0, 1.0)
            .levels(&[0.2, 0.8]),
        pane("RMI", "Relative Momentum Index", mom, |i, p| one("RMI", rmi(i.c(), n(p, 0), n(p, 1))))
            .n("Length", 20.0)
            .n("Momentum", 5.0)
            .range(0.0, 100.0)
            .levels(&[30.0, 70.0]),
        pane("PMO", "Price Momentum Oscillator", mom, |i, p| {
            let m = pmo(i.c(), n(p, 0), n(p, 1), n(p, 2));
            vec![line("PMO", m.macd), signal("Signal", m.signal)]
        })
        .int("First Smoothing", 35.0, 2.0, MAX_LENGTH)
        .int("Second Smoothing", 20.0, 2.0, MAX_LENGTH)
        .n("Signal", 10.0)
        .levels(&[0.0]),
        pane("BOP", "Balance of Power", mom, |i, p| one("BOP", bop(&i.d, n(p, 0))))
            .n("Smoothing", 14.0)
            .range(-1.0, 1.0)
            .levels(&[0.0]),
        pane("ELDER", "Elder Ray Index", mom, |i, p| {
            let (bull, bear) = elder_ray(&i.d, n(p, 0));
            vec![up("Bull Power", bull), down("Bear Power", bear)]
        })
        .alias(&["ELDERRAY", "ERAY"])
        .n("Length", 13.0)
        .levels(&[0.0]),
        pane("BBP", "Bull Bear Power", mom, |i, p| hist("BBP", bull_bear_power(&i.d, n(p, 0))))
            .alias(&["BULLBEAR"])
            .n("Length", 13.0)
            .levels(&[0.0]),
        pane("QSTICK", "Qstick", mom, |i, p| one("Qstick", qstick(&i.d, n(p, 0)))).n("Length", 8.0).levels(&[0.0]),
        pane("PSY", "Psychological Line", mom, |i, p| one("PSY", psychological(i.c(), n(p, 0))))
            .alias(&["PSYCH", "PSYCHOLOGICAL"])
            .n("Length", 12.0)
            .range(0.0, 100.0)
            .levels(&[25.0, 75.0]),
        pane("CFO", "Chande Forecast Oscillator", mom, |i, p| one("CFO", cfo(i.c(), n(p, 0))))
            .int("Length", 14.0, 2.0, MAX_LENGTH)
            .levels(&[0.0]),
        pane("IMI", "Intraday Momentum Index", mom, |i, p| one("IMI", imi(&i.d, n(p, 0))))
            .n("Length", 14.0)
            .range(0.0, 100.0)
            .levels(&[30.0, 70.0]),
        pane("STC", "Schaff Trend Cycle", mom, |i, p| one("STC", stc(i.c(), n(p, 0), n(p, 1), n(p, 2))))
            .alias(&["SCHAFF"])
            .n("Cycle", 10.0)
            .n("Fast", 23.0)
            .n("Slow", 50.0)
            .range(0.0, 100.0)
            .levels(&[25.0, 75.0]),
        pane("COG", "Center of Gravity", mom, |i, p| with_signal("COG", center_of_gravity(i.c(), n(p, 0))))
            .alias(&["CENTERGRAVITY"])
            .n("Length", 10.0),
        pane("IFTRSI", "Inverse Fisher Transform RSI", mom, |i, p| one("IFT RSI", ift_rsi(i.c(), n(p, 0), n(p, 1))))
            .alias(&["IFT"])
            .n("RSI Length", 5.0)
            .n("WMA Length", 9.0)
            .range(-1.0, 1.0)
            .levels(&[-0.5, 0.5]),
        pane("PGO", "Pretty Good Oscillator", mom, |i, p| one("PGO", pgo(&i.d, n(p, 0))))
            .n("Length", 14.0)
            .levels(&[-3.0, 3.0]),
        pane("QQE", "Quantitative Qualitative Estimation", mom, |i, p| {
            with_signal("QQE", qqe(i.c(), n(p, 0), n(p, 1), p[2]))
        })
        .n("RSI Length", 14.0)
        .n("Smoothing", 5.0)
        .real("Factor", 4.236, 0.1, 20.0)
        .levels(&[50.0]),
        pane("WT", "WaveTrend", mom, |i, p| with_signal("WT1", wavetrend(&i.d, n(p, 0), n(p, 1))))
            .alias(&["WAVETREND", "WTO"])
            .n("Channel Length", 10.0)
            .n("Average Length", 21.0)
            .levels(&[-60.0, 0.0, 60.0]),
        pane("WTX", "WaveTrend with Crosses", mom, |i, p| waves(&i.d, n(p, 0), n(p, 1)))
            .alias(&["WTCROSS", "WTC"])
            .n("Channel Length", 10.0)
            .n("Average Length", 21.0)
            .levels(&[-60.0, 0.0, 60.0]),
        pane("WAVES", "Waves (WaveTrend with RSI and Stoch RSI)", mom, |i, p| {
            let mut outs = waves(&i.d, n(p, 0), n(p, 1));
            let s = stoch_rsi(i.c(), n(p, 2), n(p, 3), 3, 3);
            // Between the waves and the cross dots, so the dots stay on top.
            outs.splice(
                2..2,
                [
                    out("RSI", centred(rsi(i.c(), n(p, 2))), Plot::Line, Tone::Fixed([186, 104, 255, 255])),
                    out("Stoch RSI", centred(s.k), Plot::Line, Tone::Fixed([38, 214, 220, 255])),
                ],
            );
            outs
        })
        .alias(&["CIPHER", "WAVE"])
        .n("Channel Length", 9.0)
        .n("Average Length", 12.0)
        .n("RSI Length", 14.0)
        .n("Stoch Length", 14.0)
        .levels(&[-60.0, 0.0, 60.0]),
        pane("CYBER", "Ehlers Cyber Cycle", mom, |i, p| with_signal("Cycle", cyber_cycle(i.c(), p[0])))
            .alias(&["CYBERCYCLE"])
            .real("Alpha", 0.07, 0.01, 0.99)
            .levels(&[0.0]),
        pane("DISP", "Disparity Index", mom, |i, p| one("Disparity", disparity(i.c(), n(p, 0))))
            .alias(&["DISPARITY"])
            .n("Length", 14.0)
            .levels(&[0.0]),
        pane("PFE", "Polarized Fractal Efficiency", mom, |i, p| one("PFE", pfe(i.c(), n(p, 0), n(p, 1))))
            .n("Length", 10.0)
            .n("Smoothing", 5.0)
            .range(-100.0, 100.0)
            .levels(&[0.0]),
        // Trend.
        pane("ADX", "Average Directional Index", trd, |i, p| one("ADX", dmi(&i.d, n(p, 0), n(p, 1)).adx))
            .n("DI Length", 14.0)
            .n("ADX Smoothing", 14.0)
            .range(0.0, 100.0)
            .levels(&[25.0]),
        pane("DMI", "Directional Movement Index", trd, |i, p| {
            let d = dmi(&i.d, n(p, 0), n(p, 1));
            vec![up("+DI", d.plus), down("-DI", d.minus), line("ADX", d.adx)]
        })
        .alias(&["DX"])
        .n("DI Length", 14.0)
        .n("ADX Smoothing", 14.0)
        .range(0.0, 100.0)
        .levels(&[25.0]),
        pane("ADXR", "Average Directional Movement Rating", trd, |i, p| {
            let adx = dmi(&i.d, n(p, 0), n(p, 1)).adx;
            let back = n(p, 0).saturating_sub(1);
            let adxr = (0..adx.len()).map(|k| (adx[k] + ago(&adx, k, back)) / 2.0).collect();
            vec![line("ADXR", adxr), out("ADX", adx, Plot::Line, Tone::Dim)]
        })
        .n("DI Length", 14.0)
        .n("ADX Smoothing", 14.0)
        .range(0.0, 100.0)
        .levels(&[25.0]),
        pane("AROON", "Aroon", trd, |i, p| {
            let (u, d) = aroon(&i.d, n(p, 0));
            vec![up("Aroon Up", u), down("Aroon Down", d)]
        })
        .n("Length", 14.0)
        .range(0.0, 100.0)
        .levels(&[30.0, 70.0]),
        pane("AROONOSC", "Aroon Oscillator", trd, |i, p| {
            let (u, d) = aroon(&i.d, n(p, 0));
            one("Aroon Osc", zip(&u, &d, |a, b| a - b))
        })
        .n("Length", 14.0)
        .range(-100.0, 100.0)
        .levels(&[0.0]),
        pane("VORTEX", "Vortex Indicator", trd, |i, p| {
            let (plus, minus) = vortex(&i.d, n(p, 0));
            vec![up("VI+", plus), down("VI-", minus)]
        })
        .alias(&["VI"])
        .int("Length", 14.0, 2.0, MAX_LENGTH)
        .levels(&[1.0]),
        overlay("PSAR", "Parabolic SAR", trd, |i, p| trail(psar(&i.d, p[0], p[1], p[2]), Plot::Dots))
            .alias(&["SAR", "PARABOLIC"])
            .real("Start", 0.02, 0.001, 1.0)
            .real("Increment", 0.02, 0.001, 1.0)
            .real("Maximum", 0.2, 0.001, 1.0),
        overlay("SUPERTREND", "SuperTrend", trd, |i, p| trail(supertrend(&i.d, n(p, 0), p[1]), Plot::Line))
            .alias(&["STREND"])
            .n("ATR Length", 10.0)
            .real("Factor", 3.0, 0.1, 20.0),
        overlay("VSTOP", "Volatility Stop", trd, |i, p| trail(volatility_stop(&i.d, n(p, 0), p[1]), Plot::Dots))
            .alias(&["VOLSTOP"])
            .n("ATR Length", 20.0)
            .real("Multiplier", 2.0, 0.1, 20.0),
        overlay("GANNHILO", "Gann HiLo Activator", trd, |i, p| trail(gann_hilo(&i.d, n(p, 0), n(p, 1)), Plot::Line))
            .alias(&["HILO", "GANN"])
            .n("High Length", 13.0)
            .n("Low Length", 21.0),
        overlay("ICHIMOKU", "Ichimoku Cloud", trd, |i, p| {
            let k = ichimoku(&i.d, n(p, 0), n(p, 1), n(p, 2), n(p, 3));
            vec![
                nth("Conversion", k.conversion, 0),
                nth("Base", k.base, 1),
                nth("Lagging", k.lagging, 2),
                up("Leading A", k.span_a),
                down("Leading B", k.span_b),
            ]
        })
        .alias(&["ICHI", "CLOUD", "KUMO"])
        .n("Conversion", 9.0)
        .n("Base", 26.0)
        .n("Span B", 52.0)
        .n("Displacement", 26.0),
        overlay("ALLIGATOR", "Williams Alligator", trd, |i, p| {
            let a = alligator(&i.d, [n(p, 0), n(p, 1), n(p, 2)], [n(p, 3), n(p, 4), n(p, 5)]);
            vec![nth("Jaw", a.jaw, 0), down("Teeth", a.teeth), up("Lips", a.lips)]
        })
        .alias(&["GATORLINES"])
        .n("Jaw", 13.0)
        .n("Teeth", 8.0)
        .n("Lips", 5.0)
        .int("Jaw Offset", 8.0, 0.0, 100.0)
        .int("Teeth Offset", 5.0, 0.0, 100.0)
        .int("Lips Offset", 3.0, 0.0, 100.0),
        pane("GATOR", "Gator Oscillator", trd, |i, p| {
            let (upper, lower) = gator(&i.d, [n(p, 0), n(p, 1), n(p, 2)], [n(p, 3), n(p, 4), n(p, 5)]);
            vec![out("Upper", upper, Plot::Hist, Tone::Up), out("Lower", lower, Plot::Hist, Tone::Down)]
        })
        .n("Jaw", 13.0)
        .n("Teeth", 8.0)
        .n("Lips", 5.0)
        .int("Jaw Offset", 8.0, 0.0, 100.0)
        .int("Teeth Offset", 5.0, 0.0, 100.0)
        .int("Lips Offset", 3.0, 0.0, 100.0)
        .levels(&[0.0]),
        overlay("FRACTALS", "Williams Fractals", trd, |i, p| {
            let (high, low) = fractals(&i.d, n(p, 0));
            vec![out("High", high, Plot::Dots, Tone::Down), out("Low", low, Plot::Dots, Tone::Up)]
        })
        .alias(&["FRACTAL"])
        .int("Periods", 2.0, 1.0, 50.0),
        overlay("ZIGZAG", "ZigZag", trd, |i, p| one("ZigZag", zigzag(&i.d, p[0]))).alias(&["ZZ"]).real(
            "Deviation %",
            5.0,
            0.1,
            50.0,
        ),
        pane("CHOP", "Choppiness Index", trd, |i, p| one("CHOP", choppiness(&i.d, n(p, 0))))
            .alias(&["CHOPPINESS"])
            .int("Length", 14.0, 2.0, MAX_LENGTH)
            .levels(&[38.2, 61.8]),
        pane("VHF", "Vertical Horizontal Filter", trd, |i, p| one("VHF", vhf(i.c(), n(p, 0)))).n("Length", 28.0),
        pane("RWI", "Random Walk Index", trd, |i, p| {
            let (high, low) = rwi(&i.d, n(p, 0));
            vec![up("RWI High", high), down("RWI Low", low)]
        })
        .n("Length", 14.0)
        .levels(&[1.0]),
        pane("TII", "Trend Intensity Index", trd, |i, p| one("TII", tii(i.c(), n(p, 0), n(p, 1))))
            .n("MA Length", 60.0)
            .n("Length", 30.0)
            .range(0.0, 100.0)
            .levels(&[20.0, 80.0]),
        pane("ER", "Efficiency Ratio", trd, |i, p| one("ER", efficiency_ratio(i.c(), n(p, 0))))
            .alias(&["KER", "EFFICIENCY"])
            .n("Length", 10.0)
            .range(0.0, 1.0),
        overlay("CKSTOP", "Chande Kroll Stop", trd, |i, p| {
            let (short, long) = chande_kroll(&i.d, n(p, 0), p[1], n(p, 2));
            vec![down("Stop Short", short), up("Stop Long", long)]
        })
        .alias(&["CHANDEKROLL", "CKS"])
        .n("ATR Length", 10.0)
        .real("Multiplier", 1.0, 0.1, 20.0)
        .n("Stop Length", 9.0),
        overlay("CHANDELIER", "Chandelier Exit", trd, |i, p| {
            let (long, short) = chandelier(&i.d, n(p, 0), p[1]);
            vec![up("Long Exit", long), down("Short Exit", short)]
        })
        .alias(&["CE", "CHEXIT"])
        .n("Length", 22.0)
        .real("Multiplier", 3.0, 0.1, 20.0),
        // Volatility.
        pane("ATR", "Average True Range", vol, |i, p| one("ATR", atr(&i.d, n(p, 0)))).n("Length", 14.0),
        pane("NATR", "Normalized Average True Range", vol, |i, p| one("NATR", natr(&i.d, n(p, 0)))).n("Length", 14.0),
        pane("TRANGE", "True Range", vol, |i, _| one("TR", true_range(&i.d))).alias(&["TRUERANGE"]),
        pane("ADR", "Average Day Range", vol, |i, p| one("ADR", sma(&zip(&i.d.high, &i.d.low, |h, l| h - l), n(p, 0))))
            .n("Length", 14.0),
        pane("BBW", "Bollinger BandWidth", vol, |i, p| one("BBW", bandwidth(i.c(), n(p, 0), p[1])))
            .alias(&["BANDWIDTH"])
            .n("Length", 20.0)
            .real("StdDev", 2.0, 0.1, 10.0),
        pane("PCTB", "Bollinger %B", vol, |i, p| one("%B", percent_b(i.c(), n(p, 0), p[1])))
            .alias(&["%B", "PERCENTB", "BBPCT"])
            .n("Length", 20.0)
            .real("StdDev", 2.0, 0.1, 10.0)
            .levels(&[0.0, 1.0]),
        pane("HV", "Historical Volatility", vol, |i, p| {
            one("HV", historical_volatility(i.c(), n(p, 0), periods_per_year(i.tf)))
        })
        .alias(&["HVOL", "HISTVOL"])
        .int("Length", 20.0, 2.0, MAX_LENGTH),
        pane("PARKINSON", "Parkinson Volatility", vol, |i, p| {
            one("Parkinson", parkinson(&i.d, n(p, 0), periods_per_year(i.tf)))
        })
        .alias(&["PKV"])
        .n("Length", 20.0),
        pane("GK", "Garman-Klass Volatility", vol, |i, p| {
            one("Garman-Klass", garman_klass(&i.d, n(p, 0), periods_per_year(i.tf)))
        })
        .alias(&["GARMANKLASS"])
        .n("Length", 20.0),
        pane("RSVOL", "Rogers-Satchell Volatility", vol, |i, p| {
            one("Rogers-Satchell", rogers_satchell(&i.d, n(p, 0), periods_per_year(i.tf)))
        })
        .alias(&["ROGERSSATCHELL"])
        .n("Length", 20.0),
        pane("YZ", "Yang-Zhang Volatility", vol, |i, p| {
            one("Yang-Zhang", yang_zhang(&i.d, n(p, 0), periods_per_year(i.tf)))
        })
        .alias(&["YANGZHANG"])
        .int("Length", 20.0, 2.0, MAX_LENGTH),
        pane("ULCER", "Ulcer Index", vol, |i, p| one("Ulcer", ulcer(i.c(), n(p, 0)))).alias(&["UI"]).n("Length", 14.0),
        pane("CHV", "Chaikin Volatility", vol, |i, p| one("CHV", chaikin_volatility(&i.d, n(p, 0), n(p, 1))))
            .alias(&["CHAIKINVOL"])
            .n("EMA Length", 10.0)
            .n("ROC Length", 10.0)
            .levels(&[0.0]),
        pane("MASS", "Mass Index", vol, |i, p| one("Mass", mass_index(&i.d, n(p, 0), n(p, 1))))
            .alias(&["MI", "MASSINDEX"])
            .n("EMA Length", 9.0)
            .n("Sum Length", 25.0)
            .levels(&[26.5, 27.0]),
        pane("RVI", "Relative Volatility Index", vol, |i, p| {
            one("RVI", relative_volatility_index(i.c(), n(p, 0), n(p, 1)))
        })
        .n("StdDev Length", 10.0)
        .n("Smoothing", 14.0)
        .range(0.0, 100.0)
        .levels(&[50.0]),
        pane("SQUEEZE", "Squeeze Momentum", vol, |i, p| {
            let s = squeeze(&i.d, n(p, 0), p[1], n(p, 2), p[3]);
            let dots =
                |flags: Vec<bool>| -> Vec<f64> { flags.into_iter().map(|f| if f { 0.0 } else { NAN }).collect() };
            vec![
                out("Momentum", s.momentum, Plot::Hist, Tone::Main),
                out("Squeeze On", dots(s.on), Plot::Dots, Tone::Down),
                out("Squeeze Off", dots(s.off), Plot::Dots, Tone::Up),
            ]
        })
        .alias(&["SQZ", "SQZMOM", "TTM"])
        .int("BB Length", 20.0, 2.0, MAX_LENGTH)
        .real("BB StdDev", 2.0, 0.1, 10.0)
        .int("KC Length", 20.0, 2.0, MAX_LENGTH)
        .real("KC Multiplier", 1.5, 0.1, 10.0),
        // Volume.
        overlay("VWAP", "Volume-Weighted Average Price", vlm, |i, _| one("VWAP", vwap_by(i.bars, session_key(i.tf)))),
        pane("OBV", "On-Balance Volume", vlm, |i, _| one("OBV", obv(&i.d))),
        pane("AD", "Accumulation/Distribution", vlm, |i, _| one("A/D", adl(&i.d))).alias(&["ADL", "ACCDIST"]),
        pane("CMF", "Chaikin Money Flow", vlm, |i, p| one("CMF", cmf(&i.d, n(p, 0)))).n("Length", 20.0).levels(&[0.0]),
        pane("CHO", "Chaikin Oscillator", vlm, |i, p| one("CHO", chaikin_osc(&i.d, n(p, 0), n(p, 1))))
            .alias(&["CHAIKINOSC"])
            .n("Fast", 3.0)
            .n("Slow", 10.0)
            .levels(&[0.0]),
        pane("MFI", "Money Flow Index", vlm, |i, p| one("MFI", mfi(&i.d, n(p, 0))))
            .n("Length", 14.0)
            .range(0.0, 100.0)
            .levels(&[20.0, 80.0]),
        pane("EFI", "Elder Force Index", vlm, |i, p| one("EFI", force_index(&i.d, n(p, 0))))
            .alias(&["FORCE"])
            .n("Length", 13.0)
            .levels(&[0.0]),
        pane("EOM", "Ease of Movement", vlm, |i, p| one("EOM", ease_of_movement(&i.d, n(p, 0), p[1])))
            .alias(&["EMV", "EASE"])
            .n("Length", 14.0)
            .real("Divisor", 10_000.0, 1.0, 1e9)
            .levels(&[0.0]),
        pane("PVT", "Price Volume Trend", vlm, |i, _| one("PVT", pvt(&i.d))).alias(&["VPT"]),
        pane("NVI", "Negative Volume Index", vlm, |i, p| with_signal("NVI", nvi(&i.d, n(p, 0)))).n("Signal", 255.0),
        pane("PVI", "Positive Volume Index", vlm, |i, p| with_signal("PVI", pvi(&i.d, n(p, 0)))).n("Signal", 255.0),
        pane("KVO", "Klinger Volume Oscillator", vlm, |i, p| {
            let k = klinger(&i.d, n(p, 0), n(p, 1), n(p, 2));
            vec![line("KVO", k.macd), signal("Signal", k.signal)]
        })
        .alias(&["KLINGER"])
        .n("Fast", 34.0)
        .n("Slow", 55.0)
        .n("Signal", 13.0)
        .levels(&[0.0]),
        pane("VOSC", "Volume Oscillator", vlm, |i, p| one("VOSC", volume_osc(&i.d, n(p, 0), n(p, 1))))
            .alias(&["VOLOSC"])
            .n("Fast", 5.0)
            .n("Slow", 10.0)
            .levels(&[0.0]),
        pane("PVO", "Percentage Volume Oscillator", vlm, |i, p| macd_like(pvo(&i.d, n(p, 0), n(p, 1), n(p, 2)), "PVO"))
            .n("Fast", 12.0)
            .n("Slow", 26.0)
            .n("Signal", 9.0)
            .levels(&[0.0]),
        pane("VROC", "Volume Rate of Change", vlm, |i, p| one("VROC", vroc(&i.d, n(p, 0))))
            .n("Length", 14.0)
            .levels(&[0.0]),
        pane("RVOL", "Relative Volume", vlm, |i, p| one("RVOL", relative_volume(&i.d, n(p, 0))))
            .alias(&["RELVOL"])
            .n("Length", 20.0)
            .levels(&[1.0]),
        pane("TMF", "Twiggs Money Flow", vlm, |i, p| one("TMF", twiggs_mf(&i.d, n(p, 0))))
            .alias(&["TWIGGS"])
            .n("Length", 21.0)
            .levels(&[0.0]),
        pane("WAD", "Williams Accumulation/Distribution", vlm, |i, _| one("WAD", williams_ad(&i.d))).alias(&["WILLAD"]),
        pane("ASI", "Accumulative Swing Index", vlm, |i, p| one("ASI", asi(&i.d, p[0]))).alias(&["SWING"]).real(
            "Limit Move",
            10_000.0,
            0.01,
            1e9,
        ),
        pane("BWMFI", "Market Facilitation Index", vlm, |i, _| one("MFI", bw_mfi(&i.d))).alias(&["MFACIL"]),
        pane("NETVOL", "Net Volume", vlm, |i, _| hist("Net Volume", net_volume(&i.d))).levels(&[0.0]),
        pane("VZO", "Volume Zone Oscillator", vlm, |i, p| one("VZO", vzo(&i.d, n(p, 0))))
            .n("Length", 14.0)
            .range(-100.0, 100.0)
            .levels(&[-40.0, 0.0, 40.0]),
        pane("TSV", "Time Segmented Volume", vlm, |i, p| hist("TSV", tsv(&i.d, n(p, 0))))
            .n("Length", 13.0)
            .levels(&[0.0]),
        // Statistics.
        pane("STDEV", "Standard Deviation", sta, |i, p| one("StdDev", stdev(i.c(), n(p, 0))))
            .alias(&["SD", "STD"])
            .n("Length", 20.0),
        pane("VAR", "Variance", sta, |i, p| one("Variance", variance(i.c(), n(p, 0))))
            .alias(&["VARIANCE"])
            .n("Length", 20.0),
        pane("ZSCORE", "Z-Score", sta, |i, p| one("Z", zscore(i.c(), n(p, 0))))
            .int("Length", 20.0, 2.0, MAX_LENGTH)
            .levels(&[-2.0, 0.0, 2.0]),
        pane("SLOPE", "Linear Regression Slope", sta, |i, p| one("Slope", linreg_slope(i.c(), n(p, 0))))
            .alias(&["LRS"])
            .int("Length", 14.0, 2.0, MAX_LENGTH)
            .levels(&[0.0]),
        pane("R2", "R-Squared", sta, |i, p| one("R²", r_squared(i.c(), n(p, 0))))
            .alias(&["RSQ", "RSQUARED"])
            .int("Length", 14.0, 2.0, MAX_LENGTH)
            .range(0.0, 1.0),
        pane("STDERR", "Standard Error", sta, |i, p| one("StdErr", linreg_map(i.c(), n(p, 0), |f| f.se)))
            .alias(&["SE"])
            .int("Length", 14.0, 3.0, MAX_LENGTH),
        overlay("TSF", "Time Series Forecast", sta, |i, p| one("TSF", tsf(i.c(), n(p, 0))))
            .int("Length", 14.0, 2.0, MAX_LENGTH)
            .multi(),
        overlay("MEDIAN", "Moving Median", sta, |i, p| one("Median", median(i.c(), n(p, 0))))
            .alias(&["MED"])
            .n("Length", 20.0)
            .multi(),
        pane("PRANK", "Percent Rank", sta, |i, p| one("Rank", percent_rank(i.c(), n(p, 0))))
            .alias(&["PERCENTRANK", "PCTRANK"])
            .n("Length", 50.0)
            .range(0.0, 100.0)
            .levels(&[20.0, 80.0]),
        overlay("HIGHEST", "Highest High", sta, |i, p| one("Highest", highest(&i.d.high, n(p, 0))))
            .alias(&["HHV"])
            .n("Length", 20.0)
            .multi(),
        overlay("LOWEST", "Lowest Low", sta, |i, p| one("Lowest", lowest(&i.d.low, n(p, 0))))
            .alias(&["LLV"])
            .n("Length", 20.0)
            .multi(),
        // Levels.
        overlay("PIVOT", "Pivot Points (Classic)", lvl, |i, _| pivot_levels(i, PivotKind::Classic))
            .alias(&["PIVOTS", "PP"]),
        overlay("PIVOTFIB", "Pivot Points (Fibonacci)", lvl, |i, _| pivot_levels(i, PivotKind::Fibonacci)),
        overlay("PIVOTCAM", "Pivot Points (Camarilla)", lvl, |i, _| pivot_levels(i, PivotKind::Camarilla))
            .alias(&["CAMARILLA"]),
        overlay("PIVOTWOODIE", "Pivot Points (Woodie)", lvl, |i, _| pivot_levels(i, PivotKind::Woodie))
            .alias(&["WOODIE"]),
        overlay("PIVOTDM", "Pivot Points (DeMark)", lvl, |i, _| pivot_levels(i, PivotKind::Demark)).alias(&["DEMARK"]),
    ]
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::catalog::CATALOG as SECURITIES;

    /// A seeded random walk with consistent OHLC and varying volume.
    fn walk(len: usize, tf: Timeframe) -> Vec<Bar> {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut c = 100.0;
        (0..len)
            .map(|i| {
                let o = c;
                c *= 1.0 + (rnd() - 0.5) * 0.04;
                let h = o.max(c) * (1.0 + rnd() * 0.01);
                let l = o.min(c) * (1.0 - rnd() * 0.01);
                Bar::new(i as i64 * tf.seconds(), o, h, l, c, 1000.0 + rnd() * 9000.0)
            })
            .collect()
    }

    #[test]
    fn codes_and_aliases_are_unique_and_uppercase() {
        let mut seen = HashSet::new();
        for d in all() {
            for code in std::iter::once(&d.code).chain(d.aliases) {
                assert_eq!(*code, code.to_ascii_uppercase(), "{code} should be uppercase");
                assert!(!code.contains(char::is_whitespace), "{code} must be one word");
                assert!(seen.insert(*code), "{code} is used twice");
            }
            assert_eq!(find(d.code).map(|f| f.code), Some(d.code));
        }
        assert!(all().len() >= 140, "catalog has {} indicators", all().len());
    }

    #[test]
    fn codes_do_not_shadow_listed_securities() {
        for d in all() {
            for code in std::iter::once(&d.code).chain(d.aliases) {
                assert!(!SECURITIES.iter().any(|l| l.ticker.eq_ignore_ascii_case(code)), "{code} is also a ticker");
            }
        }
    }

    #[test]
    fn categories_are_contiguous() {
        let order: Vec<Category> = all().iter().map(|d| d.category).collect();
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(order, sorted, "catalog entries should be grouped by category");
        for c in Category::ALL {
            assert!(order.contains(&c), "{} is empty", c.label());
        }
    }

    #[test]
    fn params_are_well_formed() {
        for d in all() {
            for p in &d.params {
                assert!(p.min <= p.default && p.default <= p.max, "{} {}", d.code, p.name);
                assert!(!p.int || p.default.fract() == 0.0, "{} {}", d.code, p.name);
            }
            assert!(d.range.is_none() || d.pane, "{} has a range but draws over prices", d.code);
            assert!(d.levels.is_empty() || d.pane, "{} has levels but draws over prices", d.code);
        }
    }

    #[test]
    fn every_indicator_computes_on_a_random_walk() {
        for tf in [Timeframe::M5, Timeframe::D1] {
            let bars = walk(600, tf);
            let input = Input::new(&bars, tf);
            for d in all() {
                let outs = d.run(&input, &[]);
                assert!(!outs.is_empty(), "{} has no outputs", d.code);
                for o in &outs {
                    assert!(o.values.len() >= bars.len(), "{} {} is short", d.code, o.name);
                    let finite: Vec<f64> = o.values.iter().copied().filter(|v| v.is_finite()).collect();
                    assert!(
                        !finite.is_empty() || d.code == "FRACTALS" || o.name.starts_with("Squeeze"),
                        "{} {} is blank",
                        d.code,
                        o.name
                    );
                    assert!(
                        o.values.iter().all(|v| v.is_finite() || v.is_nan()),
                        "{} {} has infinities",
                        d.code,
                        o.name
                    );
                    if let Some((lo, hi)) = d.range {
                        let eps = 1e-9 * (hi - lo);
                        assert!(
                            finite.iter().all(|v| *v >= lo - eps && *v <= hi + eps),
                            "{} {} leaves {lo}..{hi}",
                            d.code,
                            o.name
                        );
                    }
                    assert!(d.pane || o.plot != Plot::Hist, "{} draws a histogram over prices", d.code);
                }
            }
        }
    }

    #[test]
    fn overlays_stay_on_the_price_scale_across_their_parameter_bounds() {
        for tf in [Timeframe::M5, Timeframe::D1] {
            let bars = walk(1200, tf);
            let input = Input::new(&bars, tf);
            let lo = bars.iter().map(|b| b.low).fold(f64::INFINITY, f64::min);
            let hi = bars.iter().map(|b| b.high).fold(f64::NEG_INFINITY, f64::max);
            let (floor, ceil) = (lo - 10.0 * (hi - lo), hi + 10.0 * (hi - lo));
            for d in all().iter().filter(|d| !d.pane) {
                let lows: Vec<f64> = d.params.iter().map(|p| p.min).collect();
                let highs: Vec<f64> = d.params.iter().map(|p| p.max).collect();
                for params in [&lows[..], &[][..], &highs[..]] {
                    for o in d.run(&input, params) {
                        let off = o.values.iter().find(|v| v.is_finite() && !(floor..=ceil).contains(*v));
                        assert!(off.is_none(), "{} {} {params:?} reaches {off:?}", d.code, o.name);
                    }
                }
            }
        }
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        let flat: Vec<Bar> = (0..60).map(|i| Bar::new(i * 60, 5.0, 5.0, 5.0, 5.0, 0.0)).collect();
        let short = walk(3, Timeframe::D1);
        for bars in [&[][..], &short[..1], &short[..2], &short[..], &flat[..]] {
            let input = Input::new(bars, Timeframe::M1);
            for d in all() {
                let lows: Vec<f64> = d.params.iter().map(|p| p.min).collect();
                let highs: Vec<f64> = d.params.iter().map(|p| p.max).collect();
                for params in [&[][..], &lows[..], &highs[..]] {
                    for o in d.run(&input, params) {
                        assert!(o.values.len() >= bars.len(), "{} {}", d.code, o.name);
                    }
                }
            }
        }
    }

    #[test]
    fn parse_fills_and_clamps_parameters() {
        assert_eq!(Spec::parse("rsi", &[]), Some(Spec { code: "RSI".into(), params: vec![14.0] }));
        assert_eq!(Spec::parse("Bollinger", &["10"]).unwrap().params, vec![10.0, 2.0]);
        assert_eq!(Spec::parse("SMA", &["0"]).unwrap().params, vec![1.0]);
        assert_eq!(Spec::parse("SMA", &["20.4"]).unwrap().params, vec![20.0]);
        assert_eq!(Spec::parse("SMA", &["20", "30"]), None);
        assert_eq!(Spec::parse("SMA", &["abc"]), None);
        assert_eq!(Spec::parse("NOPE", &[]), None);
        assert_eq!(Spec::parse("BB", &["20", "2.5"]).unwrap().label(), "BB 20 2.5");
        assert_eq!(Spec::parse("VWAP", &[]).unwrap().label(), "VWAP");
        assert_eq!(Spec::parse("psar", &[]).unwrap().label(), "PSAR 0.02 0.02 0.2");
    }

    #[test]
    fn same_kind_keeps_multiple_moving_averages() {
        let sma = |n: f64| Spec::new("SMA", &[n]).unwrap();
        let rsi = |n: f64| Spec::new("RSI", &[n]).unwrap();
        assert!(!sma(20.0).same_kind(&sma(50.0)));
        assert!(sma(20.0).same_kind(&sma(20.0)));
        assert!(rsi(14.0).same_kind(&rsi(21.0)));
        assert!(!rsi(14.0).same_kind(&sma(14.0)));
        assert!(rsi(14.0).is_pane() && !sma(20.0).is_pane());
    }

    #[test]
    fn specs_round_trip_and_read_legacy_layouts() {
        let specs = vec![Spec::new("BB", &[20.0, 2.5]).unwrap(), Spec::new("ICHIMOKU", &[]).unwrap()];
        let json = serde_json::to_string(&specs).unwrap();
        assert_eq!(json, r#"["BB 20 2.5","ICHIMOKU 9 26 52 26"]"#);
        assert_eq!(serde_json::from_str::<Vec<Spec>>(&json).unwrap(), specs);
        let legacy = r#"[{"Sma":20},{"Ema":9},"Vwap",{"Bollinger":[20,2.0]},{"Rsi":14},{"Macd":[12,26,9]}]"#;
        let labels: Vec<String> = serde_json::from_str::<Vec<Spec>>(legacy).unwrap().iter().map(Spec::label).collect();
        assert_eq!(labels, ["SMA 20", "EMA 9", "VWAP", "BB 20 2", "RSI 14", "MACD 12 26 9"]);
        // Codes this build doesn't know survive a round trip untouched.
        let future: Spec = serde_json::from_str(r#""HYPERWAVE 7 1.5""#).unwrap();
        assert_eq!((future.code.as_str(), future.label().as_str()), ("HYPERWAVE", "HYPERWAVE 7 1.5"));
        assert!(future.def().is_none() && future.compute(&Input::new(&[], Timeframe::D1)).is_empty());
    }
}
