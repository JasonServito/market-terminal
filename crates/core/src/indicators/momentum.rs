//! Oscillators and momentum: RSI and its variants, stochastics, MACD-style differences,
//! rate of change, and the cycle oscillators.

use super::ma::{ema, rma, sma, smooth, swma, wma};
use super::stats::{change, highest, linreg, lowest, percent_rank, sum};
use super::volatility::true_range;
use super::{NAN, Ohlcv, ago, from_first, ratio, zip};

/// Relative Strength Index with Wilder smoothing (first value at index `n`).
/// Single pass with no intermediate buffers: on large series the cost is dominated by
/// first-touch of output memory, so only the output is allocated.
pub fn rsi(closes: &[f64], n: usize) -> Vec<f64> {
    from_first(closes, |closes| {
        let mut out = vec![NAN; closes.len()];
        if n == 0 || closes.len() <= n {
            return out;
        }
        let value = |g: f64, l: f64| {
            if l == 0.0 { if g == 0.0 { 50.0 } else { 100.0 } } else { 100.0 - 100.0 / (1.0 + g / l) }
        };
        let (mut ag, mut al) = (0.0, 0.0);
        for i in 1..=n {
            let d = closes[i] - closes[i - 1];
            ag += d.max(0.0);
            al += (-d).max(0.0);
        }
        ag /= n as f64;
        al /= n as f64;
        out[n] = value(ag, al);
        let alpha = 1.0 / n as f64;
        for i in (n + 1)..closes.len() {
            let d = closes[i] - closes[i - 1];
            ag = alpha * d.max(0.0) + (1.0 - alpha) * ag;
            al = alpha * (-d).max(0.0) + (1.0 - alpha) * al;
            out[i] = value(ag, al);
        }
        out
    })
}

#[derive(Debug, Clone)]
pub struct Macd {
    pub macd: Vec<f64>,
    pub signal: Vec<f64>,
    pub histogram: Vec<f64>,
}

/// Streaming EMA seeded with the SMA of its first `n` inputs.
struct Ema {
    n: usize,
    alpha: f64,
    seen: usize,
    acc: f64,
}

impl Ema {
    fn new(n: usize) -> Self {
        Self { n, alpha: 2.0 / (n as f64 + 1.0), seen: 0, acc: 0.0 }
    }

    /// Feed one value; returns the EMA once `n` values have been seen.
    fn push(&mut self, v: f64) -> Option<f64> {
        self.seen += 1;
        if self.seen < self.n {
            self.acc += v;
            None
        } else if self.seen == self.n {
            self.acc = (self.acc + v) / self.n as f64;
            Some(self.acc)
        } else {
            self.acc = self.alpha * v + (1.0 - self.alpha) * self.acc;
            Some(self.acc)
        }
    }
}

/// MACD line (fast EMA - slow EMA), its signal EMA and histogram, in one pass.
pub fn macd(closes: &[f64], fast: usize, slow: usize, signal: usize) -> Macd {
    let len = closes.len();
    let mut line = vec![f64::NAN; len];
    let mut sig = vec![f64::NAN; len];
    let mut histogram = vec![f64::NAN; len];
    if fast == 0 || slow == 0 || signal == 0 {
        return Macd { macd: line, signal: sig, histogram };
    }
    let (mut ef, mut es, mut eg) = (Ema::new(fast), Ema::new(slow), Ema::new(signal));
    for (i, &c) in closes.iter().enumerate() {
        let (f, s) = (ef.push(c), es.push(c));
        let (Some(f), Some(s)) = (f, s) else { continue };
        let m = f - s;
        line[i] = m;
        // The signal EMA starts once the MACD line itself is defined.
        if let Some(g) = eg.push(m) {
            sig[i] = g;
            histogram[i] = m - g;
        }
    }
    Macd { macd: line, signal: sig, histogram }
}

/// A line with an EMA signal and their difference.
fn with_signal(line: Vec<f64>, n: usize) -> Macd {
    let signal = ema(&line, n);
    let histogram = zip(&line, &signal, |a, b| a - b);
    Macd { macd: line, signal, histogram }
}

/// Percentage price oscillator: (fast EMA − slow EMA) / slow EMA × 100, with signal.
pub fn ppo(values: &[f64], fast: usize, slow: usize, signal: usize) -> Macd {
    let line = zip(&ema(values, fast), &ema(values, slow), |f, s| ratio(f - s, s, 0.0) * 100.0);
    with_signal(line, signal)
}

/// Absolute price oscillator: fast EMA − slow EMA.
pub fn apo(values: &[f64], fast: usize, slow: usize) -> Vec<f64> {
    zip(&ema(values, fast), &ema(values, slow), |f, s| f - s)
}

/// Rate of change in percent over `n` bars.
pub fn roc(values: &[f64], n: usize) -> Vec<f64> {
    (0..values.len())
        .map(|i| if i >= n { ratio(values[i] - values[i - n], values[i - n], NAN) * 100.0 } else { NAN })
        .collect()
}

/// Momentum: the change over `n` bars.
pub fn momentum(values: &[f64], n: usize) -> Vec<f64> {
    change(values, n)
}

/// %K and %D lines of a stochastic.
#[derive(Debug, Clone)]
pub struct Stoch {
    pub k: Vec<f64>,
    pub d: Vec<f64>,
}

/// Raw stochastic: where `close` sits in the `n`-bar high/low range, 0..100 (50 when flat).
fn stoch_raw(high: &[f64], low: &[f64], close: &[f64], n: usize) -> Vec<f64> {
    let (hh, ll) = (highest(high, n), lowest(low, n));
    (0..close.len()).map(|i| 100.0 * ratio(close[i] - ll[i], hh[i] - ll[i], 0.5)).collect()
}

/// Stochastic oscillator: %K = SMA(raw, `smooth_k`), %D = SMA(%K, `d`).
pub fn stochastic(o: &Ohlcv, n: usize, smooth_k: usize, d: usize) -> Stoch {
    let k = sma(&stoch_raw(&o.high, &o.low, &o.close, n), smooth_k);
    let d = sma(&k, d);
    Stoch { k, d }
}

/// Stochastic RSI: a stochastic of RSI(`rsi_n`) over `stoch_n` bars.
pub fn stoch_rsi(values: &[f64], rsi_n: usize, stoch_n: usize, smooth_k: usize, d: usize) -> Stoch {
    let r = rsi(values, rsi_n);
    let k = sma(&stoch_raw(&r, &r, &r, stoch_n), smooth_k);
    let d = sma(&k, d);
    Stoch { k, d }
}

/// Commodity channel index of `values` (usually the typical price) over `n` bars.
pub fn cci(values: &[f64], n: usize) -> Vec<f64> {
    let m = sma(values, n);
    (0..values.len())
        .map(|i| {
            if m[i].is_nan() {
                return NAN;
            }
            let md = values[i + 1 - n..=i].iter().map(|v| (v - m[i]).abs()).sum::<f64>() / n as f64;
            ratio(values[i] - m[i], 0.015 * md, 0.0)
        })
        .collect()
}

/// Williams %R: −100 (at the `n`-bar low) to 0 (at the high).
pub fn williams_r(o: &Ohlcv, n: usize) -> Vec<f64> {
    stoch_raw(&o.high, &o.low, &o.close, n).into_iter().map(|k| k - 100.0).collect()
}

/// Ultimate oscillator: buying pressure over true range on three horizons, weighted 4:2:1.
pub fn ultimate(o: &Ohlcv, a: usize, b: usize, c: usize) -> Vec<f64> {
    let prev = |i: usize| ago(&o.close, i, 1);
    let bp: Vec<f64> = (0..o.len()).map(|i| o.close[i] - o.low[i].min(prev(i))).collect();
    let tr: Vec<f64> = (0..o.len()).map(|i| o.high[i].max(prev(i)) - o.low[i].min(prev(i))).collect();
    let avg = |n| zip(&sum(&bp, n), &sum(&tr, n), |p, t| ratio(p, t, 0.0));
    let (x, y, z) = (avg(a), avg(b), avg(c));
    (0..o.len()).map(|i| 100.0 * (4.0 * x[i] + 2.0 * y[i] + z[i]) / 7.0).collect()
}

/// Awesome oscillator: SMA(median price, `fast`) − SMA(median price, `slow`).
pub fn awesome(o: &Ohlcv, fast: usize, slow: usize) -> Vec<f64> {
    let hl2 = o.hl2();
    zip(&sma(&hl2, fast), &sma(&hl2, slow), |a, b| a - b)
}

/// Accelerator oscillator: AO − SMA(AO, `n`).
pub fn accelerator(o: &Ohlcv, fast: usize, slow: usize, n: usize) -> Vec<f64> {
    let ao = awesome(o, fast, slow);
    zip(&ao, &sma(&ao, n), |a, b| a - b)
}

/// True strength index: double-smoothed momentum over double-smoothed |momentum|, × 100.
pub fn tsi(values: &[f64], long: usize, short: usize) -> Vec<f64> {
    let pc = change(values, 1);
    let apc: Vec<f64> = pc.iter().map(|v| v.abs()).collect();
    let num = ema(&ema(&pc, long), short);
    let den = ema(&ema(&apc, long), short);
    zip(&num, &den, |a, b| ratio(a, b, 0.0) * 100.0)
}

/// TRIX: one-bar percent change of a triple-smoothed EMA.
pub fn trix(values: &[f64], n: usize) -> Vec<f64> {
    let e3 = ema(&ema(&ema(values, n), n), n);
    roc(&e3, 1)
}

/// Pring's Know Sure Thing: weighted sum of four smoothed rates of change, with a signal.
pub fn kst(values: &[f64], roc_n: [usize; 4], sma_n: [usize; 4], signal: usize) -> Macd {
    let mut line = vec![0.0; values.len()];
    for (w, (r, s)) in roc_n.iter().zip(sma_n).enumerate() {
        let part = sma(&roc(values, *r), s);
        for (l, p) in line.iter_mut().zip(&part) {
            *l += (w + 1) as f64 * p;
        }
    }
    let signal = sma(&line, signal);
    let histogram = zip(&line, &signal, |a, b| a - b);
    Macd { macd: line, signal, histogram }
}

/// Up and down moves (`max(Δ, 0)`, `max(−Δ, 0)`) of `values` over `k` bars.
fn moves(values: &[f64], k: usize) -> (Vec<f64>, Vec<f64>) {
    let ch = change(values, k);
    let up = ch.iter().map(|c| if c.is_nan() { NAN } else { c.max(0.0) }).collect();
    let dn = ch.iter().map(|c| if c.is_nan() { NAN } else { (-c).max(0.0) }).collect();
    (up, dn)
}

/// Chande momentum oscillator, −100..100.
pub fn cmo(values: &[f64], n: usize) -> Vec<f64> {
    let (up, dn) = moves(values, 1);
    zip(&sum(&up, n), &sum(&dn, n), |u, d| 100.0 * ratio(u - d, u + d, 0.0))
}

/// Detrended price oscillator: close − SMA(n) from n/2 + 1 bars ago.
pub fn dpo(values: &[f64], n: usize) -> Vec<f64> {
    let m = sma(values, n);
    let back = n / 2 + 1;
    (0..values.len()).map(|i| values[i] - ago(&m, i, back)).collect()
}

/// Coppock curve: WMA(ROC(long) + ROC(short), `n`).
pub fn coppock(values: &[f64], n: usize, long: usize, short: usize) -> Vec<f64> {
    wma(&zip(&roc(values, long), &roc(values, short), |a, b| a + b), n)
}

/// Relative vigor index: close − open over high − low, summed over `n`, with an SWMA signal.
pub fn rvgi(o: &Ohlcv, n: usize) -> Macd {
    let num = swma(&zip(&o.close, &o.open, |c, op| c - op));
    let den = swma(&zip(&o.high, &o.low, |h, l| h - l));
    let line = zip(&sum(&num, n), &sum(&den, n), |a, b| ratio(a, b, 0.0));
    let signal = swma(&line);
    let histogram = zip(&line, &signal, |a, b| a - b);
    Macd { macd: line, signal, histogram }
}

/// Ehlers' Fisher transform of the median price's position in its `n`-bar range, with the
/// previous value as trigger.
pub fn fisher(o: &Ohlcv, n: usize) -> (Vec<f64>, Vec<f64>) {
    let hl2 = o.hl2();
    let (hh, ll) = (highest(&hl2, n), lowest(&hl2, n));
    let mut fish = vec![NAN; o.len()];
    let (mut v, mut f) = (0.0, 0.0);
    for i in 0..o.len() {
        if hh[i].is_nan() {
            continue;
        }
        v = (0.66 * (ratio(hl2[i] - ll[i], hh[i] - ll[i], 0.5) - 0.5) + 0.67 * v).clamp(-0.999, 0.999);
        f = 0.5 * ((1.0 + v) / (1.0 - v)).ln() + 0.5 * f;
        fish[i] = f;
    }
    let trigger = (0..fish.len()).map(|i| ago(&fish, i, 1)).collect();
    (fish, trigger)
}

/// Blau's stochastic momentum index: distance from the midpoint of the `n`-bar range,
/// double-smoothed over `d`, as a percentage of half the range; with an EMA signal.
pub fn smi(o: &Ohlcv, n: usize, d: usize, signal: usize) -> Macd {
    let (hh, ll) = (highest(&o.high, n), lowest(&o.low, n));
    let rel: Vec<f64> = (0..o.len()).map(|i| o.close[i] - (hh[i] + ll[i]) / 2.0).collect();
    let range = zip(&hh, &ll, |h, l| h - l);
    let num = ema(&ema(&rel, d), d);
    let den = ema(&ema(&range, d), d);
    with_signal(zip(&num, &den, |a, b| ratio(a, b / 2.0, 0.0) * 100.0), signal)
}

/// SMI ergodic indicator: TSI(`long`, `short`) with an EMA signal and their difference.
pub fn smi_ergodic(values: &[f64], short: usize, long: usize, signal: usize) -> Macd {
    with_signal(tsi(values, long, short), signal)
}

/// Connors RSI: the mean of RSI(`rsi_n`), RSI(`streak_n`) of the up/down streak length, and
/// the percent rank of the one-bar change over `rank_n` bars.
pub fn connors_rsi(values: &[f64], rsi_n: usize, streak_n: usize, rank_n: usize) -> Vec<f64> {
    let mut streak = vec![0.0; values.len()];
    for i in 1..values.len() {
        let s = streak[i - 1];
        streak[i] = if values[i] > values[i - 1] {
            if s > 0.0 { s + 1.0 } else { 1.0 }
        } else if values[i] < values[i - 1] {
            if s < 0.0 { s - 1.0 } else { -1.0 }
        } else {
            0.0
        };
    }
    let (a, b, c) = (rsi(values, rsi_n), rsi(&streak, streak_n), percent_rank(&roc(values, 1), rank_n));
    (0..values.len()).map(|i| (a[i] + b[i] + c[i]) / 3.0).collect()
}

/// Ehlers' Laguerre RSI with damping `gamma`, 0..1.
pub fn laguerre_rsi(values: &[f64], gamma: f64) -> Vec<f64> {
    from_first(values, |p| {
        let g = gamma;
        let mut l = [p.first().copied().unwrap_or_default(); 4];
        p.iter()
            .map(|&x| {
                let prev = l;
                l[0] = (1.0 - g) * x + g * prev[0];
                l[1] = -g * l[0] + prev[0] + g * prev[1];
                l[2] = -g * l[1] + prev[1] + g * prev[2];
                l[3] = -g * l[2] + prev[2] + g * prev[3];
                let (mut cu, mut cd) = (0.0, 0.0);
                for w in l.windows(2) {
                    let d = w[0] - w[1];
                    if d >= 0.0 { cu += d } else { cd -= d }
                }
                ratio(cu, cu + cd, 0.0)
            })
            .collect()
    })
}

/// Relative momentum index: an RSI of the change over `m` bars, Wilder-smoothed over `n`.
pub fn rmi(values: &[f64], n: usize, m: usize) -> Vec<f64> {
    let (up, dn) = moves(values, m);
    zip(&rma(&up, n), &rma(&dn, n), |u, d| 100.0 * ratio(u, u + d, 0.5))
}

/// DecisionPoint price momentum oscillator: the one-bar ROC smoothed twice with the custom
/// 2/n factor (×10 after the first), with an EMA signal.
pub fn pmo(values: &[f64], a: usize, b: usize, signal: usize) -> Macd {
    let r = roc(values, 1);
    let s1: Vec<f64> = smooth(&r, a, 2.0 / a as f64).into_iter().map(|v| v * 10.0).collect();
    with_signal(smooth(&s1, b, 2.0 / b as f64), signal)
}

/// Balance of power: SMA of (close − open) / (high − low).
pub fn bop(o: &Ohlcv, n: usize) -> Vec<f64> {
    let raw: Vec<f64> = (0..o.len()).map(|i| ratio(o.close[i] - o.open[i], o.high[i] - o.low[i], 0.0)).collect();
    sma(&raw, n)
}

/// Elder ray: bull power (high − EMA) and bear power (low − EMA).
pub fn elder_ray(o: &Ohlcv, n: usize) -> (Vec<f64>, Vec<f64>) {
    let e = ema(&o.close, n);
    (zip(&o.high, &e, |h, e| h - e), zip(&o.low, &e, |l, e| l - e))
}

/// Bull bear power: (high − EMA) + (low − EMA).
pub fn bull_bear_power(o: &Ohlcv, n: usize) -> Vec<f64> {
    let (bull, bear) = elder_ray(o, n);
    zip(&bull, &bear, |a, b| a + b)
}

/// Chande's Qstick: SMA of close − open.
pub fn qstick(o: &Ohlcv, n: usize) -> Vec<f64> {
    sma(&zip(&o.close, &o.open, |c, op| c - op), n)
}

/// Psychological line: the percentage of up closes over `n` bars.
pub fn psychological(values: &[f64], n: usize) -> Vec<f64> {
    let up: Vec<f64> =
        (0..values.len()).map(|i| if i == 0 { NAN } else { f64::from(values[i] > values[i - 1]) }).collect();
    sum(&up, n).into_iter().map(|s| 100.0 * s / n as f64).collect()
}

/// Chande forecast oscillator: percent distance of the close from its `n`-bar regression.
pub fn cfo(values: &[f64], n: usize) -> Vec<f64> {
    zip(values, &linreg(values, n), |c, l| 100.0 * ratio(c - l, c, 0.0))
}

/// Intraday momentum index: an RSI of open-to-close moves, 0..100.
pub fn imi(o: &Ohlcv, n: usize) -> Vec<f64> {
    let gains: Vec<f64> = (0..o.len()).map(|i| (o.close[i] - o.open[i]).max(0.0)).collect();
    let losses: Vec<f64> = (0..o.len()).map(|i| (o.open[i] - o.close[i]).max(0.0)).collect();
    zip(&sum(&gains, n), &sum(&losses, n), |g, l| 100.0 * ratio(g, g + l, 0.5))
}

/// Schaff trend cycle: a twice-stochasticised, half-smoothed MACD(`fast`, `slow`) over a
/// `cycle`-bar window, 0..100.
pub fn stc(values: &[f64], cycle: usize, fast: usize, slow: usize) -> Vec<f64> {
    let m = apo(values, fast, slow);
    let stage = |x: &[f64]| {
        let (lo, hi) = (lowest(x, cycle), highest(x, cycle));
        let mut out = vec![NAN; x.len()];
        let (mut raw, mut sm) = (NAN, NAN);
        for i in 0..x.len() {
            if lo[i].is_nan() {
                continue;
            }
            let range = hi[i] - lo[i];
            if range > 0.0 {
                raw = (x[i] - lo[i]) / range * 100.0;
            } else if raw.is_nan() {
                raw = 0.0;
            }
            sm = if sm.is_nan() { raw } else { sm + 0.5 * (raw - sm) };
            out[i] = sm;
        }
        out
    };
    stage(&stage(&m))
}

/// Ehlers' center of gravity over `n` bars, with the previous value as signal.
pub fn center_of_gravity(values: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut cg = vec![NAN; values.len()];
    for i in (n.max(1) - 1..values.len()).filter(|_| n > 0) {
        let (mut num, mut den) = (0.0, 0.0);
        for k in 0..n {
            num += (k + 1) as f64 * values[i - k];
            den += values[i - k];
        }
        cg[i] = if den != 0.0 { -num / den + (n as f64 + 1.0) / 2.0 } else { 0.0 };
    }
    let signal = (0..cg.len()).map(|i| ago(&cg, i, 1)).collect();
    (cg, signal)
}

/// Inverse Fisher transform of RSI(`rsi_n`), WMA-smoothed over `n`, −1..1.
pub fn ift_rsi(values: &[f64], rsi_n: usize, n: usize) -> Vec<f64> {
    let v: Vec<f64> = rsi(values, rsi_n).into_iter().map(|r| 0.1 * (r - 50.0)).collect();
    wma(&v, n).into_iter().map(f64::tanh).collect()
}

/// Pretty good oscillator: distance from the SMA in units of the average true range.
pub fn pgo(o: &Ohlcv, n: usize) -> Vec<f64> {
    let m = sma(&o.close, n);
    let atr = ema(&true_range(o), n);
    (0..o.len()).map(|i| ratio(o.close[i] - m[i], atr[i], 0.0)).collect()
}

/// QQE: RSI smoothed over `smooth` bars and a trailing level `factor` times its smoothed
/// volatility away, flipping sides when crossed.
pub fn qqe(values: &[f64], rsi_n: usize, smooth: usize, factor: f64) -> (Vec<f64>, Vec<f64>) {
    let line = ema(&rsi(values, rsi_n), smooth);
    let wilders = (2 * rsi_n).saturating_sub(1).max(1);
    let atr: Vec<f64> = (0..line.len()).map(|i| (line[i] - ago(&line, i, 1)).abs()).collect();
    let dar: Vec<f64> = ema(&ema(&atr, wilders), wilders).into_iter().map(|v| v * factor).collect();
    let mut trail = vec![NAN; line.len()];
    let (mut long, mut short, mut trend) = (NAN, NAN, 1);
    let mut prev = NAN;
    for i in 0..line.len() {
        let (r, d) = (line[i], dar[i]);
        if r.is_nan() || d.is_nan() {
            continue;
        }
        let (nl, ns) = (r - d, r + d);
        let (pl, ps) = (long, short);
        long = if !pl.is_nan() && prev > pl && r > pl { pl.max(nl) } else { nl };
        short = if !ps.is_nan() && prev < ps && r < ps { ps.min(ns) } else { ns };
        if !ps.is_nan() && prev <= ps && r > ps {
            trend = 1;
        } else if !pl.is_nan() && prev >= pl && r < pl {
            trend = -1;
        }
        trail[i] = if trend == 1 { long } else { short };
        prev = r;
    }
    (line, trail)
}

/// WaveTrend (LazyBear): a CCI-like channel index of the typical price, EMA-smoothed, with
/// a 4-bar SMA signal.
pub fn wavetrend(o: &Ohlcv, n1: usize, n2: usize) -> (Vec<f64>, Vec<f64>) {
    let ap = o.hlc3();
    let esa = ema(&ap, n1);
    let dev = ema(&zip(&ap, &esa, |a, e| (a - e).abs()), n1);
    let ci: Vec<f64> = (0..ap.len()).map(|i| ratio(ap[i] - esa[i], 0.015 * dev[i], 0.0)).collect();
    let wt1 = ema(&ci, n2);
    let wt2 = sma(&wt1, 4);
    (wt1, wt2)
}

/// Where `fast` crosses `slow`: `slow`'s value at each bar where `fast` moves from below it
/// to above (`up`) or from above it to below (`down`), NaN elsewhere. Marks WaveTrend's turns.
pub fn crosses(fast: &[f64], slow: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut up = vec![NAN; fast.len()];
    let mut down = vec![NAN; fast.len()];
    for i in 1..fast.len().min(slow.len()) {
        let (a0, b0, a1, b1) = (fast[i - 1], slow[i - 1], fast[i], slow[i]);
        if ![a0, b0, a1, b1].iter().all(|v| v.is_finite()) {
            continue;
        }
        if a0 <= b0 && a1 > b1 {
            up[i] = b1;
        } else if a0 >= b0 && a1 < b1 {
            down[i] = b1;
        }
    }
    (up, down)
}

/// Ehlers' cyber cycle with smoothing `alpha`, and its previous value as trigger.
pub fn cyber_cycle(values: &[f64], alpha: f64) -> (Vec<f64>, Vec<f64>) {
    let cycle = from_first(values, |p| {
        let a = alpha;
        let pr = |i: usize, k: usize| if i >= k { p[i - k] } else { p[0] };
        let sm: Vec<f64> =
            (0..p.len()).map(|i| (pr(i, 0) + 2.0 * pr(i, 1) + 2.0 * pr(i, 2) + pr(i, 3)) / 6.0).collect();
        let mut c = vec![0.0; p.len()];
        for i in 0..p.len() {
            c[i] = if i < 7 {
                (pr(i, 0) - 2.0 * pr(i, 1) + pr(i, 2)) / 4.0
            } else {
                (1.0 - 0.5 * a).powi(2) * (sm[i] - 2.0 * sm[i - 1] + sm[i - 2]) + 2.0 * (1.0 - a) * c[i - 1]
                    - (1.0 - a).powi(2) * c[i - 2]
            };
        }
        c
    });
    let trigger = (0..cycle.len()).map(|i| ago(&cycle, i, 1)).collect();
    (cycle, trigger)
}

/// Disparity index: percent distance of the close from its `n`-bar SMA.
pub fn disparity(values: &[f64], n: usize) -> Vec<f64> {
    zip(values, &sma(values, n), |c, m| 100.0 * ratio(c - m, m, 0.0))
}

/// Polarized fractal efficiency: straight-line distance over path length across `n` bars,
/// signed by direction and EMA-smoothed, −100..100.
pub fn pfe(values: &[f64], n: usize, smoothing: usize) -> Vec<f64> {
    let k = n as f64;
    let step: Vec<f64> = (0..values.len())
        .map(|i| if i == 0 { NAN } else { ((values[i] - values[i - 1]).powi(2) + 1.0).sqrt() })
        .collect();
    let path = sum(&step, n);
    let raw: Vec<f64> = (0..values.len())
        .map(|i| {
            if i < n || path[i].is_nan() {
                return NAN;
            }
            let d = values[i] - values[i - n];
            let p = 100.0 * (d * d + k * k).sqrt() / path[i];
            if d > 0.0 { p } else { -p }
        })
        .collect();
    ema(&raw, smoothing)
}

/// Kaufman's efficiency ratio: |net change| over the sum of |bar changes|, 0..1.
pub fn efficiency_ratio(values: &[f64], n: usize) -> Vec<f64> {
    let step: Vec<f64> = change(values, 1).into_iter().map(f64::abs).collect();
    let path = sum(&step, n);
    (0..values.len())
        .map(|i| if i >= n { ratio((values[i] - values[i - n]).abs(), path[i], 0.0) } else { NAN })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Bar;

    #[test]
    fn crosses_mark_turns_at_the_slow_line() {
        let fast = [1.0, 3.0, 4.0, 2.0, NAN, 5.0, 1.0];
        let slow = [2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0];
        let (up, down) = crosses(&fast, &slow);
        let at =
            |v: &[f64]| v.iter().enumerate().filter(|(_, x)| x.is_finite()).map(|(i, x)| (i, *x)).collect::<Vec<_>>();
        assert_eq!(at(&up), [(1, 2.0)], "a gap (NaN) never makes a cross");
        assert_eq!(at(&down), [(3, 3.0), (6, 3.0)]);
    }

    fn near(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9 * (1.0 + b.abs())
    }

    fn ohlcv(rows: &[(f64, f64, f64, f64)]) -> Ohlcv {
        let bars: Vec<Bar> =
            rows.iter().enumerate().map(|(i, r)| Bar::new(i as i64, r.0, r.1, r.2, r.3, 100.0)).collect();
        Ohlcv::new(&bars)
    }

    fn wave(n: usize) -> Ohlcv {
        let rows: Vec<(f64, f64, f64, f64)> = (0..n)
            .map(|i| {
                let c = 100.0 + (i as f64 * 0.17).sin() * 6.0 + i as f64 * 0.02;
                (c - 0.3, c + 1.0 + (i % 3) as f64 * 0.2, c - 1.1, c)
            })
            .collect();
        ohlcv(&rows)
    }

    #[test]
    fn stochastic_and_williams_hand_computed() {
        // 3-bar window: HH 12, LL 8, close 11 → raw 75, %R −25.
        let o = ohlcv(&[(9.0, 10.0, 8.0, 9.0), (10.0, 12.0, 9.0, 11.0), (11.0, 11.5, 10.0, 11.0)]);
        let s = stochastic(&o, 3, 1, 1);
        assert!(near(s.k[2], 75.0) && near(s.d[2], 75.0));
        assert!(near(williams_r(&o, 3)[2], -25.0));
        // A flat window sits in the middle instead of dividing by zero.
        let flat = ohlcv(&[(5.0, 5.0, 5.0, 5.0); 4]);
        assert_eq!(stochastic(&flat, 3, 1, 1).k[3], 50.0);
    }

    #[test]
    fn cci_hand_computed() {
        // Typical prices 1, 2, 3: mean 2, mean deviation 2/3 → (3 − 2) / (0.015 · 2/3) = 100.
        assert!(near(cci(&[1.0, 2.0, 3.0], 3)[2], 100.0));
    }

    #[test]
    fn differences_match_their_parts() {
        let o = wave(300);
        let c = &o.close;
        let p = ppo(c, 12, 26, 9);
        let (f, s) = (ema(c, 12), ema(c, 26));
        assert!(near(p.macd[200], (f[200] - s[200]) / s[200] * 100.0));
        assert!(near(apo(c, 12, 26)[200], f[200] - s[200]));
        assert!(near(roc(c, 9)[50], (c[50] / c[41] - 1.0) * 100.0));
        assert!(near(momentum(c, 10)[50], c[50] - c[40]));
        let hl2 = o.hl2();
        assert!(near(awesome(&o, 5, 34)[100], sma(&hl2, 5)[100] - sma(&hl2, 34)[100]));
        let ao = awesome(&o, 5, 34);
        assert!(near(accelerator(&o, 5, 34, 5)[100], ao[100] - sma(&ao, 5)[100]));
        let b = bull_bear_power(&o, 13);
        let (bull, bear) = elder_ray(&o, 13);
        assert!(near(b[90], bull[90] + bear[90]));
    }

    #[test]
    fn bounded_oscillators_stay_in_range() {
        let o = wave(500);
        let c = &o.close;
        let checks: Vec<(Vec<f64>, f64, f64)> = vec![
            (cmo(c, 9), -100.0, 100.0),
            (stoch_rsi(c, 14, 14, 3, 3).k, 0.0, 100.0),
            (ultimate(&o, 7, 14, 28), 0.0, 100.0),
            (connors_rsi(c, 3, 2, 100), 0.0, 100.0),
            (laguerre_rsi(c, 0.5), 0.0, 1.0),
            (rmi(c, 20, 5), 0.0, 100.0),
            (imi(&o, 14), 0.0, 100.0),
            (psychological(c, 12), 0.0, 100.0),
            (stc(c, 12, 26, 50), 0.0, 100.0),
            (ift_rsi(c, 5, 9), -1.0, 1.0),
            (qqe(c, 14, 5, 4.236).0, 0.0, 100.0),
            (pfe(c, 10, 5), -100.0, 100.0),
            (efficiency_ratio(c, 10), 0.0, 1.0),
            (smi(&o, 10, 3, 10).macd, -100.0, 100.0),
        ];
        for (k, (v, lo, hi)) in checks.iter().enumerate() {
            let finite: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
            assert!(finite.len() > 300, "check {k} has {} values", finite.len());
            assert!(finite.iter().all(|x| *x >= lo - 1e-9 && *x <= hi + 1e-9), "check {k} out of range");
        }
    }

    #[test]
    fn known_shapes() {
        let up: Vec<f64> = (0..60).map(|i| 10.0 + i as f64).collect();
        assert_eq!(cmo(&up, 9)[30], 100.0);
        assert_eq!(efficiency_ratio(&up, 10)[30], 1.0);
        assert_eq!(psychological(&up, 12)[30], 100.0);
        assert!(near(tsi(&up, 25, 13)[59], 100.0));
        // Streaks: three rises then a fall.
        let c = connors_rsi(&[1.0, 2.0, 3.0, 4.0, 3.0], 2, 2, 2);
        assert!(c[4].is_finite());
        let (cg, sig) = center_of_gravity(&[1.0; 20], 10);
        assert!(near(cg[15], -(55.0 / 10.0) + 5.5) && near(sig[16], cg[15]));
        // DPO compares with the SMA from n/2 + 1 bars back.
        let d = dpo(&up, 10);
        assert!(near(d[30], up[30] - sma(&up, 10)[24]));
    }
}
