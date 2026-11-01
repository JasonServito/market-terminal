//! Turns chart state (data, viewport, chart type, overlays) into GPU draw commands.
//! Shared by the egui widget and the headless frame-time benchmark.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use tr_core::Bar;

use crate::gpu::{DrawCmd, LineKind, LineRef, PaneTransform, SeriesRef, Style, Uniforms};
use crate::lod::{BarPyramid, LinePyramid, level_for};
use crate::viewport::{PriceScale, Viewport};

static NEXT_KEY: AtomicU64 = AtomicU64::new(1);

/// Factor that makes `line` equal `base` at the first index in `[lo, hi)` where both
/// are defined, so a compare series reads as performance relative to the left edge.
pub fn rebase_factor(base: &LineRef, line: &LineRef, lo: usize, hi: usize) -> f64 {
    let (b, l) = (&base.pyramid.data, &line.pyramid.data);
    let n = base.pyramid.level_len(0).min(line.pyramid.level_len(0));
    (lo..hi.min(n))
        .find_map(|i| {
            let (x, y) = (b[i][1] as f64, l[i][1] as f64);
            (x.is_finite() && y.is_finite() && y != 0.0).then(|| x / y)
        })
        .unwrap_or(1.0)
}

/// Process-unique key for GPU buffer caching.
pub fn next_key() -> u64 {
    NEXT_KEY.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum ChartType {
    #[default]
    Candles,
    Bars,
    Line,
    Area,
}

impl ChartType {
    pub const ALL: [ChartType; 4] = [ChartType::Candles, ChartType::Bars, ChartType::Line, ChartType::Area];

    pub fn label(self) -> &'static str {
        match self {
            ChartType::Candles => "Candles",
            ChartType::Bars => "Bars",
            ChartType::Line => "Line",
            ChartType::Area => "Area",
        }
    }
}

/// GPU-ready series: bars plus their LOD pyramid and a close-price line pyramid.
#[derive(Clone)]
pub struct SeriesGpu {
    pub bars: SeriesRef,
    pub closes: LineRef,
}

impl SeriesGpu {
    pub fn build(bars: &[Bar]) -> Self {
        let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
        Self {
            bars: SeriesRef { key: next_key(), version: 1, pyramid: Arc::new(BarPyramid::build(bars)) },
            closes: line_ref(&closes),
        }
    }

    pub fn len(&self) -> usize {
        self.bars.pyramid.len0
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub fn line_ref(values: &[f64]) -> LineRef {
    LineRef { key: next_key(), version: 1, pyramid: Arc::new(LinePyramid::build(values)) }
}

/// A [`LineKind::Dots`] series, reduced so sparse markers stay visible when zoomed out.
pub fn marker_ref(values: &[f64]) -> LineRef {
    LineRef { key: next_key(), version: 1, pyramid: Arc::new(LinePyramid::build_markers(values)) }
}

/// A line overlay with its colour.
#[derive(Clone)]
pub struct Overlay {
    pub line: LineRef,
    pub color: [f32; 4],
    pub kind: LineKind,
    /// Values are drawn multiplied by this factor (compare series rebased per frame).
    pub scale: f64,
}

/// Everything needed to draw the main price pane.
pub struct PricePane<'a> {
    pub series: &'a SeriesGpu,
    pub chart_type: ChartType,
    pub show_volume: bool,
    pub overlays: &'a [Overlay],
    pub viewport: &'a Viewport,
    pub height_px: f64,
    pub log: bool,
    /// `None` = autoscale to the visible bars.
    pub manual_scale: Option<PriceScale>,
    pub style: Style,
}

pub struct PaneOutput {
    pub cmds: Vec<DrawCmd>,
    pub scale: PriceScale,
    pub level: usize,
    pub vol_max: f64,
}

/// GPU transform for a pane laid out in points; the shader works in physical pixels.
fn transform(v: &Viewport, height_pt: f64, scale: &PriceScale, ppp: f32) -> PaneTransform {
    let (y_lo, y_hi) = scale.bounds_t();
    PaneTransform {
        width_px: v.width_px as f32 * ppp,
        height_px: height_pt as f32 * ppp,
        left: v.left(),
        visible_bars: v.visible_bars(),
        y_lo,
        y_hi,
        log: scale.log,
    }
}

/// Autoscale range for the visible part of the series and overlays.
pub fn visible_extent(
    series: &SeriesGpu,
    overlays: &[Overlay],
    v: &Viewport,
    chart_type: ChartType,
) -> Option<(f64, f64, f64)> {
    let p = &series.bars.pyramid;
    let (lo, hi) = v.visible_range(p.len0);
    let k = level_for(v.bar_px, p.levels());
    let (mut l, mut h, vmax) = match chart_type {
        ChartType::Line | ChartType::Area => {
            let (l, h) = series.closes.pyramid.extent(k, lo, hi)?;
            (l, h, p.extent(k, lo, hi).map(|e| e.2).unwrap_or(0.0))
        }
        _ => p.extent(k, lo, hi)?,
    };
    for o in overlays.iter().filter(|o| matches!(o.kind, LineKind::Line | LineKind::Dots | LineKind::Steps)) {
        let kk = level_for(v.bar_px, o.line.pyramid.levels());
        if let Some((a, b)) = o.line.pyramid.extent(kk, lo, hi) {
            l = l.min(a * o.scale);
            h = h.max(b * o.scale);
        }
    }
    Some((l, h, vmax))
}

/// The scale `price_pane` uses when `manual_scale` is `None`.
pub fn autoscale(p: &PricePane<'_>) -> PriceScale {
    autoscale_from(p, visible_extent(p.series, p.overlays, p.viewport, p.chart_type))
}

fn autoscale_from(p: &PricePane<'_>, ext: Option<(f64, f64, f64)>) -> PriceScale {
    let (l, h) = ext.map(|e| (e.0, e.1)).unwrap_or((0.0, 1.0));
    // Leave room at the bottom for the volume overlay.
    let mut s = PriceScale::padded(l, h, 0.08, p.log, p.height_px);
    if p.show_volume {
        let (a, b) = s.bounds_t();
        let extra = (b - a) * 0.12;
        let lo_t = a - extra;
        s.lo = if s.log { lo_t.exp() } else { lo_t };
    }
    s
}

pub fn price_pane(p: &PricePane<'_>) -> PaneOutput {
    let v = p.viewport;
    let pyr = &p.series.bars.pyramid;
    let (lo, hi) = v.visible_range(pyr.len0);
    let k = level_for(v.bar_px, pyr.levels());
    let ext = visible_extent(p.series, p.overlays, v, p.chart_type);
    let vol_max = ext.map(|e| e.2).unwrap_or(1.0);
    let scale = p.manual_scale.unwrap_or_else(|| autoscale_from(p, ext));
    let t = transform(v, p.height_px, &scale, p.style.pixels_per_point);
    let mut style = p.style;
    style.ohlc_bars = p.chart_type == ChartType::Bars;
    style.vol_max = vol_max as f32;
    let mut cmds = Vec::new();
    if p.show_volume {
        let mut vs = style;
        vs.color[3] = 0.35;
        cmds.push(DrawCmd::Volume {
            data: p.series.bars.clone(),
            range: pyr.instance_range(k, lo, hi),
            uniforms: Uniforms::new(&t, &vs),
        });
    }
    match p.chart_type {
        ChartType::Candles | ChartType::Bars => cmds.push(DrawCmd::Candles {
            data: p.series.bars.clone(),
            range: pyr.instance_range(k, lo, hi),
            uniforms: Uniforms::new(&t, &style),
        }),
        ChartType::Line | ChartType::Area => {
            let lp = &p.series.closes.pyramid;
            let kk = level_for(v.bar_px, lp.levels());
            let range = lp.segment_range(kk, lo, hi);
            if p.chart_type == ChartType::Area {
                let mut a = style;
                a.color[3] = 0.25;
                cmds.push(DrawCmd::Line {
                    kind: LineKind::Area,
                    data: p.series.closes.clone(),
                    range,
                    uniforms: Uniforms::new(&t, &a),
                });
            }
            let mut ls = style;
            ls.thickness_px = 2.0;
            cmds.push(DrawCmd::Line {
                kind: LineKind::Line,
                data: p.series.closes.clone(),
                range,
                uniforms: Uniforms::new(&t, &ls),
            });
        }
    }
    cmds.extend(overlay_cmds(p.overlays, v, &t, &style));
    PaneOutput { cmds, scale, level: k, vol_max }
}

fn overlay_cmds(overlays: &[Overlay], v: &Viewport, t: &PaneTransform, base: &Style) -> Vec<DrawCmd> {
    let len = overlays.first().map(|o| o.line.pyramid.level_len(0)).unwrap_or(0);
    let (lo, hi) = v.visible_range(len.max(v.right.ceil().max(0.0) as usize));
    overlays
        .iter()
        .map(|o| {
            let lp = &o.line.pyramid;
            let k = level_for(v.bar_px, lp.levels());
            let mut s = *base;
            s.color = o.color;
            // Drawing v*scale against [lo, hi] equals drawing v against [lo/scale, hi/scale].
            let mut t = *t;
            if o.scale != 1.0 && o.scale > 0.0 {
                let shift = if t.log { o.scale.ln() } else { 0.0 };
                if t.log {
                    t.y_lo -= shift;
                    t.y_hi -= shift;
                } else {
                    t.y_lo /= o.scale;
                    t.y_hi /= o.scale;
                }
            }
            let range = match o.kind {
                LineKind::Hist | LineKind::Dots | LineKind::Steps => {
                    let frac = if o.kind == LineKind::Steps { 1.0 } else { base.body_frac as f64 };
                    s.hist_width_bars = (frac * (1u64 << k) as f64) as f32;
                    let n = lp.level_len(k);
                    let off = lp.offsets[k];
                    let a = (lo >> k).min(n);
                    let b = hi.div_ceil(1 << k).min(n);
                    ((off + a) as u32, (off + b) as u32)
                }
                _ => lp.segment_range(k, lo, hi),
            };
            DrawCmd::Line { kind: o.kind, data: o.line.clone(), range, uniforms: Uniforms::new(&t, &s) }
        })
        .collect()
}

/// An indicator sub-pane (RSI, MACD): lines and histograms with their own scale.
pub fn sub_pane(
    overlays: &[Overlay],
    v: &Viewport,
    height_px: f64,
    fixed: Option<(f64, f64)>,
    style: Style,
) -> PaneOutput {
    let len = overlays.first().map(|o| o.line.pyramid.level_len(0)).unwrap_or(0);
    let (lo, hi) = v.visible_range(len);
    let (l, h) = fixed.unwrap_or_else(|| {
        let mut ext: Option<(f64, f64)> = None;
        for o in overlays {
            let k = level_for(v.bar_px, o.line.pyramid.levels());
            if let Some((a, b)) = o.line.pyramid.extent(k, lo, hi) {
                ext = Some(ext.map_or((a, b), |(x, y)| (x.min(a), y.max(b))));
            }
        }
        let (a, b) = ext.unwrap_or((-1.0, 1.0));
        // Histograms and fills grow from zero, so keep zero in range.
        if overlays.iter().any(|o| matches!(o.kind, LineKind::Hist | LineKind::Fill)) {
            (a.min(0.0), b.max(0.0))
        } else {
            (a, b)
        }
    });
    let scale = if fixed.is_some() {
        PriceScale::new(l, h, false, height_px)
    } else {
        PriceScale::padded(l, h, 0.1, false, height_px)
    };
    let t = transform(v, height_px, &scale, style.pixels_per_point);
    let cmds = overlay_cmds(overlays, v, &t, &style);
    PaneOutput { cmds, scale, level: 0, vol_max: 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tr_core::{Timeframe, sample};

    #[test]
    fn price_pane_bounds_instances_for_huge_series() {
        let bars = sample::generate("X", Timeframe::M1, 1_000_000, 1_790_000_000);
        let s = SeriesGpu::build(&bars);
        let v = Viewport { right: 1_000_000.0, bar_px: 0.0016, width_px: 1600.0 };
        let out = price_pane(&PricePane {
            series: &s,
            chart_type: ChartType::Candles,
            show_volume: true,
            overlays: &[],
            viewport: &v,
            height_px: 800.0,
            log: false,
            manual_scale: None,
            style: Style::default(),
        });
        assert_eq!(out.cmds.len(), 2);
        for c in &out.cmds {
            if let DrawCmd::Candles { range, .. } | DrawCmd::Volume { range, .. } = c {
                assert!(range.1 - range.0 <= 810, "{range:?}");
                assert!(range.1 > range.0);
            }
        }
        // Autoscale contains every visible bar.
        let lo = bars.iter().map(|b| b.low).fold(f64::INFINITY, f64::min);
        let hi = bars.iter().map(|b| b.high).fold(f64::NEG_INFINITY, f64::max);
        assert!(out.scale.lo <= lo && out.scale.hi >= hi);
    }

    #[test]
    fn rebase_uses_first_visible_common_value() {
        let base = line_ref(&[f64::NAN, 10.0, 20.0, 30.0]);
        let other = line_ref(&[1.0, f64::NAN, 4.0, 5.0]);
        assert_eq!(rebase_factor(&base, &other, 0, 4), 5.0);
        assert_eq!(rebase_factor(&base, &other, 3, 4), 6.0);
    }

    #[test]
    fn retina_scales_pane_and_strokes() {
        let s = SeriesGpu::build(&sample::generate("X", Timeframe::D1, 500, 1_790_000_000));
        let v = Viewport::fit_last(500, 100, 400.0);
        let pane = |ppp: f32| {
            price_pane(&PricePane {
                series: &s,
                chart_type: ChartType::Candles,
                show_volume: false,
                overlays: &[],
                viewport: &v,
                height_px: 300.0,
                log: false,
                manual_scale: None,
                style: Style { pixels_per_point: ppp, ..Style::default() },
            })
        };
        let (one, two) = (pane(1.0), pane(2.0));
        let u = |o: &PaneOutput| match &o.cmds[0] {
            DrawCmd::Candles { uniforms, .. } => *uniforms,
            _ => unreachable!(),
        };
        assert_eq!(u(&two).rect[0], 2.0 * u(&one).rect[0]);
        assert_eq!(u(&two).rect[1], 600.0);
        assert_eq!(u(&two).params[1], 2.0 * u(&one).params[1]);
        // Price mapping (points) is unchanged.
        assert_eq!(one.scale, two.scale);
    }

    #[test]
    fn autoscale_is_what_price_pane_uses() {
        let s = SeriesGpu::build(&sample::generate("X", Timeframe::D1, 500, 1_790_000_000));
        let v = Viewport::fit_last(500, 120, 600.0);
        for (show_volume, log) in [(true, false), (false, true)] {
            let p = PricePane {
                series: &s,
                chart_type: ChartType::Candles,
                show_volume,
                overlays: &[],
                viewport: &v,
                height_px: 300.0,
                log,
                manual_scale: None,
                style: Style::default(),
            };
            assert_eq!(autoscale(&p), price_pane(&p).scale);
        }
    }

    #[test]
    fn reveal_sweeps_relative_to_the_left_edge() {
        use crate::gpu::Reveal;
        let s = SeriesGpu::build(&sample::generate("X", Timeframe::D1, 500, 1_790_000_000));
        let v = Viewport::fit_last(500, 100, 400.0);
        let anim = |reveal: Reveal| {
            let out = price_pane(&PricePane {
                series: &s,
                chart_type: ChartType::Candles,
                show_volume: true,
                overlays: &[],
                viewport: &v,
                height_px: 300.0,
                log: false,
                manual_scale: None,
                style: Style { reveal, ..Style::default() },
            });
            out.cmds
                .iter()
                .map(|c| match c {
                    DrawCmd::Candles { uniforms, .. } | DrawCmd::Volume { uniforms, .. } => uniforms.anim,
                    DrawCmd::Line { uniforms, .. } => uniforms.anim,
                })
                .collect::<Vec<_>>()
        };
        // Done: everything drawn, nothing clipped.
        for a in anim(Reveal::DONE) {
            assert_eq!(a, [1e9, 1.0, 1e9, 1.0]);
        }
        assert_eq!(Reveal::sweep(1.0, v.left(), v.visible_bars()), Reveal::DONE);
        // Start: the front sits on the left edge, so no bar has grown yet.
        let start = Reveal::sweep(0.0, v.left(), v.visible_bars());
        let left_lo = (v.left() - v.left().floor()) as f32;
        for a in anim(start) {
            assert!((a[0] - left_lo).abs() < 1e-3, "{a:?}");
            assert!(a[2] < a[0] && a[1] >= 2.0);
        }
        // Midway the front is inside the visible range, lines trailing it.
        let mid = Reveal::sweep(0.5, v.left(), v.visible_bars());
        assert!(mid.front > v.left() && mid.front < v.right && mid.line_front < mid.front);
    }

    #[test]
    fn sub_pane_fixed_and_auto() {
        let vals: Vec<f64> = (0..100).map(|i| (i as f64 * 0.2).sin()).collect();
        let o = Overlay { line: line_ref(&vals), color: [1.0; 4], kind: LineKind::Hist, scale: 1.0 };
        let v = Viewport::fit_last(100, 100, 500.0);
        let out = sub_pane(std::slice::from_ref(&o), &v, 100.0, None, Style::default());
        assert!(out.scale.lo < -0.9 && out.scale.hi > 0.9);
        let fixed = sub_pane(&[o], &v, 100.0, Some((0.0, 100.0)), Style::default());
        assert_eq!((fixed.scale.lo, fixed.scale.hi), (0.0, 100.0));
    }

    #[test]
    fn dots_draw_one_instance_per_point_and_scale_the_price_pane() {
        let bars = sample::generate("X", Timeframe::D1, 200, 1_790_000_000);
        let s = SeriesGpu::build(&bars);
        let v = Viewport::fit_last(200, 200, 800.0);
        let top = bars.iter().map(|b| b.high).fold(f64::MIN, f64::max) * 2.0;
        let mut marks = vec![f64::NAN; 200];
        marks[150] = top;
        let o = Overlay { line: line_ref(&marks), color: [1.0; 4], kind: LineKind::Dots, scale: 1.0 };
        let out = price_pane(&PricePane {
            series: &s,
            chart_type: ChartType::Candles,
            show_volume: false,
            overlays: std::slice::from_ref(&o),
            viewport: &v,
            height_px: 400.0,
            log: false,
            manual_scale: None,
            style: Style::default(),
        });
        let Some(DrawCmd::Line { kind: LineKind::Dots, range, .. }) = out.cmds.last() else { panic!("no dots drawn") };
        assert_eq!(range.1 - range.0, 200);
        assert!(out.scale.hi >= top);
    }

    #[test]
    fn zoomed_out_markers_keep_sparse_points() {
        let bars = sample::generate("X", Timeframe::M1, 4096, 1_790_000_000);
        let s = SeriesGpu::build(&bars);
        let v = Viewport { right: 4096.0, bar_px: 0.2, width_px: 800.0 };
        let top = bars.iter().map(|b| b.high).fold(f64::MIN, f64::max) * 2.0;
        // A fractal-like marker whose neighbours are blank.
        let mut marks = vec![f64::NAN; 4096];
        marks[2002] = top;
        let pane = |line: LineRef| {
            let o = Overlay { line, color: [1.0; 4], kind: LineKind::Dots, scale: 1.0 };
            let out = price_pane(&PricePane {
                series: &s,
                chart_type: ChartType::Candles,
                show_volume: false,
                overlays: std::slice::from_ref(&o),
                viewport: &v,
                height_px: 400.0,
                log: false,
                manual_scale: None,
                style: Style::default(),
            });
            let Some(DrawCmd::Line { data, range, .. }) = out.cmds.last() else { panic!("no dots drawn") };
            let drawn = data.pyramid.data[range.0 as usize..range.1 as usize].contains(&[2002.5, top as f32]);
            (out.level, drawn, out.scale.hi >= top)
        };
        let (level, drawn, scaled) = pane(marker_ref(&marks));
        assert!(level > 0 && drawn && scaled, "level {level}: drawn {drawn}, scaled {scaled}");
        assert_eq!(pane(line_ref(&marks)), (level, false, false), "line reduction drops the marker");
    }

    #[test]
    fn steps_draw_one_full_bar_dash_per_point() {
        let s = SeriesGpu::build(&sample::generate("X", Timeframe::H4, 200, 1_790_000_000));
        let levels: Vec<f64> = (0..200).map(|i| if i / 2 % 2 == 0 { 1.0 } else { 1e6 }).collect();
        let o = Overlay { line: line_ref(&levels), color: [1.0; 4], kind: LineKind::Steps, scale: 1.0 };
        for (bar_px, group) in [(4.0, 1.0), (0.5, 4.0)] {
            let v = Viewport { right: 200.0, bar_px, width_px: 800.0 };
            let out = price_pane(&PricePane {
                series: &s,
                chart_type: ChartType::Candles,
                show_volume: false,
                overlays: std::slice::from_ref(&o),
                viewport: &v,
                height_px: 400.0,
                log: false,
                manual_scale: None,
                style: Style::default(),
            });
            let Some(DrawCmd::Line { kind: LineKind::Steps, range, uniforms, .. }) = out.cmds.last() else {
                panic!("no steps drawn")
            };
            assert_eq!(uniforms.extra[2], group, "a dash spans its whole group of bars");
            assert_eq!(range.1 - range.0, 200 / group as u32);
            assert!(out.scale.hi >= 1e6);
        }
    }
}
