//! Chart pane: GPU-rendered price/volume/indicator panes with egui-drawn axes, legend,
//! crosshair, drawings, congressional trade markers and earnings markers.

use std::sync::Arc;

use chrono::{Datelike, NaiveDate};
use egui::{
    Align2, Color32, CornerRadius, CursorIcon, Pos2, Rect, RichText, Sense, Stroke, StrokeKind, Ui, UiBuilder, pos2,
    vec2,
};
use tr_chart::drawings::{self, Anchor, Drawing, Shape, Tool};
use tr_chart::gpu::{ChartCallback, Frame, LineKind, LineRef, Reveal, Style};
use tr_chart::scale;
use tr_chart::scene::{self, ChartType, Overlay, PricePane, SeriesGpu};
use tr_chart::viewport::{PriceScale, Viewport};
use tr_congress::{Trade, TxType};
use tr_core::indicators::registry::{self, Category};
use tr_core::{AssetClass, Bar, Security, Timeframe};
use tr_data::earnings::{Earning, Outcome};

use crate::command::{IndicatorSpec, Range};
use crate::congress_view::Action;
use crate::theme::{self, amber, amber_dim, mono, muted, text, ui_font};
use crate::widgets;
use crate::worker::{BuiltIndicator, PaneId, Request, Worker, initial_limit};

const AXIS_W: f32 = 66.0;
const TIME_H: f32 = 16.0;
/// Fallback header row, used when the pane's tab strip has no room for the controls.
const HEADER_H: f32 = 24.0;
const RANGE_H: f32 = 18.0;
/// Height of the interval / chart-type segmented controls.
pub const CONTROL_H: f32 = 20.0;
const DEFAULT_VISIBLE: usize = 160;
/// Oscillator panes shown under the price pane. GPU frame ids are `id * 8 + 1 + k`, so a
/// chart has room for at most seven.
const MAX_SUB_PANES: usize = 6;

/// Length of the load-in sweep across the visible bars.
const REVEAL_SECS: f32 = 0.7;
/// Length of an eased jump (quick ranges, reset, keyboard zoom and pan).
const GLIDE_SECS: f32 = 0.32;
/// Time constant of the autoscale easing.
const SCALE_TAU: f64 = 0.07;

/// A programmatic viewport change in progress; `start` is set on the first frame shown.
#[derive(Debug, Clone, Copy)]
struct Glide {
    from: Viewport,
    to: Viewport,
    start: Option<f64>,
}

impl Glide {
    /// Pans linearly and zooms geometrically, so a zoom feels even all the way.
    fn at(&self, e: f64) -> Viewport {
        let (a, b) = (self.from.bar_px.ln(), self.to.bar_px.ln());
        Viewport {
            right: self.from.right + (self.to.right - self.from.right) * e,
            bar_px: (a + (b - a) * e).exp(),
            ..self.to
        }
    }
}

#[derive(Default)]
struct DrawState {
    tool: Option<Tool>,
    pending: Option<Anchor>,
    selected: Option<usize>,
}

pub struct ChartState {
    pub id: PaneId,
    pub symbol: String,
    pub tf: Timeframe,
    pub chart_type: ChartType,
    pub log: bool,
    pub magnet: bool,
    pub show_volume: bool,
    pub show_congress: bool,
    pub show_earnings: bool,
    pub indicators: Vec<IndicatorSpec>,
    pub drawings: Vec<Drawing>,
    pub compare: Option<String>,

    bars: Arc<Vec<Bar>>,
    gpu: Option<SeriesGpu>,
    generation: u64,
    viewport: Option<Viewport>,
    manual_scale: Option<PriceScale>,
    glide: Option<Glide>,
    /// Shown autoscale bounds (transformed) and whether they are log, easing to the target.
    shown_scale: Option<(f64, f64, bool)>,
    /// Renewed when a load's first bars arrive, which replays the load-in sweep.
    reveal_key: u64,
    /// Renewed when a load's first Congress markers arrive, which pops them in.
    markers_key: u64,
    loading: bool,
    paging: bool,
    exhausted: bool,
    error: Option<String>,
    built: Vec<BuiltIndicator>,
    compare_line: Option<LineRef>,
    compare_scale: f64,
    markers: Vec<Trade>,
    /// Earnings reports for a stock: the next one first, then newest first.
    earnings: Vec<Earning>,
    /// The report whose details card is open (by date), from a click on its badge.
    open_earning: Option<NaiveDate>,
    /// Renewed when a load's earnings arrive, which pops their markers in.
    earnings_key: u64,
    pending_range: Option<Range>,
    draw: DrawState,
    hover_marker: Option<usize>,
    /// Resolved security for `symbol`.
    sec: Security,
    last_live: Option<web_time::Instant>,
    last_indicator_req: Option<web_time::Instant>,
    /// App-level actions requested from this chart (context menu), drained by the app.
    pub actions: Vec<Action>,
    /// False while a pane is being dragged over the workspace: no crosshair or hover cards.
    pub hover_enabled: bool,
    /// The workspace drew this chart's controls in the pane's tab strip, so skip the
    /// header row and give the plot the space.
    pub controls_in_strip: bool,
}

impl ChartState {
    pub fn new(id: PaneId, symbol: &str, tf: Timeframe) -> Self {
        Self {
            id,
            symbol: tr_data::security(symbol).key(),
            tf,
            chart_type: if tr_data::security(symbol).class == AssetClass::Govt {
                ChartType::Line
            } else {
                ChartType::Candles
            },
            log: false,
            magnet: false,
            show_volume: true,
            show_congress: true,
            show_earnings: true,
            indicators: crate::command::default_indicators(),
            drawings: Vec::new(),
            compare: None,
            bars: Arc::new(Vec::new()),
            gpu: None,
            generation: 0,
            viewport: None,
            manual_scale: None,
            glide: None,
            shown_scale: None,
            reveal_key: 0,
            markers_key: 0,
            loading: false,
            paging: false,
            exhausted: false,
            error: None,
            built: Vec::new(),
            compare_line: None,
            compare_scale: 1.0,
            markers: Vec::new(),
            earnings: Vec::new(),
            open_earning: None,
            earnings_key: 0,
            pending_range: None,
            draw: DrawState::default(),
            hover_marker: None,
            sec: tr_data::security(symbol),
            last_live: None,
            last_indicator_req: None,
            actions: Vec::new(),
            hover_enabled: true,
            controls_in_strip: false,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.bars.is_empty()
    }

    /// (Re)load history for the current symbol and timeframe.
    pub fn load(&mut self, w: &Worker) {
        self.generation += 1;
        self.loading = true;
        self.paging = false;
        self.exhausted = false;
        self.error = None;
        self.viewport = None;
        self.manual_scale = None;
        self.glide = None;
        self.compare_line = None;
        self.markers.clear();
        self.earnings.clear();
        self.open_earning = None;
        // Never show the previous symbol's bars under the new title.
        self.bars = Arc::new(Vec::new());
        self.gpu = None;
        self.built.clear();
        w.send(Request::History {
            pane: self.id,
            generation: self.generation,
            symbol: self.symbol.clone(),
            tf: self.tf,
            limit: initial_limit(self.tf),
        });
        self.last_live = None;
        if self.sec.has_congress_trades() {
            w.send(Request::Markers { pane: self.id, symbol: self.symbol.clone() });
        }
        if self.sec.class == AssetClass::Equity {
            w.send(Request::Earnings { pane: self.id, symbol: self.symbol.clone() });
        }
    }

    pub fn set_symbol(&mut self, symbol: &str, w: &Worker) {
        let sec = tr_data::security(symbol);
        if sec.class == AssetClass::Govt {
            self.chart_type = ChartType::Line;
        } else if self.sec.class == AssetClass::Govt {
            self.chart_type = ChartType::Candles;
        }
        self.symbol = sec.key();
        self.sec = sec;
        self.draw = DrawState::default();
        self.load(w);
    }

    pub fn set_timeframe(&mut self, tf: Timeframe, w: &Worker) {
        if tf != self.tf {
            self.tf = tf;
            self.load(w);
        }
    }

    pub fn set_range(&mut self, r: Range, w: &Worker) {
        self.pending_range = Some(r);
        if r.timeframe() != self.tf {
            self.tf = r.timeframe();
            self.load(w);
        } else {
            self.glide(Self::apply_range);
        }
    }

    /// Apply `change` to the view (on top of any glide's destination) and ease to it.
    fn glide(&mut self, change: impl FnOnce(&mut Self)) {
        let shown = self.viewport;
        if let Some(g) = self.glide.take() {
            self.viewport = Some(g.to);
        }
        change(self);
        if let (Some(from), Some(to)) = (shown, self.viewport)
            && from != to
            && !crate::anim::reduced()
        {
            self.viewport = Some(from);
            self.glide = Some(Glide { from, to, start: None });
        }
    }

    fn apply_range(&mut self) {
        let Some(r) = self.pending_range.take() else { return };
        let (Some(v), Some(last)) = (self.viewport.as_mut(), self.bars.last()) else { return };
        let count = match r.span_secs(last.ts) {
            Some(span) => self.bars.len() - self.bars.partition_point(|b| b.ts < last.ts - span),
            None => self.bars.len(),
        };
        *v = Viewport::fit_last(self.bars.len(), count.max(2), v.width_px);
        self.manual_scale = None;
    }

    pub fn toggle_indicator(&mut self, spec: IndicatorSpec, w: &Worker) {
        if let Some(i) = self.indicators.iter().position(|s| s.same_kind(&spec)) {
            if self.indicators[i] == spec {
                self.indicators.remove(i);
            } else {
                self.indicators[i] = spec;
            }
        } else {
            self.indicators.push(spec);
        }
        self.request_indicators(w);
    }

    /// Remove every instance of the indicator `code` (all moving averages of one kind).
    fn remove_indicator(&mut self, code: &str, w: &Worker) {
        self.indicators.retain(|s| s.code != code);
        self.built.retain(|b| b.spec.code != code);
        self.request_indicators(w);
    }

    pub fn clear_indicators(&mut self) {
        self.indicators.clear();
        self.built.clear();
    }

    pub fn set_compare(&mut self, symbol: Option<String>, w: &Worker) {
        self.compare = symbol;
        self.compare_line = None;
        self.request_compare(w);
    }

    pub fn start_tool(&mut self, tool: Tool) {
        self.draw.tool = Some(tool);
        self.draw.pending = None;
    }

    /// Esc: cancel an in-progress drawing, then deselect. Returns whether it consumed the key.
    pub fn cancel(&mut self) -> bool {
        if self.open_earning.take().is_some() {
            return true;
        }
        if self.draw.tool.is_some() || self.draw.selected.is_some() {
            self.draw = DrawState::default();
            true
        } else {
            false
        }
    }

    pub fn delete_selected(&mut self) -> bool {
        match self.draw.selected.take() {
            Some(i) if i < self.drawings.len() => {
                self.drawings.remove(i);
                true
            }
            _ => false,
        }
    }

    pub fn reset_view(&mut self) {
        self.glide(|s| {
            if let Some(v) = s.viewport.as_mut() {
                *v = Viewport::fit_last(s.bars.len(), DEFAULT_VISIBLE, v.width_px);
            }
            s.manual_scale = None;
        });
    }

    pub fn pan_bars(&mut self, n: f64) {
        self.glide(|s| {
            if let Some(v) = s.viewport.as_mut() {
                v.pan_bars(n);
                v.clamp(s.bars.len());
            }
        });
    }

    pub fn zoom(&mut self, factor: f64) {
        self.glide(|s| {
            if let Some(v) = s.viewport.as_mut() {
                let anchor = v.width_px * 0.85;
                v.zoom(factor, anchor);
                v.clamp(s.bars.len());
            }
        });
    }

    /// Ease the shown autoscale range toward `target`, so the axis glides as bars scroll
    /// in rather than jumping. Snaps on new data and when switching log scale.
    fn ease_scale(&mut self, ctx: &egui::Context, target: PriceScale) -> PriceScale {
        let (a, b) = target.bounds_t();
        let (lo, hi) = match self.shown_scale {
            Some((lo, hi, log)) if log == target.log && !crate::anim::reduced() => {
                let dt = ctx.input(|i| i.stable_dt).clamp(0.0, 0.1) as f64;
                let k = 1.0 - (-dt / SCALE_TAU).exp();
                let (lo, hi) = (lo + (a - lo) * k, hi + (b - hi) * k);
                if (lo - a).abs() + (hi - b).abs() < (b - a).abs() * 1e-4 {
                    (a, b)
                } else {
                    ctx.request_repaint();
                    (lo, hi)
                }
            }
            _ => (a, b),
        };
        self.shown_scale = Some((lo, hi, target.log));
        let inv = |x: f64| if target.log { x.exp() } else { x };
        PriceScale { lo: inv(lo), hi: inv(hi), ..target }
    }

    fn request_indicators(&self, w: &Worker) {
        if !self.bars.is_empty() {
            w.send(Request::Indicators {
                pane: self.id,
                generation: self.generation,
                bars: self.bars.clone(),
                tf: self.tf,
                specs: self.indicators.clone(),
                palette: theme::palette(),
            });
        }
    }

    fn request_compare(&self, w: &Worker) {
        if let (Some(sym), false) = (&self.compare, self.bars.is_empty()) {
            w.send(Request::Compare {
                pane: self.id,
                generation: self.generation,
                symbol: sym.clone(),
                tf: self.tf,
                base: self.bars.clone(),
            });
        }
    }

    // ---- worker responses -------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    pub fn on_history(
        &mut self,
        generation: u64,
        bars: Arc<Vec<Bar>>,
        gpu: SeriesGpu,
        prepended: usize,
        from_cache: bool,
        exhausted: bool,
        full: bool,
        live: bool,
        w: &Worker,
    ) {
        if generation != self.generation {
            return;
        }
        if live {
            self.last_live = Some(web_time::Instant::now());
        }
        let old_len = self.bars.len();
        let new_len = bars.len();
        if old_len == 0 && new_len > 0 {
            self.reveal_key = crate::anim::fresh_key();
            self.shown_scale = None;
        }
        // Keep a glide in progress on the same bars as the data shifts under it.
        if let Some(g) = self.glide.as_mut() {
            let shift = if prepended > 0 { prepended as f64 } else { new_len as f64 - old_len as f64 };
            g.from.right += shift;
            g.to.right += shift;
        }
        self.bars = bars;
        self.gpu = Some(gpu);
        self.paging = false;
        self.exhausted = end_of_history(self.exhausted, exhausted, prepended, full);
        if !from_cache {
            self.loading = false;
        }
        self.error = None;
        match self.viewport.as_mut() {
            None => {
                self.viewport = Some(Viewport::fit_last(new_len, DEFAULT_VISIBLE, 800.0));
            }
            Some(v) if prepended > 0 => v.right += prepended as f64,
            Some(v) => {
                // A refresh replaced the data: keep the same distance from the latest bar.
                let from_end = old_len as f64 - v.right;
                v.right = new_len as f64 - from_end;
            }
        }
        self.apply_range();
        // Live ticks only move the last bar: refresh indicators at most every 1.5 s and
        // leave the compare series alone. (A live feed filling a gap arrives as a refresh.)
        let due = self.last_indicator_req.is_none_or(|t| t.elapsed().as_millis() >= 1500);
        if !live || due {
            self.request_indicators(w);
            self.last_indicator_req = Some(web_time::Instant::now());
        }
        if !live {
            self.request_compare(w);
        }
    }

    pub fn on_history_error(&mut self, generation: u64, error: String) {
        if generation == self.generation {
            self.loading = false;
            self.paging = false;
            self.exhausted = true;
            if self.bars.is_empty() {
                self.error = Some(error);
            }
        }
    }

    pub fn on_indicators(&mut self, generation: u64, built: Vec<BuiltIndicator>) {
        if generation == self.generation {
            self.built = built;
        }
    }

    pub fn on_compare(&mut self, generation: u64, symbol: &str, line: LineRef) {
        if generation == self.generation && self.compare.as_deref() == Some(symbol) {
            self.compare_line = Some(line);
        }
    }

    pub fn on_markers(&mut self, symbol: &str, trades: Vec<Trade>) {
        if symbol == self.symbol {
            if self.markers.is_empty() && !trades.is_empty() {
                self.markers_key = crate::anim::fresh_key();
            }
            self.markers = trades;
        }
    }

    pub fn on_earnings(&mut self, symbol: &str, earnings: Vec<Earning>) {
        if symbol == self.symbol {
            if self.earnings.is_empty() && !earnings.is_empty() {
                self.earnings_key = crate::anim::fresh_key();
            }
            self.earnings = earnings;
        }
    }

    // ---- coordinate helpers -----------------------------------------------------------

    /// Fractional bar index for a timestamp (extrapolated past either end).
    fn index_of_ts(&self, ts: i64) -> f64 {
        let b = &self.bars;
        let (Some(first), Some(last)) = (b.first(), b.last()) else { return 0.0 };
        let step = self.tf.seconds() as f64;
        if ts > last.ts {
            return (b.len() - 1) as f64 + (ts - last.ts) as f64 / step + 0.5;
        }
        if ts < first.ts {
            return (ts - first.ts) as f64 / step + 0.5;
        }
        b.partition_point(|x| x.ts <= ts).saturating_sub(1) as f64 + 0.5
    }

    fn ts_of_index(&self, idx: f64) -> i64 {
        let b = &self.bars;
        let Some(last) = b.last() else { return 0 };
        let i = idx.floor();
        if i >= b.len() as f64 {
            return last.ts + ((i - (b.len() - 1) as f64) * self.tf.seconds() as f64) as i64;
        }
        if i < 0.0 {
            return b[0].ts + (i * self.tf.seconds() as f64) as i64;
        }
        b[i as usize].ts
    }

    // ---- rendering --------------------------------------------------------------------

    /// Draw the pane into `rect`. Returns true if the pane was clicked (focus request).
    /// Width of [`Self::controls`] (interval and chart-type segmented controls).
    pub fn controls_width(ui: &Ui) -> f32 {
        let tf: Vec<&str> = Timeframe::ALL.iter().map(|t| t.label()).collect();
        let ct: Vec<&str> = ChartType::ALL.iter().map(|c| c.label()).collect();
        widgets::segmented_width(ui, &tf, CONTROL_H) + widgets::segmented_width(ui, &ct, CONTROL_H) + 8.0
    }

    /// Interval and chart-type controls, laid out left to right in `ui`. Returns whether
    /// one was clicked.
    pub fn controls(&mut self, ui: &mut Ui, w: &Worker) -> bool {
        let mut clicked = false;
        let tf_labels: Vec<&str> = Timeframe::ALL.iter().map(|t| t.label()).collect();
        let selected_tf = Timeframe::ALL.iter().position(|t| *t == self.tf);
        if let Some(i) = widgets::segmented(ui, (self.id, "tf"), &tf_labels, selected_tf, CONTROL_H) {
            self.set_timeframe(Timeframe::ALL[i], w);
            clicked = true;
        }
        ui.add_space(8.0);
        let ct_labels: Vec<&str> = ChartType::ALL.iter().map(|c| c.label()).collect();
        let selected_ct = ChartType::ALL.iter().position(|c| *c == self.chart_type);
        if let Some(i) = widgets::segmented(ui, (self.id, "ct"), &ct_labels, selected_ct, CONTROL_H) {
            self.chart_type = ChartType::ALL[i];
            clicked = true;
        }
        clicked
    }

    pub fn show(&mut self, ui: &mut Ui, rect: Rect, focused: bool, w: &Worker) -> bool {
        let painter = ui.painter_at(rect);
        // Sits under a tab strip inside a card: square top, rounded bottom.
        let r = theme::CARD_RADIUS.saturating_sub(1);
        painter.rect_filled(rect, CornerRadius { nw: 0, ne: 0, sw: r, se: r }, theme::card());
        let mut clicked = false;

        // Controls live in the tab strip when it has room; otherwise a slim header row.
        let header_h = if self.controls_in_strip { 0.0 } else { HEADER_H };
        let header = Rect::from_min_size(rect.min, vec2(rect.width(), header_h));
        if header_h > 0.0 {
            painter.hline(header.x_range(), header.bottom(), Stroke::new(1.0, theme::border()));
            let mut hui = ui.new_child(
                UiBuilder::new()
                    .max_rect(header.shrink2(vec2(8.0, 0.0)))
                    .layout(egui::Layout::left_to_right(egui::Align::Center))
                    .id_salt((self.id, "header")),
            );
            hui.spacing_mut().item_spacing.x = 6.0;
            hui.label(RichText::new(&self.sec.ticker).font(mono(12.0)).color(amber()));
            if hui.available_width() > Self::controls_width(&hui) {
                clicked |= self.controls(&mut hui, w);
            } else {
                // Narrow pane: the same controls as compact menus.
                hui.menu_button(RichText::new(self.tf.label()).font(ui_font(11.5)), |ui| {
                    for tf in Timeframe::ALL {
                        if ui.radio(self.tf == tf, tf.label()).clicked() {
                            self.set_timeframe(tf, w);
                            clicked = true;
                            ui.close();
                        }
                    }
                });
                hui.menu_button(RichText::new(self.chart_type.label()).font(ui_font(11.5)), |ui| {
                    for ct in ChartType::ALL {
                        if ui.radio(self.chart_type == ct, ct.label()).clicked() {
                            self.chart_type = ct;
                            clicked = true;
                            ui.close();
                        }
                    }
                });
            }
        }
        if self.last_live.is_some_and(|t| t.elapsed().as_secs() < 5) {
            // Pulsing live dot in the plot's top-right corner (clear of the price axis labels).
            let pulse = 0.6 + 0.4 * ((ui.input(|i| i.time) * 3.0).sin() as f32).abs();
            let c = pos2(rect.right() - AXIS_W + 12.0, header.bottom() + 10.0);
            painter.circle_filled(c, 3.0, theme::up().gamma_multiply(pulse));
            painter.text(c + vec2(7.0, 0.0), Align2::LEFT_CENTER, "LIVE", ui_font(10.0), theme::up());
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
        }

        // Bottom quick-range row.
        let range_row = Rect::from_min_max(pos2(rect.left(), rect.bottom() - RANGE_H), rect.max);
        let mut rx = range_row.left() + 6.0;
        for r in Range::ALL {
            let lr = Rect::from_min_size(pos2(rx, range_row.top()), vec2(r.label().len() as f32 * 7.4 + 8.0, RANGE_H));
            let resp = ui.interact(lr, ui.id().with((self.id, "range", r.label())), Sense::click());
            painter.text(
                lr.center(),
                Align2::CENTER_CENTER,
                r.label(),
                mono(11.0),
                if resp.hovered() { amber() } else { muted() },
            );
            if resp.clicked() {
                self.set_range(r, w);
                clicked = true;
            }
            rx = lr.right();
        }
        let status = match (&self.error, self.loading, self.paging) {
            (Some(e), _, _) => format!("ERR {e}"),
            (_, true, _) => "loading…".into(),
            (_, _, true) => "loading history…".into(),
            _ => format!(
                "{} bars{}{}",
                self.bars.len(),
                if self.log { "  LOG" } else { "" },
                if self.magnet { "  MAGNET" } else { "" }
            ),
        };
        painter.text(
            pos2(range_row.right() - 6.0, range_row.center().y),
            Align2::RIGHT_CENTER,
            status,
            mono(10.5),
            muted(),
        );

        let body = Rect::from_min_max(pos2(rect.left(), header.bottom()), pos2(rect.right(), range_row.top()));
        let plot_all = Rect::from_min_max(body.min, pos2(body.right() - AXIS_W, body.bottom() - TIME_H));
        if plot_all.width() < 40.0 || plot_all.height() < 40.0 {
            return clicked;
        }

        let Some(gpu) = self.gpu.clone() else {
            let msg = self.error.clone().unwrap_or_else(|| format!("Loading {} {}…", self.symbol, self.tf));
            painter.text(plot_all.center(), Align2::CENTER_CENTER, msg, mono(13.0), amber_dim());
            return clicked;
        };

        // Split plot height between the price pane and oscillator sub-panes.
        let pane_specs: Vec<usize> = (0..self.built.len()).filter(|&i| self.built[i].spec.is_pane()).collect();
        let n_sub = pane_specs.len().min(MAX_SUB_PANES);
        // Sub-panes share at most 60% of the plot so prices stay readable.
        let sub_h = if n_sub > 0 {
            (plot_all.height() * 0.2).clamp(50.0, 160.0).min(plot_all.height() * 0.6 / n_sub as f32)
        } else {
            0.0
        };
        let price_rect =
            Rect::from_min_max(plot_all.min, pos2(plot_all.right(), plot_all.bottom() - sub_h * n_sub as f32));

        let len = self.bars.len();
        if let Some(g) = self.glide.as_mut() {
            let now = ui.input(|i| i.time);
            let t = crate::anim::since(ui.ctx(), now - *g.start.get_or_insert(now), GLIDE_SECS);
            self.viewport = Some(g.at(crate::anim::ease_out(t) as f64));
            if t >= 1.0 {
                self.glide = None;
            }
        }
        let v =
            self.viewport.get_or_insert_with(|| Viewport::fit_last(len, DEFAULT_VISIBLE, price_rect.width() as f64));
        v.set_width(price_rect.width() as f64);

        // ---- input ----
        let plot_resp = ui.interact(plot_all, ui.id().with((self.id, "plot")), Sense::click_and_drag());
        let axis_rect =
            Rect::from_min_max(pos2(price_rect.right(), price_rect.top()), pos2(body.right(), price_rect.bottom()));
        let axis_resp = ui.interact(axis_rect, ui.id().with((self.id, "paxis")), Sense::click_and_drag());
        let time_rect =
            Rect::from_min_max(pos2(plot_all.left(), plot_all.bottom()), pos2(plot_all.right(), body.bottom()));
        let time_resp = ui.interact(time_rect, ui.id().with((self.id, "taxis")), Sense::click_and_drag());
        if plot_resp.clicked() || plot_resp.drag_started() || axis_resp.clicked() || time_resp.clicked() {
            clicked = true;
        }
        let hover = plot_resp.hover_pos().filter(|p| self.hover_enabled && plot_all.contains(*p));
        if self.draw.tool.is_none() {
            plot_resp.context_menu(|ui| self.context_menu(ui, w));
        }

        if plot_resp.hovered() || axis_resp.hovered() || time_resp.hovered() {
            let (scroll, zoom_delta) = ui.input(|i| (i.smooth_scroll_delta, i.zoom_delta()));
            if scroll != egui::Vec2::ZERO || zoom_delta != 1.0 {
                self.glide = None;
            }
            let v = self.viewport.as_mut().expect("viewport");
            if let Some(p) = ui.input(|i| i.pointer.hover_pos()) {
                let anchor = (p.x - price_rect.left()).clamp(0.0, price_rect.width()) as f64;
                if scroll.y != 0.0 {
                    v.zoom((scroll.y as f64 * 0.003).exp(), anchor);
                }
                if zoom_delta != 1.0 {
                    v.zoom(zoom_delta as f64, anchor);
                }
            }
            if scroll.x != 0.0 {
                v.pan_px(scroll.x as f64);
            }
        }
        if (self.draw.tool.is_none() && plot_resp.dragged()) || time_resp.dragged() {
            self.glide = None;
        }
        if self.draw.tool.is_none() && plot_resp.dragged() {
            let d = plot_resp.drag_delta();
            self.viewport.as_mut().expect("viewport").pan_px(d.x as f64);
            if let Some(s) = self.manual_scale.as_mut() {
                s.pan_px(d.y as f64);
            }
            ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
        }
        if time_resp.dragged() {
            let v = self.viewport.as_mut().expect("viewport");
            let w_px = v.width_px;
            v.zoom((-time_resp.drag_delta().x as f64 * 0.005).exp(), w_px);
            ui.ctx().set_cursor_icon(CursorIcon::ResizeHorizontal);
        }
        if time_resp.double_clicked() || plot_resp.double_clicked() && self.draw.tool.is_none() {
            self.reset_view();
        }
        let v = *self.viewport.as_ref().expect("viewport");
        let mut v = v;
        v.clamp(len);
        self.viewport = Some(v);

        // Page older history when the left edge approaches loaded data.
        if !self.paging && !self.loading && !self.exhausted && v.left() < v.visible_bars() * 0.3 {
            self.paging = true;
            w.send(Request::MoreHistory {
                pane: self.id,
                generation: self.generation,
                symbol: self.symbol.clone(),
                tf: self.tf,
                existing: self.bars.clone(),
                limit: initial_limit(self.tf),
            });
        }

        // ---- price pane GPU frame ----
        let mut overlays: Vec<Overlay> = self
            .built
            .iter()
            .filter(|b| !b.spec.is_pane())
            .flat_map(|b| b.overlays.iter().map(|o| o.1.clone()))
            .collect();
        if let Some(c) = &self.compare_line {
            let (lo, hi) = self.viewport.as_ref().map(|v| v.visible_range(len)).unwrap_or((0, len));
            let scale = scene::rebase_factor(&gpu.closes, c, lo, hi);
            self.compare_scale = scale;
            overlays.push(Overlay {
                line: c.clone(),
                color: theme::rgba(theme::series(5)),
                kind: LineKind::Line,
                scale,
            });
        }
        let reveal_t =
            crate::anim::progress(ui.ctx(), egui::Id::new(("chart-reveal", self.id)), self.reveal_key, REVEAL_SECS);
        let reveal = Reveal { opacity: ui.opacity(), ..Reveal::sweep(reveal_t, v.left(), v.visible_bars()) };
        let style = Style {
            up: theme::rgba(theme::candle_up()),
            down: theme::rgba(theme::candle_down()),
            color: theme::rgba(amber()),
            pixels_per_point: ui.ctx().pixels_per_point(),
            reveal,
            ..Style::default()
        };
        let mut pane = PricePane {
            series: &gpu,
            chart_type: self.chart_type,
            show_volume: self.show_volume,
            overlays: &overlays,
            viewport: &v,
            height_px: price_rect.height() as f64,
            log: self.log,
            manual_scale: self.manual_scale.map(|mut s| {
                s.height_px = price_rect.height() as f64;
                s
            }),
            style,
        };
        pane.manual_scale = Some(match pane.manual_scale {
            Some(s) => {
                let (a, b) = s.bounds_t();
                self.shown_scale = Some((a, b, s.log));
                s
            }
            None => self.ease_scale(ui.ctx(), scene::autoscale(&pane)),
        });
        let out = scene::price_pane(&pane);
        let mut scale = out.scale;
        scale.height_px = price_rect.height() as f64;
        if axis_resp.dragged() {
            let mut s = self.manual_scale.unwrap_or(scale);
            s.stretch((axis_resp.drag_delta().y as f64 * -0.01).exp());
            self.manual_scale = Some(s);
            ui.ctx().set_cursor_icon(CursorIcon::ResizeVertical);
        }
        if axis_resp.double_clicked() {
            self.manual_scale = None;
        }

        // Grid first (under the GPU geometry is not possible inside one pass, so draw it
        // faintly; candles are opaque on top visually).
        let price_painter = ui.painter_at(price_rect);
        let ticks = scale::price_ticks(scale.lo, scale.hi, (price_rect.height() / 42.0).max(2.0) as usize);
        let step = if ticks.len() > 1 { ticks[1] - ticks[0] } else { 1.0 };
        let dec = scale::decimals_for(step).max(self.sec.decimals(scale.hi).min(3));
        for t in &ticks {
            let y = price_rect.top() + scale.y_of(*t) as f32;
            price_painter.hline(price_rect.x_range(), y, Stroke::new(1.0, theme::grid()));
        }
        let frame_no = ui.ctx().cumulative_frame_nr();
        let frame = Frame { pane: self.id * 8, frame_no, cmds: out.cmds };
        ui.painter().add(tr_chart::egui_wgpu::Callback::new_paint_callback(price_rect, ChartCallback(Arc::new(frame))));

        // Sub panes.
        let mut sub_rects = Vec::new();
        for (k, &bi) in pane_specs.iter().take(n_sub).enumerate() {
            let b = &self.built[bi];
            let top = price_rect.bottom() + sub_h * k as f32;
            let r = Rect::from_min_max(pos2(plot_all.left(), top), pos2(plot_all.right(), top + sub_h));
            let def = b.spec.def();
            let ovs: Vec<Overlay> = b.overlays.iter().map(|o| o.1.clone()).collect();
            let mut s = style;
            s.thickness_px = 1.3;
            let so = scene::sub_pane(&ovs, &v, r.height() as f64, def.and_then(|d| d.range), s);
            let p = ui.painter_at(r);
            p.hline(r.x_range(), r.top(), Stroke::new(1.0, theme::border()));
            for &lvl in def.map_or(&[][..], |d| d.levels) {
                let y = r.top() + so.scale.y_of(lvl) as f32;
                if r.y_range().contains(y) {
                    p.hline(r.x_range(), y, Stroke::new(1.0, muted().gamma_multiply(0.35)));
                }
            }
            let frame = Frame { pane: self.id * 8 + 1 + k as u64, frame_no, cmds: so.cmds };
            ui.painter().add(tr_chart::egui_wgpu::Callback::new_paint_callback(r, ChartCallback(Arc::new(frame))));
            sub_rects.push((r, bi, so.scale));
        }

        // ---- axes ----
        let axis_painter = ui.painter_at(Rect::from_min_max(pos2(plot_all.right(), body.top()), body.max));
        axis_painter.vline(plot_all.right(), body.y_range(), Stroke::new(1.0, theme::border()));
        let positive_prices = self.bars.last().is_some_and(|b| b.low > 0.0);
        for t in &ticks {
            let y = price_rect.top() + scale.y_of(*t) as f32;
            // The volume headroom can dip below zero; don't label impossible prices.
            if positive_prices && *t < 0.0 {
                continue;
            }
            if y > price_rect.top() + 6.0 && y < price_rect.bottom() - 6.0 {
                axis_painter.text(
                    pos2(plot_all.right() + 6.0, y),
                    Align2::LEFT_CENTER,
                    scale::format_price(*t, dec),
                    mono(11.0),
                    muted(),
                );
            }
        }
        for (r, bi, s) in &sub_rects {
            let label = self.built[*bi].spec.label();
            ui.painter_at(*r).text(r.left_top() + vec2(6.0, 3.0), Align2::LEFT_TOP, label, mono(10.5), muted());
            for t in scale::price_ticks(s.lo, s.hi, 3) {
                let y = r.top() + s.y_of(t) as f32;
                if y > r.top() + 8.0 && y < r.bottom() - 4.0 {
                    axis_painter.text(
                        pos2(plot_all.right() + 6.0, y),
                        Align2::LEFT_CENTER,
                        scale::format_value(t),
                        mono(10.5),
                        muted(),
                    );
                }
            }
        }
        // Time axis labels (sampled so 1M visible bars stays cheap).
        let (lo, hi) = v.visible_range(len);
        let time_painter = ui.painter_at(time_rect);
        time_painter.hline(time_rect.x_range(), time_rect.top(), Stroke::new(1.0, theme::border()));
        if hi > lo {
            let stride = ((hi - lo) / 600).max(1);
            let idxs: Vec<usize> = (lo..hi).step_by(stride).collect();
            let times: Vec<i64> = idxs.iter().map(|&i| self.bars[i].ts).collect();
            let gap = 90.0 / (v.bar_px * stride as f64);
            for (j, label) in scale::time_ticks(&times, 0, times.len(), gap, self.tf) {
                let x = price_rect.left() + v.x_center(idxs[j]) as f32;
                ui.painter_at(plot_all).vline(x, plot_all.y_range(), Stroke::new(1.0, theme::grid()));
                time_painter.text(pos2(x, time_rect.center().y), Align2::CENTER_CENTER, label, mono(10.5), muted());
            }
        }

        // Last price line + axis tag, fading in as the sweep reaches the last bar.
        let tag_fade = crate::anim::span(reveal_t, 0.8, 1.0);
        if let Some(last) = self.bars.last()
            && tag_fade > 0.0
        {
            let y = price_rect.top() + scale.y_of(last.close) as f32;
            if price_rect.y_range().contains(y) {
                let col = if last.is_up() { theme::up() } else { theme::down() };
                let mut xx = price_rect.left();
                while xx < price_rect.right() {
                    price_painter.hline(
                        xx..=(xx + 3.0).min(price_rect.right()),
                        y,
                        Stroke::new(1.0, col.gamma_multiply(0.6 * tag_fade)),
                    );
                    xx += 6.0;
                }
                let tag = Rect::from_min_size(pos2(plot_all.right() + 1.0, y - 8.0), vec2(AXIS_W - 2.0, 16.0));
                axis_painter.rect_filled(tag, 0.0, col.gamma_multiply(tag_fade));
                axis_painter.text(
                    tag.left_center() + vec2(5.0, 0.0),
                    Align2::LEFT_CENTER,
                    scale::format_price(last.close, self.sec.decimals(last.close)),
                    mono(11.0),
                    Color32::BLACK.gamma_multiply(tag_fade),
                );
            }
        }

        // ---- drawings ----
        self.paint_drawings(ui, price_rect, &v, &scale, hover, &plot_resp);

        // ---- congress markers ----
        self.hover_marker = None;
        if self.show_congress {
            self.paint_markers(ui, price_rect, &v, &scale, hover, &reveal);
        }
        if self.show_earnings {
            self.paint_earnings(ui, price_rect, &v, hover, &reveal, &plot_resp);
        }

        // ---- crosshair + legend ----
        let hovered_bar = hover.and_then(|p| v.bar_at((p.x - price_rect.left()) as f64, len));
        if let Some(p) = hover {
            let ch = Stroke::new(1.0, Color32::from_gray(110));
            let plot_painter = ui.painter_at(plot_all);
            let xi = hovered_bar.map(|i| price_rect.left() + v.x_center(i) as f32).unwrap_or(p.x);
            dashed_v(&plot_painter, xi, plot_all.y_range(), ch);
            dashed_h(&plot_painter, plot_all.x_range(), p.y, ch);
            if price_rect.contains(p) {
                let price = scale.price_at((p.y - price_rect.top()) as f64);
                let tag = Rect::from_min_size(pos2(plot_all.right() + 1.0, p.y - 8.0), vec2(AXIS_W - 2.0, 16.0));
                axis_painter.rect_filled(tag, 0.0, Color32::from_gray(70));
                axis_painter.text(
                    tag.left_center() + vec2(5.0, 0.0),
                    Align2::LEFT_CENTER,
                    scale::format_price(price, self.sec.decimals(price)),
                    mono(11.0),
                    text(),
                );
            }
            if let Some(i) = hovered_bar {
                let label = scale::crosshair_label(self.bars[i].ts, self.tf);
                let gw = label.len() as f32 * 7.0 + 10.0;
                let tag = Rect::from_center_size(pos2(xi, time_rect.center().y), vec2(gw, TIME_H));
                time_painter.rect_filled(tag, 0.0, Color32::from_gray(70));
                time_painter.text(tag.center(), Align2::CENTER_CENTER, label, mono(10.5), text());
            }
            if self.draw.tool.is_some() {
                ui.ctx().set_cursor_icon(CursorIcon::Crosshair);
            }
        }
        self.paint_legend(ui, price_rect, hovered_bar.unwrap_or(len.saturating_sub(1)), &sub_rects);
        if matches!(self.sec.class, tr_core::AssetClass::Equity | tr_core::AssetClass::Etf) {
            clicked |= self.trade_buttons(ui, plot_all);
        }

        let _ = focused; // the surrounding card shows focus
        clicked
    }

    /// Buy / Sell buttons in the plot's top-right corner: they open the order ticket.
    fn trade_buttons(&mut self, ui: &mut Ui, plot: Rect) -> bool {
        let h = 20.0;
        let mut x = plot.right() - 8.0;
        let mut clicked = false;
        for (label, cmd, color, tip) in
            [("SELL", "SELL", theme::down(), "Sell (⇧S)"), ("BUY", "BUY", theme::up(), "Buy (⇧B)")]
        {
            let g = ui.painter().layout_no_wrap(label.into(), mono(11.0), Color32::WHITE);
            let w = g.size().x + 18.0;
            let r = Rect::from_min_size(pos2(x - w, plot.top() + 6.0), vec2(w, h));
            x -= w + 6.0;
            let resp = ui.interact(r, ui.id().with((self.id, "trade", label)), Sense::click()).on_hover_text(tip);
            let fill = if resp.hovered() { color } else { color.gamma_multiply(0.8) };
            let p = ui.painter();
            p.rect_filled(r, h / 2.0, fill);
            p.galley(r.center() - g.size() / 2.0, g, Color32::WHITE);
            if resp.clicked() {
                self.actions.push(Action::Command(format!("{cmd} {}", self.symbol)));
                clicked = true;
            }
        }
        clicked
    }

    fn context_menu(&mut self, ui: &mut Ui, w: &Worker) {
        ui.set_min_width(190.0);
        if ui.button("Reset View").clicked() {
            self.reset_view();
            ui.close();
        }
        ui.menu_button("Chart Type", |ui| {
            for ct in ChartType::ALL {
                if ui.radio(self.chart_type == ct, ct.label()).clicked() {
                    self.chart_type = ct;
                    ui.close();
                }
            }
        });
        ui.menu_button("Indicators", |ui| {
            ui.set_min_width(220.0);
            if ui.button("Search Indicators…").clicked() {
                self.actions.push(Action::Command("IND".into()));
                ui.close();
            }
            if !self.indicators.is_empty() {
                ui.separator();
                ui.label(RichText::new("On this chart").small().color(muted()));
                for spec in self.indicators.clone() {
                    if ui.checkbox(&mut true, spec.label()).on_hover_text(spec.name()).clicked() {
                        self.toggle_indicator(spec, w);
                    }
                }
            }
            ui.separator();
            for cat in Category::ALL {
                ui.menu_button(cat.label(), |ui| {
                    ui.set_min_width(260.0);
                    egui::ScrollArea::vertical().max_height(440.0).show(ui, |ui| {
                        for def in registry::all().iter().filter(|d| d.category == cat) {
                            let on = self.indicators.iter().any(|s| s.code == def.code);
                            let hover = IndicatorSpec::of(def).label();
                            if ui.checkbox(&mut on.clone(), def.name).on_hover_text(hover).clicked() {
                                if on {
                                    self.remove_indicator(def.code, w);
                                } else {
                                    self.toggle_indicator(IndicatorSpec::of(def), w);
                                }
                            }
                        }
                    });
                });
            }
            ui.separator();
            if ui.button("Remove All").clicked() {
                self.clear_indicators();
                ui.close();
            }
        });
        ui.menu_button("Draw", |ui| {
            let alt = if cfg!(target_os = "macos") { "⌥" } else { "Alt+" };
            for (tool, key) in
                [(Tool::Trendline, "T"), (Tool::Horizontal, "H"), (Tool::Vertical, "V"), (Tool::Fib, "F")]
            {
                if ui.add(egui::Button::new(tool.label()).shortcut_text(format!("{alt}{key}"))).clicked() {
                    self.start_tool(tool);
                    ui.close();
                }
            }
            if ui.button("Remove All Drawings").clicked() {
                self.drawings.clear();
                ui.close();
            }
        });
        ui.checkbox(&mut self.log, "Log Scale");
        ui.checkbox(&mut self.magnet, "Magnet");
        ui.checkbox(&mut self.show_volume, "Volume");
        ui.separator();
        if ui.button("Add to Watchlist").clicked() {
            self.actions.push(Action::AddToWatchlist(self.symbol.clone()));
            ui.close();
        }
        if self.sec.has_congress_trades() {
            ui.checkbox(&mut self.show_congress, "Congress Trade Markers");
            if ui.button("Who in Congress Traded This…").clicked() {
                self.actions.push(Action::OpenTicker(self.symbol.clone()));
                ui.close();
            }
        }
        if self.sec.class == AssetClass::Equity {
            ui.checkbox(&mut self.show_earnings, "Earnings Markers");
        }
    }

    fn paint_legend(&self, ui: &Ui, price_rect: Rect, i: usize, sub_rects: &[(Rect, usize, PriceScale)]) {
        let Some(b) = self.bars.get(i) else { return };
        let p = ui.painter_at(price_rect);
        let prev = if i > 0 { self.bars[i - 1].close } else { b.open };
        let chg = b.close - prev;
        let col = if chg >= 0.0 { theme::up() } else { theme::down() };
        let dec = self.sec.decimals(b.close);
        let mut x = price_rect.left() + 6.0;
        let y = price_rect.top() + 4.0;
        let put = |text: String, color: Color32, x: &mut f32| {
            let g = p.layout_no_wrap(text, mono(11.5), color);
            p.galley(pos2(*x, y), g.clone(), color);
            *x += g.size().x + 8.0;
        };
        put(format!("{} · {}", self.sec.ticker, self.tf), amber(), &mut x);
        for (k, val) in [("O", b.open), ("H", b.high), ("L", b.low), ("C", b.close)] {
            put(format!("{k}{val:.dec$}"), col, &mut x);
        }
        put(format!("{:+.dec$} ({:+.2}%)", chg, if prev != 0.0 { chg / prev * 100.0 } else { 0.0 }), col, &mut x);
        put(format!("Vol {}", scale::format_volume(b.volume)), muted(), &mut x);

        // Overlay indicator values, one per line.
        let mut ly = y + 16.0;
        let value_at = |l: &LineRef| l.pyramid.data.get(i).map(|p| p[1] as f64).filter(|v| v.is_finite());
        for bi in self.built.iter().filter(|b| !b.spec.is_pane()) {
            let vals: Vec<String> = bi
                .overlays
                .iter()
                .filter_map(|(_, o)| value_at(&o.line).map(|v| format!("{v:.prec$}", prec = dec.min(4))))
                .collect();
            let c = bi.overlays.get(bi.overlays.len() / 2).map_or(text(), |o| opaque(&o.1));
            p.text(
                pos2(price_rect.left() + 6.0, ly),
                Align2::LEFT_TOP,
                format!("{}  {}", bi.spec.label(), vals.join(" ")),
                mono(11.0),
                c,
            );
            ly += 14.0;
        }
        let hidden = self.built.iter().filter(|b| b.spec.is_pane()).count().saturating_sub(sub_rects.len());
        if hidden > 0 {
            let note =
                format!("+{hidden} more oscillator{} (max {MAX_SUB_PANES} panes)", if hidden == 1 { "" } else { "s" });
            p.text(pos2(price_rect.left() + 6.0, ly), Align2::LEFT_TOP, note, mono(11.0), muted());
            ly += 14.0;
        }
        if let (Some(sym), Some(line)) = (&self.compare, &self.compare_line) {
            // Raw price plus performance relative to the left edge of the view.
            let raw = value_at(line);
            let rel = raw.zip(self.bars.get(i)).map(|(r, b)| (r * self.compare_scale / b.close - 1.0) * 100.0);
            let text = match (raw, rel) {
                (Some(r), Some(rel)) => format!("CMP {sym}  {r:.2}  ({rel:+.2}% vs {})", self.symbol),
                _ => format!("CMP {sym}"),
            };
            p.text(pos2(price_rect.left() + 6.0, ly), Align2::LEFT_TOP, text, mono(11.0), theme::series(5));
            ly += 14.0;
        }
        if let Some(tool) = self.draw.tool {
            let msg = if self.draw.pending.is_some() {
                "click second point (Esc cancels)"
            } else {
                "click to place (Esc cancels)"
            };
            p.text(
                pos2(price_rect.left() + 6.0, ly),
                Align2::LEFT_TOP,
                format!("{}: {msg}", tool.label()),
                mono(11.0),
                amber(),
            );
        }
        for (r, bi, _) in sub_rects {
            let b = &self.built[*bi];
            let painter = ui.painter_at(*r);
            // Values start after the pane's label (drawn with the axes).
            let mut x = r.left() + 18.0 + painter.layout_no_wrap(b.spec.label(), mono(10.5), muted()).size().x;
            for (name, o) in &b.overlays {
                let Some(v) = value_at(&o.line) else { continue };
                let color = if o.kind == LineKind::Hist { text() } else { opaque(o) };
                let g = painter.layout_no_wrap(format!("{name} {}", scale::format_value(v)), mono(10.5), color);
                let w = g.size().x;
                painter.galley(pos2(x, r.top() + 3.0), g, color);
                x += w + 10.0;
            }
        }
    }

    fn drawing_shapes(&self, d: &Drawing, rect: Rect, v: &Viewport, s: &PriceScale) -> Vec<(Shape, Option<String>)> {
        let px =
            |a: &Anchor| [rect.left() as f64 + v.x_of(self.index_of_ts(a.ts)), rect.top() as f64 + s.y_of(a.price)];
        match d.tool {
            Tool::Trendline => vec![(Shape::Segment(px(&d.a), px(&d.b)), None)],
            Tool::Horizontal => vec![(Shape::HLine(px(&d.a)[1]), Some(format!("{:.2}", d.a.price)))],
            Tool::Vertical => vec![(Shape::VLine(px(&d.a)[0]), None)],
            Tool::Fib => {
                let (xa, xb) = (px(&d.a)[0], px(&d.b)[0]);
                let (x0, x1) = (xa.min(xb), xa.max(xb));
                d.fib_levels()
                    .into_iter()
                    .map(|(r, price)| {
                        let y = rect.top() as f64 + s.y_of(price);
                        (Shape::Segment([x0, y], [x1, y]), Some(format!("{r:.3} ({price:.2})")))
                    })
                    .collect()
            }
        }
    }

    fn paint_drawings(
        &mut self,
        ui: &Ui,
        rect: Rect,
        v: &Viewport,
        s: &PriceScale,
        hover: Option<Pos2>,
        resp: &egui::Response,
    ) {
        let p = ui.painter_at(rect);
        let pointer_anchor = |this: &Self, pos: Pos2| -> Anchor {
            let idx = v.index_at((pos.x - rect.left()) as f64);
            let mut price = s.price_at((pos.y - rect.top()) as f64);
            if this.magnet
                && let Some(b) = this.bars.get(idx.floor().max(0.0) as usize)
            {
                price = drawings::magnet(price, [b.open, b.high, b.low, b.close]);
            }
            Anchor { ts: this.ts_of_index(idx), price }
        };

        // Placement.
        if let (Some(tool), true, Some(pos)) = (self.draw.tool, resp.clicked(), resp.interact_pointer_pos()) {
            let a = pointer_anchor(self, pos);
            if tool.points() == 1 {
                self.drawings.push(Drawing::new(tool, a, a));
                self.draw = DrawState::default();
            } else if let Some(first) = self.draw.pending.take() {
                self.drawings.push(Drawing::new(tool, first, a));
                self.draw = DrawState::default();
            } else {
                self.draw.pending = Some(a);
            }
        } else if self.draw.tool.is_none() && resp.clicked() {
            // Selection by proximity.
            self.draw.selected = resp.interact_pointer_pos().and_then(|pos| {
                self.drawings.iter().enumerate().find_map(|(i, d)| {
                    let near = self
                        .drawing_shapes(d, rect, v, s)
                        .iter()
                        .any(|(sh, _)| drawings::distance(sh, [pos.x as f64, pos.y as f64]) < 6.0);
                    near.then_some(i)
                })
            });
        }

        let mut all: Vec<(Drawing, bool)> =
            self.drawings.iter().cloned().enumerate().map(|(i, d)| (d, self.draw.selected == Some(i))).collect();
        if let (Some(tool), Some(first), Some(h)) = (self.draw.tool, self.draw.pending, hover) {
            all.push((Drawing::new(tool, first, pointer_anchor(self, h)), true));
        }
        for (d, selected) in all {
            let color = if selected { amber() } else { theme::cyan() };
            let stroke = Stroke::new(if selected { 2.0 } else { 1.3 }, color);
            for (shape, label) in self.drawing_shapes(&d, rect, v, s) {
                match shape {
                    Shape::Segment(a, b) => {
                        p.line_segment([pos2(a[0] as f32, a[1] as f32), pos2(b[0] as f32, b[1] as f32)], stroke);
                        if let Some(l) = label {
                            p.text(
                                pos2(a[0] as f32 + 3.0, a[1] as f32 - 1.0),
                                Align2::LEFT_BOTTOM,
                                l,
                                mono(10.0),
                                color,
                            );
                        }
                        if selected && d.tool == Tool::Trendline {
                            for q in [a, b] {
                                p.circle_filled(pos2(q[0] as f32, q[1] as f32), 3.5, amber());
                            }
                        }
                    }
                    Shape::HLine(y) => {
                        p.hline(rect.x_range(), y as f32, stroke);
                        if let Some(l) = label {
                            p.text(
                                pos2(rect.right() - 4.0, y as f32 - 1.0),
                                Align2::RIGHT_BOTTOM,
                                l,
                                mono(10.0),
                                color,
                            );
                        }
                    }
                    Shape::VLine(x) => {
                        p.vline(x as f32, rect.y_range(), stroke);
                    }
                }
            }
        }
    }

    /// Congress trade markers. They pop in once the reveal sweep has passed their bar, and
    /// left to right when they arrive after it.
    fn paint_markers(
        &mut self,
        ui: &Ui,
        rect: Rect,
        v: &Viewport,
        s: &PriceScale,
        hover: Option<Pos2>,
        reveal: &Reveal,
    ) {
        if self.markers.is_empty() || self.bars.is_empty() {
            return;
        }
        let arrive = crate::anim::progress(ui.ctx(), egui::Id::new(("chart-markers", self.id)), self.markers_key, 0.6);
        let p = ui.painter_at(rect);
        let len = self.bars.len();
        let (lo, hi) = v.visible_range(len);
        // Stack multiple trades on the same bar.
        let mut stack: std::collections::HashMap<(usize, bool), u32> = std::collections::HashMap::new();
        let full_r = (v.bar_px as f32 * 0.45).clamp(4.0, 7.0);
        let mut best: Option<(f32, usize, Pos2)> = None;
        for (mi, t) in self.markers.iter().enumerate() {
            let Some(ts) = t.tx_date.and_hms_opt(12, 0, 0).map(|d| d.and_utc().timestamp()) else { continue };
            let idx = self.index_of_ts(ts).floor();
            if idx < lo as f64 || idx >= hi as f64 || idx < 0.0 || idx >= len as f64 {
                continue;
            }
            let i = idx as usize;
            let bar = self.bars[i];
            let sell = t.tx_type.is_sell();
            let n = stack.entry((i, sell)).or_insert(0);
            let x = rect.left() + v.x_center(i) as f32;
            let off = 6.0 + full_r + *n as f32 * (2.0 * full_r + 1.0);
            let swept = ((reveal.front - (i as f64 + 0.5)) / reveal.ramp as f64).clamp(0.0, 1.0) as f32;
            let lead = 0.5 * (x - rect.left()) / rect.width().max(1.0);
            let pop = crate::anim::ease_out_back(swept.min(crate::anim::span(arrive, lead, lead + 0.5)));
            if pop <= 0.01 {
                *n += 1;
                continue;
            }
            let r = full_r * pop;
            let y = if sell {
                rect.top() + s.y_of(bar.high) as f32 - off
            } else {
                rect.top() + s.y_of(bar.low) as f32 + off
            };
            *n += 1;
            let c = Pos2::new(x, y);
            let col = theme::party_color(t.party);
            p.circle_filled(c, r, col);
            if full_r >= 5.0 {
                let letter = match t.tx_type {
                    TxType::Purchase => "B",
                    TxType::Sale | TxType::SalePartial => "S",
                    _ => "E",
                };
                let fade = crate::anim::span(pop, 0.5, 1.0);
                p.text(c, Align2::CENTER_CENTER, letter, mono(full_r * 1.6), Color32::WHITE.gamma_multiply(fade));
            } else {
                p.circle_stroke(c, r, Stroke::new(1.0, Color32::WHITE));
            }
            if let Some(h) = hover {
                let d = h.distance(c);
                if d <= r + 3.0 && best.is_none_or(|b| d < b.0) {
                    best = Some((d, mi, c));
                }
            }
        }
        if let Some((_, mi, c)) = best {
            self.hover_marker = Some(mi);
            let t = &self.markers[mi];
            hover_card(ui, c, t);
        }
    }
}

impl ChartState {
    /// Bar index of an earnings date: the bar holding it, or for a report still to come, where
    /// its bar will be (counting weekdays, since the axis skips weekends and nights).
    fn earning_index(&self, date: NaiveDate) -> Option<f64> {
        let last = self.bars.last()?;
        let ts = date.and_hms_opt(20, 0, 0)?.and_utc().timestamp();
        if ts < last.ts + self.tf.seconds() {
            return Some(self.index_of_ts(ts).floor());
        }
        let last_day = chrono::DateTime::from_timestamp(last.ts, 0)?.date_naive();
        let weekdays =
            last_day.iter_days().skip(1).take_while(|d| *d <= date).filter(|d| d.weekday().number_from_monday() <= 5);
        let per_day = match self.tf {
            Timeframe::D1 => 1.0,
            Timeframe::W1 => 0.2,
            Timeframe::MN1 => 1.0 / 21.0,
            // A US session is six and a half hours.
            tf => (23_400.0 / tf.seconds() as f64).ceil(),
        };
        let ahead = (weekdays.count() as f64 * per_day).floor().max(1.0);
        Some((self.bars.len() - 1) as f64 + ahead)
    }

    /// Earnings markers: an "E" badge along the bottom of the price pane at each report,
    /// green for an EPS beat, red for a miss and purple for the next report. Hovering shows a
    /// summary; a click opens the details card, and a click elsewhere or Esc closes it. They
    /// pop in behind the reveal sweep like the Congress markers.
    fn paint_earnings(
        &mut self,
        ui: &Ui,
        rect: Rect,
        v: &Viewport,
        hover: Option<Pos2>,
        reveal: &Reveal,
        plot_resp: &egui::Response,
    ) {
        if self.earnings.is_empty() || self.bars.is_empty() {
            return;
        }
        let arrive =
            crate::anim::progress(ui.ctx(), egui::Id::new(("chart-earnings", self.id)), self.earnings_key, 0.6);
        let len = self.bars.len();
        let size = 15.0;
        let y = rect.bottom() - size * 0.5 - 3.0;
        let reach = size * 0.5 + 3.0;
        let mut badges: Vec<(usize, Pos2, f32)> = Vec::new();
        for (ei, e) in self.earnings.iter().enumerate() {
            let Some(idx) = self.earning_index(e.date) else { continue };
            if idx < 0.0 || (idx >= len as f64 && !e.upcoming) {
                continue;
            }
            let x = rect.left() + v.x_of(idx + 0.5) as f32;
            if x < rect.left() - size || x > rect.right() + size {
                continue;
            }
            let swept = ((reveal.front - (idx + 0.5)) / reveal.ramp as f64).clamp(0.0, 1.0) as f32;
            let lead = 0.5 * (x - rect.left()) / rect.width().max(1.0);
            let pop = crate::anim::ease_out_back(swept.min(crate::anim::span(arrive, lead, lead + 0.5)));
            if pop > 0.01 {
                badges.push((ei, Pos2::new(x, y), pop));
            }
        }
        let nearest = |at: Pos2| {
            badges
                .iter()
                .map(|(ei, c, _)| (at.distance(*c), *ei, *c))
                .filter(|(d, _, _)| *d <= reach)
                .min_by(|a, b| a.0.total_cmp(&b.0))
        };

        // A click on a badge toggles its card; a click anywhere else on the plot closes it.
        if plot_resp.clicked() && self.draw.tool.is_none() {
            let hit = plot_resp.interact_pointer_pos().and_then(nearest).map(|(_, ei, _)| self.earnings[ei].date);
            self.open_earning = hit.filter(|d| self.open_earning != Some(*d));
        }

        let p = ui.painter_at(rect);
        for &(ei, c, pop) in &badges {
            let e = &self.earnings[ei];
            let col = earning_color(e);
            let open = self.open_earning == Some(e.date);
            let badge = Rect::from_center_size(c, vec2(size, size) * pop);
            let (fill, letter) = if open {
                (col, Color32::BLACK)
            } else {
                (theme::card().lerp_to_gamma(col, if e.outcome() == Outcome::Unknown { 0.0 } else { 0.22 }), col)
            };
            p.rect_filled(badge, CornerRadius::same(3), fill);
            p.rect_stroke(badge, CornerRadius::same(3), Stroke::new(1.0, col), StrokeKind::Inside);
            if pop >= 0.6 {
                p.text(c, Align2::CENTER_CENTER, "E", mono(11.0 * pop), letter);
            }
        }

        let hovered = hover.and_then(nearest);
        if let Some((_, ei, c)) = hovered {
            ui.ctx().set_cursor_icon(CursorIcon::PointingHand);
            if self.open_earning != Some(self.earnings[ei].date) {
                earnings_card(ui, c, &self.earnings[ei]);
            }
        }
        let open = self.open_earning.and_then(|d| badges.iter().find(|(ei, _, _)| self.earnings[*ei].date == d));
        if let Some(&(ei, c, _)) = open {
            let id = egui::Id::new(("earnings-details", self.id));
            if earnings_details(ui.ctx(), id, c - vec2(0.0, size * 0.5 + 6.0), &self.earnings[ei]) {
                self.open_earning = None;
            }
        }
    }
}

const UPCOMING: Color32 = Color32::from_rgb(170, 125, 255);

/// Badge colour: green beat, red miss, purple next report, grey in line, cyan when there's
/// no estimate to judge by.
fn earning_color(e: &Earning) -> Color32 {
    match e.outcome() {
        Outcome::Beat => theme::up(),
        Outcome::Miss => theme::down(),
        Outcome::Upcoming => UPCOMING,
        Outcome::InLine => text(),
        Outcome::Unknown => theme::palette().cyan,
    }
}

fn day(d: NaiveDate) -> String {
    d.format("%b %-d, %Y").to_string()
}

fn eps(v: f64) -> String {
    format!("{v:.2}")
}

/// 94.93B, 1.02T, 512.40M.
fn money(v: f64) -> String {
    let a = v.abs();
    let (div, unit) = if a >= 1e12 {
        (1e12, "T")
    } else if a >= 1e9 {
        (1e9, "B")
    } else if a >= 1e6 {
        (1e6, "M")
    } else if a >= 1e3 {
        (1e3, "K")
    } else {
        (1.0, "")
    };
    format!("{:.2}{unit}", v / div)
}

/// "+0.08 (+4.52%)" for a surprise, with its colour.
fn surprise_text(s: tr_data::earnings::Surprise, fmt: fn(f64) -> String) -> (String, Color32) {
    let sign = if s.diff >= 0.0 { "+" } else { "-" };
    let mut out = format!("{sign}{}", fmt(s.diff.abs()));
    if let Some(pct) = s.pct {
        out.push_str(&format!(" ({pct:+.2}%)"));
    }
    let col = if s.diff.abs() < 0.005 {
        text()
    } else if s.diff > 0.0 {
        theme::up()
    } else {
        theme::down()
    };
    (out, col)
}

fn outcome_label(e: &Earning) -> &'static str {
    match e.outcome() {
        Outcome::Beat => "Beat",
        Outcome::Miss => "Missed",
        Outcome::InLine => "In line",
        Outcome::Upcoming => "Upcoming",
        Outcome::Unknown => "No estimate",
    }
}

/// The details card for one report, above its badge: dates, EPS (GAAP, reported, estimate,
/// surprise) and revenue (reported, estimate, surprise). Returns true when its close button
/// was pressed.
fn earnings_details(ctx: &egui::Context, id: egui::Id, anchor: Pos2, e: &Earning) -> bool {
    let shown = crate::anim::ease_out(crate::anim::appear(ctx, id.with(e.date), 0.16));
    let col = earning_color(e);
    let mut close = false;
    egui::Area::new(id)
        .order(egui::Order::Foreground)
        .pivot(Align2::CENTER_BOTTOM)
        .fixed_pos(anchor + vec2(0.0, 6.0 * (1.0 - shown)))
        .constrain(true)
        .show(ctx, |ui| {
            ui.multiply_opacity(shown);
            egui::Frame::new()
                .fill(theme::card())
                .stroke(Stroke::new(1.0, theme::border()))
                .corner_radius(CornerRadius::same(theme::CARD_RADIUS))
                .shadow(egui::Shadow { offset: [0, 8], blur: 24, spread: 0, color: Color32::from_black_alpha(80) })
                .inner_margin(12.0)
                .show(ui, |ui| {
                    // Fixed, or the right-aligned header row stretches the card across the screen.
                    ui.set_width(290.0);
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Earnings & Revenue").font(ui_font(13.5)).color(text()).strong());
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let x = ui.add(
                                egui::Button::new(RichText::new("×").font(ui_font(15.0)).color(muted())).frame(false),
                            );
                            close = x.on_hover_text("Close (Esc)").clicked();
                            let pill = RichText::new(format!(" {} ", outcome_label(e))).font(ui_font(11.0));
                            let pill = if e.outcome() == Outcome::Miss || e.upcoming {
                                pill.color(Color32::WHITE).background_color(col)
                            } else {
                                pill.color(col).background_color(theme::card().lerp_to_gamma(col, 0.2))
                            };
                            ui.label(pill);
                        });
                    });
                    ui.add_space(6.0);
                    let label = |ui: &mut Ui, s: &str| ui.label(RichText::new(s).font(mono(11.5)).color(muted()));
                    let value =
                        |ui: &mut Ui, s: String, c: Color32| ui.label(RichText::new(s).font(mono(11.5)).color(c));
                    let section = |ui: &mut Ui, s: &str| {
                        ui.add_space(8.0);
                        ui.label(RichText::new(s).font(ui_font(10.5)).color(muted()).strong());
                        ui.add_space(2.0);
                    };
                    // One grid per section (a grid can't hold spacing); a shared first-column
                    // width keeps their values lined up.
                    let grid = |ui: &mut Ui, name: &str, rows: &mut dyn FnMut(&mut Ui)| {
                        egui::Grid::new(id.with(name))
                            .num_columns(2)
                            .min_col_width(110.0)
                            .spacing(vec2(16.0, 4.0))
                            .show(ui, |ui| rows(ui));
                    };
                    let dash = || "—".to_string();
                    grid(ui, "dates", &mut |ui| {
                        label(ui, "Date");
                        let when = if e.upcoming { format!("{} (expected)", day(e.date)) } else { day(e.date) };
                        value(ui, when, text());
                        ui.end_row();
                        label(ui, "Period ending");
                        let period = match e.period {
                            Some(p) if e.upcoming => format!("{} (est.)", day(p)),
                            Some(p) => day(p),
                            None => dash(),
                        };
                        value(ui, period, text());
                        ui.end_row();
                    });
                    section(ui, "EARNINGS PER SHARE");
                    grid(ui, "eps", &mut |ui| {
                        if !e.upcoming {
                            label(ui, "GAAP (diluted)");
                            value(ui, e.eps_gaap.map(eps).unwrap_or_else(dash), text());
                            ui.end_row();
                            label(ui, "Reported");
                            value(ui, e.eps_actual.map(eps).unwrap_or_else(dash), text());
                            ui.end_row();
                        }
                        label(ui, "Estimate");
                        value(ui, e.eps_estimate.map(eps).unwrap_or_else(dash), text());
                        ui.end_row();
                        if !e.upcoming {
                            label(ui, "Surprise");
                            let (s, c) = e.eps_surprise().map(|s| surprise_text(s, eps)).unwrap_or((dash(), muted()));
                            value(ui, s, c);
                            ui.end_row();
                        }
                    });
                    section(ui, "REVENUE");
                    grid(ui, "revenue", &mut |ui| {
                        if !e.upcoming {
                            label(ui, "Reported");
                            value(ui, e.revenue.map(money).unwrap_or_else(dash), text());
                            ui.end_row();
                        }
                        label(ui, "Estimate");
                        value(ui, e.revenue_estimate.map(money).unwrap_or_else(dash), text());
                        ui.end_row();
                        if !e.upcoming {
                            label(ui, "Surprise");
                            let (s, c) =
                                e.revenue_surprise().map(|s| surprise_text(s, money)).unwrap_or((dash(), muted()));
                            value(ui, s, c);
                            ui.end_row();
                        }
                    });
                    ui.add_space(8.0);
                    let source = match e.form.as_str() {
                        _ if e.upcoming => "Estimates: Yahoo Finance".to_string(),
                        "" => "Estimates and EPS: Yahoo Finance".to_string(),
                        f => format!("Date: SEC {f} · Figures: SEC filings · Estimates: Yahoo Finance"),
                    };
                    ui.label(RichText::new(source).font(ui_font(10.0)).color(muted()));
                });
        });
    close
}

/// Hover card for an earnings marker: the result in one line and a hint to click.
fn earnings_card(ui: &Ui, at: Pos2, e: &Earning) {
    let col = earning_color(e);
    let head = if e.upcoming { "Next earnings" } else { "Earnings" };
    let mut lines = vec![(format!("{head}  {}", day(e.date)), col)];
    let summary = match (e.outcome(), e.eps_actual, e.eps_estimate) {
        (Outcome::Upcoming, _, Some(est)) => Some(format!("EPS estimate {}", eps(est))),
        (Outcome::Beat | Outcome::Miss | Outcome::InLine, Some(a), Some(est)) => {
            let pct = e.eps_surprise().and_then(|s| s.pct).map(|p| format!(" ({p:+.1}%)")).unwrap_or_default();
            Some(format!("{}: EPS {} vs {} est.{pct}", outcome_label(e), eps(a), eps(est)))
        }
        _ => e.period.map(|p| format!("Period ended {}", day(p))),
    };
    if let Some(s) = summary {
        lines.push((s, text()));
    }
    lines.push(("Click for details".to_string(), muted()));
    let w = lines.iter().map(|(l, _)| l.chars().count()).max().unwrap_or(10) as f32 * 7.2 + 14.0;
    let h = lines.len() as f32 * 14.0 + 8.0;
    let screen = ui.ctx().content_rect();
    let mut min = at + vec2(12.0, -h - 12.0);
    if min.x + w > screen.right() {
        min.x = at.x - w - 12.0;
    }
    if min.y < screen.top() {
        min.y = at.y + 12.0;
    }
    let card = Rect::from_min_size(min, vec2(w, h));
    let layer = egui::LayerId::new(egui::Order::Tooltip, egui::Id::new("earnings-card"));
    let p = ui.ctx().layer_painter(layer);
    p.rect_filled(card, 0.0, Color32::from_rgb(16, 16, 16));
    p.rect_stroke(card, 0.0, Stroke::new(1.0, amber_dim()), StrokeKind::Inside);
    for (k, (l, c)) in lines.iter().enumerate() {
        p.text(card.left_top() + vec2(7.0, 4.0 + k as f32 * 14.0), Align2::LEFT_TOP, l, mono(11.5), *c);
    }
}

fn opaque(o: &Overlay) -> Color32 {
    let [r, g, b, _] = o.color.map(|c| (c.clamp(0.0, 1.0) * 255.0) as u8);
    Color32::from_rgb(r, g, b)
}

fn dashed_h(p: &egui::Painter, x: egui::Rangef, y: f32, s: Stroke) {
    let mut a = x.min;
    while a < x.max {
        p.hline(a..=(a + 4.0).min(x.max), y, s);
        a += 7.0;
    }
}

fn dashed_v(p: &egui::Painter, x: f32, y: egui::Rangef, s: Stroke) {
    let mut a = y.min;
    while a < y.max {
        p.vline(x, a..=(a + 4.0).min(y.max), s);
        a += 7.0;
    }
}

/// Hover card for a congressional trade marker, including the disclosure lag.
fn hover_card(ui: &Ui, at: Pos2, t: &Trade) {
    let lines = [
        (
            format!("{} ({}-{})", t.politician, t.party.code(), t.state.as_deref().unwrap_or("??")),
            theme::party_color(t.party),
        ),
        (format!("{} · {}", t.chamber.as_str(), t.owner.as_deref().unwrap_or("Self")), muted()),
        (
            format!(
                "{} {}  {}",
                match t.tx_type {
                    TxType::Purchase => "BUY",
                    TxType::Sale => "SELL",
                    TxType::SalePartial => "SELL (partial)",
                    TxType::Exchange => "EXCHANGE",
                    TxType::Other => "OTHER",
                },
                t.ticker.as_deref().unwrap_or("-"),
                t.amount.map(|a| a.to_string()).unwrap_or_else(|| "amount n/a".into())
            ),
            if t.tx_type.is_buy() { theme::up() } else { theme::down() },
        ),
        (format!("Traded  {}", t.tx_date), text()),
        (format!("Filed   {}", t.filed_date), text()),
        (
            format!("Disclosure lag {} days", t.disclosure_lag_days()),
            if t.disclosure_lag_days() > 45 { theme::down() } else { amber() },
        ),
    ];
    let w = lines.iter().map(|(l, _)| l.len()).max().unwrap_or(10) as f32 * 7.2 + 14.0;
    let h = lines.len() as f32 * 14.0 + 8.0;
    let screen = ui.ctx().content_rect();
    let mut min = at + vec2(12.0, 12.0);
    if min.x + w > screen.right() {
        min.x = at.x - w - 12.0;
    }
    if min.y + h > screen.bottom() {
        min.y = at.y - h - 12.0;
    }
    let card = Rect::from_min_size(min, vec2(w, h));
    let layer = egui::LayerId::new(egui::Order::Tooltip, egui::Id::new("congress-card"));
    let p = ui.ctx().layer_painter(layer);
    p.rect_filled(card, 0.0, Color32::from_rgb(16, 16, 16));
    p.rect_stroke(card, 0.0, Stroke::new(1.0, amber_dim()), StrokeKind::Inside);
    for (k, (l, c)) in lines.iter().enumerate() {
        p.text(card.left_top() + vec2(7.0, 4.0 + k as f32 * 14.0), Align2::LEFT_TOP, l, mono(11.5), *c);
    }
}

/// Whether the chart has reached the start of its history after a response. A full load decides
/// it, since it may come from another source than the bars before it. A page decides it too. Live
/// updates and refreshes keep what the chart knew.
fn end_of_history(was: bool, exhausted: bool, prepended: usize, full: bool) -> bool {
    if full { exhausted } else { exhausted || was && prepended == 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_full_load_from_a_new_source_can_page_again() {
        assert!(!end_of_history(true, false, 0, true), "a reload from a source with more history");
        assert!(end_of_history(false, true, 0, true));
        assert!(end_of_history(true, false, 0, false), "a live update keeps the end");
        assert!(!end_of_history(true, false, 5, false), "a page that added bars");
        assert!(end_of_history(false, true, 0, false), "an empty page");
    }

    #[test]
    fn glide_pans_linearly_and_zooms_evenly() {
        let from = Viewport { right: 100.0, bar_px: 2.0, width_px: 800.0 };
        let to = Viewport { right: 200.0, bar_px: 8.0, width_px: 800.0 };
        let g = Glide { from, to, start: None };
        assert_eq!(g.at(0.0), from);
        let end = g.at(1.0);
        assert_eq!(end.right, to.right);
        assert!((end.bar_px - to.bar_px).abs() < 1e-9);
        let mid = g.at(0.5);
        assert_eq!(mid.right, 150.0);
        assert!((mid.bar_px - 4.0).abs() < 1e-9, "geometric midpoint");
    }

    #[test]
    fn view_changes_glide_from_what_is_shown() {
        let mut c = ChartState::new(1, "AAPL", Timeframe::D1);
        c.bars = Arc::new(tr_core::sample::generate("AAPL", Timeframe::D1, 500, 1_790_000_000));
        let shown = Viewport::fit_last(500, 100, 800.0);
        c.viewport = Some(shown);
        c.pan_bars(-20.0);
        assert_eq!(c.viewport, Some(shown), "stays where it is shown until the glide runs");
        assert_eq!(c.glide.map(|g| g.to.right), Some(shown.right - 20.0));
        // A second press adds to the destination and keeps gliding from the shown view.
        c.pan_bars(-20.0);
        let g = c.glide.expect("gliding");
        assert_eq!((g.from, g.to.right), (shown, shown.right - 40.0));
        c.reset_view();
        assert_eq!(c.glide.map(|g| g.to), Some(Viewport::fit_last(500, DEFAULT_VISIBLE, 800.0)));
    }
}
