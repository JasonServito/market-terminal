//! wgpu renderer for chart geometry.
//!
//! Series data (LOD pyramids) is uploaded once per data version into storage buffers and
//! cached by key; per frame only a few hundred bytes of uniforms change, so pan and zoom
//! cost a uniform write plus O(plot width) instances.

use std::collections::HashMap;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use egui_wgpu::wgpu;
use wgpu::util::DeviceExt;

use crate::lod::{BarPyramid, GpuBar, LinePyramid};

const SLOT: u64 = 256;
const SLOTS: u64 = 4096;
/// Evict GPU buffers not referenced for this many frames.
const EVICT_AFTER: u64 = 600;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct Uniforms {
    pub rect: [f32; 4],
    pub view: [f32; 4],
    pub up: [f32; 4],
    pub down: [f32; 4],
    pub color: [f32; 4],
    pub params: [f32; 4],
    pub extra: [f32; 4],
    pub anim: [f32; 4],
}

/// Load-in sweep for a pane. Bars right of `front` (a fractional bar index) are hidden
/// and the `ramp` bars behind it grow in from their open; lines are drawn up to
/// `line_front`. `opacity` fades everything in the pane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reveal {
    pub front: f64,
    pub ramp: f32,
    pub line_front: f64,
    pub opacity: f32,
}

impl Reveal {
    pub const DONE: Self = Self { front: f64::INFINITY, ramp: 1.0, line_front: f64::INFINITY, opacity: 1.0 };

    /// The sweep at progress `t` (0..1) across `visible_bars` bars starting at `left`.
    pub fn sweep(t: f32, left: f64, visible_bars: f64) -> Self {
        if t >= 1.0 {
            return Self::DONE;
        }
        let ramp = (visible_bars * 0.12).max(2.0);
        let front = left + t as f64 * (visible_bars + ramp);
        Self { front, ramp: ramp as f32, line_front: front - ramp * 0.5, opacity: 1.0 }
    }
}

/// Pane-level transform shared by every draw in a pane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PaneTransform {
    pub width_px: f32,
    pub height_px: f32,
    /// Fractional bar index at the left edge.
    pub left: f64,
    pub visible_bars: f64,
    /// Transformed (linear or ln) y bounds.
    pub y_lo: f64,
    pub y_hi: f64,
    pub log: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Style {
    pub up: [f32; 4],
    pub down: [f32; 4],
    pub color: [f32; 4],
    pub body_frac: f32,
    pub wick_px: f32,
    pub thickness_px: f32,
    pub ohlc_bars: bool,
    pub vol_frac: f32,
    pub vol_max: f32,
    pub hist_width_bars: f32,
    pub baseline: f32,
    /// Physical pixels per logical point (2.0 on Retina). Pixel-sized strokes are given
    /// in points and scaled by this; pane sizes in [`PaneTransform`] are physical.
    pub pixels_per_point: f32,
    pub reveal: Reveal,
}

impl Default for Style {
    fn default() -> Self {
        Self {
            up: [0.15, 0.82, 0.49, 1.0],
            down: [1.0, 0.27, 0.27, 1.0],
            color: [1.0, 0.69, 0.0, 1.0],
            body_frac: 0.7,
            wick_px: 1.0,
            thickness_px: 1.5,
            ohlc_bars: false,
            vol_frac: 0.2,
            vol_max: 1.0,
            hist_width_bars: 0.7,
            baseline: 0.0,
            pixels_per_point: 1.0,
            reveal: Reveal::DONE,
        }
    }
}

impl Uniforms {
    pub fn new(t: &PaneTransform, s: &Style) -> Self {
        let left_hi = t.left.floor();
        // Relative to `left_hi` for f32 precision on long series.
        let rel = |x: f64| if x.is_finite() { (x - left_hi) as f32 } else { 1e9 };
        let r = s.reveal;
        Self {
            rect: [t.width_px, t.height_px, left_hi as f32, (t.left - left_hi) as f32],
            view: [t.visible_bars as f32, t.y_lo as f32, t.y_hi as f32, if t.log { 1.0 } else { 0.0 }],
            up: s.up,
            down: s.down,
            color: s.color,
            params: [
                s.body_frac,
                s.wick_px * s.pixels_per_point,
                s.thickness_px * s.pixels_per_point,
                if s.ohlc_bars { 1.0 } else { 0.0 },
            ],
            extra: [s.vol_frac, s.vol_max, s.hist_width_bars, s.baseline],
            anim: [rel(r.front), r.ramp.max(1e-3), rel(r.line_front), r.opacity],
        }
    }
}

/// Shared handle to an uploaded-on-demand bar pyramid.
#[derive(Clone)]
pub struct SeriesRef {
    pub key: u64,
    pub version: u64,
    pub pyramid: Arc<BarPyramid>,
}

#[derive(Clone)]
pub struct LineRef {
    pub key: u64,
    pub version: u64,
    pub pyramid: Arc<LinePyramid>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineKind {
    Line,
    Area,
    Hist,
    /// A square marker at each finite point.
    Dots,
    /// A larger square marker at each finite point, for signals.
    Marks,
    /// A flat dash across each finite point's bar, with no joins between bars.
    Steps,
    /// The area between the line and the baseline, filled (oscillator waves).
    Fill,
}

#[derive(Clone)]
pub enum DrawCmd {
    Candles { data: SeriesRef, range: (u32, u32), uniforms: Uniforms },
    Volume { data: SeriesRef, range: (u32, u32), uniforms: Uniforms },
    Line { kind: LineKind, data: LineRef, range: (u32, u32), uniforms: Uniforms },
}

/// Everything one pane draws in one frame.
#[derive(Clone)]
pub struct Frame {
    pub pane: u64,
    pub frame_no: u64,
    pub cmds: Vec<DrawCmd>,
}

struct GpuBuffer {
    version: u64,
    bind_group: wgpu::BindGroup,
    last_used: u64,
    _buffer: wgpu::Buffer,
}

#[derive(Clone, Copy)]
enum Pipe {
    Candle,
    Volume,
    Line,
    Area,
    Hist,
    Dots,
    Steps,
    Fill,
}

struct Resolved {
    pipe: Pipe,
    key: (bool, u64),
    slot: u32,
    range: (u32, u32),
}

pub struct ChartRenderer {
    candle: wgpu::RenderPipeline,
    volume: wgpu::RenderPipeline,
    line: wgpu::RenderPipeline,
    area: wgpu::RenderPipeline,
    hist: wgpu::RenderPipeline,
    dots: wgpu::RenderPipeline,
    steps: wgpu::RenderPipeline,
    fill: wgpu::RenderPipeline,
    uniform_buf: wgpu::Buffer,
    uniform_bg: wgpu::BindGroup,
    storage_bgl: wgpu::BindGroupLayout,
    /// (is_line, key) -> uploaded buffer
    buffers: HashMap<(bool, u64), GpuBuffer>,
    panes: HashMap<u64, Vec<Resolved>>,
    frame_no: u64,
    next_slot: u64,
    max_binding: u64,
}

fn shader_sources() -> (String, String) {
    let bars = include_str!("shader.wgsl");
    let prelude = bars.split("// ---- bars").next().unwrap_or(bars);
    (bars.to_string(), format!("{prelude}\n{}", include_str!("shader_line.wgsl")))
}

impl ChartRenderer {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, msaa_samples: u32) -> Self {
        let (bar_src, line_src) = shader_sources();
        let bar_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("chart-bars"),
            source: wgpu::ShaderSource::Wgsl(bar_src.into()),
        });
        let line_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("chart-lines"),
            source: wgpu::ShaderSource::Wgsl(line_src.into()),
        });
        let uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("chart-uniforms"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<Uniforms>() as u64),
                },
                count: None,
            }],
        });
        let storage_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("chart-storage"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("chart-layout"),
            bind_group_layouts: &[Some(&uniform_bgl), Some(&storage_bgl)],
            immediate_size: 0,
        });
        let make = |module: &wgpu::ShaderModule, entry: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState { count: msaa_samples.max(1), ..Default::default() },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chart-uniform-ring"),
            size: SLOT * SLOTS,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chart-uniforms"),
            layout: &uniform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buf,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<Uniforms>() as u64),
                }),
            }],
        });
        Self {
            candle: make(&bar_mod, "vs_candle"),
            volume: make(&bar_mod, "vs_volume"),
            line: make(&line_mod, "vs_line"),
            area: make(&line_mod, "vs_area"),
            hist: make(&line_mod, "vs_hist"),
            dots: make(&line_mod, "vs_dots"),
            steps: make(&line_mod, "vs_steps"),
            fill: make(&line_mod, "vs_fill"),
            uniform_buf,
            uniform_bg,
            storage_bgl,
            buffers: HashMap::new(),
            panes: HashMap::new(),
            frame_no: u64::MAX,
            next_slot: 0,
            max_binding: device.limits().max_storage_buffer_binding_size,
        }
    }

    fn ensure(
        &mut self,
        device: &wgpu::Device,
        is_line: bool,
        key: u64,
        version: u64,
        bytes: impl FnOnce() -> Vec<u8>,
    ) {
        let frame = self.frame_no;
        if let Some(b) = self.buffers.get_mut(&(is_line, key)) {
            b.last_used = frame;
            if b.version == version {
                return;
            }
        }
        let mut contents = bytes();
        if contents.is_empty() {
            contents = vec![0; 32];
        }
        contents.truncate(self.max_binding as usize);
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(if is_line { "chart-line" } else { "chart-bars" }),
            contents: &contents,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.storage_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
        });
        self.buffers.insert((is_line, key), GpuBuffer { version, bind_group, last_used: frame, _buffer: buffer });
    }

    /// Upload any changed data and this frame's uniforms for one pane.
    pub fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, frame: &Frame) {
        if frame.frame_no != self.frame_no {
            self.frame_no = frame.frame_no;
            self.next_slot = 0;
            let now = self.frame_no;
            self.buffers.retain(|_, b| now.saturating_sub(b.last_used) < EVICT_AFTER);
        }
        let mut resolved = Vec::with_capacity(frame.cmds.len());
        for cmd in &frame.cmds {
            if self.next_slot >= SLOTS {
                break;
            }
            let slot = self.next_slot as u32;
            self.next_slot += 1;
            let (pipe, key, range, uniforms) = match cmd {
                DrawCmd::Candles { data, range, uniforms } | DrawCmd::Volume { data, range, uniforms } => {
                    self.ensure(device, false, data.key, data.version, || {
                        bytemuck::cast_slice::<GpuBar, u8>(&data.pyramid.data).to_vec()
                    });
                    let pipe = if matches!(cmd, DrawCmd::Candles { .. }) { Pipe::Candle } else { Pipe::Volume };
                    (pipe, (false, data.key), *range, uniforms)
                }
                DrawCmd::Line { kind, data, range, uniforms } => {
                    self.ensure(device, true, data.key, data.version, || {
                        bytemuck::cast_slice::<[f32; 2], u8>(&data.pyramid.data).to_vec()
                    });
                    let pipe = match kind {
                        LineKind::Line => Pipe::Line,
                        LineKind::Area => Pipe::Area,
                        LineKind::Hist => Pipe::Hist,
                        LineKind::Dots | LineKind::Marks => Pipe::Dots,
                        LineKind::Steps => Pipe::Steps,
                        LineKind::Fill => Pipe::Fill,
                    };
                    (pipe, (true, data.key), *range, uniforms)
                }
            };
            queue.write_buffer(&self.uniform_buf, slot as u64 * SLOT, bytemuck::bytes_of(uniforms));
            resolved.push(Resolved { pipe, key, slot, range });
        }
        self.panes.insert(frame.pane, resolved);
    }

    /// Record a pane's draws. The caller sets viewport/scissor to the pane rect.
    pub fn paint(&self, pane: u64, pass: &mut wgpu::RenderPass<'_>) {
        let Some(draws) = self.panes.get(&pane) else { return };
        for d in draws {
            let Some(buf) = self.buffers.get(&d.key) else { continue };
            if d.range.1 <= d.range.0 {
                continue;
            }
            let (pipeline, verts) = match d.pipe {
                Pipe::Candle => (&self.candle, 18),
                Pipe::Volume => (&self.volume, 6),
                Pipe::Line => (&self.line, 6),
                Pipe::Area => (&self.area, 6),
                Pipe::Hist => (&self.hist, 6),
                Pipe::Dots => (&self.dots, 6),
                Pipe::Steps => (&self.steps, 6),
                Pipe::Fill => (&self.fill, 6),
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.uniform_bg, &[d.slot * SLOT as u32]);
            pass.set_bind_group(1, &buf.bind_group, &[]);
            pass.draw(0..verts, d.range.0..d.range.1);
        }
    }

    pub fn cached_buffers(&self) -> usize {
        self.buffers.len()
    }
}

/// egui paint-callback adapter.
pub struct ChartCallback(pub Arc<Frame>);

impl egui_wgpu::CallbackTrait for ChartCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(r) = resources.get_mut::<ChartRenderer>() {
            r.prepare(device, queue, &self.0);
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        if let Some(r) = resources.get::<ChartRenderer>() {
            r.paint(self.0.pane, pass);
        }
    }
}
