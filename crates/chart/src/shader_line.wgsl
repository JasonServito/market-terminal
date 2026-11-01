// Line, area and histogram geometry from a point buffer. Appended to the common prelude
// (uniforms and helpers) from shader.wgsl's first section at build time.

@group(1) @binding(0) var<storage, read> pts: array<vec2<f32>>;

fn finite(v: f32) -> bool {
    return v == v && abs(v) < 1e30;
}

fn to_px(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>((x_ndc(p.x) + 1.0) * 0.5 * u.rect.x, (y_ndc(p.y) + 1.0) * 0.5 * u.rect.y);
}

fn px_to_ndc(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(p.x / u.rect.x * 2.0 - 1.0, p.y / u.rect.y * 2.0 - 1.0);
}

// Segment end `b` cut back to the line front while the pane is revealing.
fn traced(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let fa = a.x - u.rect.z;
    let fb = b.x - u.rect.z;
    if (fb <= u.anim.z) {
        return b;
    }
    return mix(a, b, clamp((u.anim.z - fa) / max(fb - fa, 1e-6), 0.0, 1.0));
}

// Thick segment between points ii and ii+1, expanded in pixel space.
@vertex
fn vs_line(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    let a = pts[ii];
    if (!(finite(a.y) && finite(pts[ii + 1u].y)) || a.x - u.rect.z >= u.anim.z) {
        return degenerate();
    }
    let b = traced(a, pts[ii + 1u]);
    let pa = to_px(a);
    let pb = to_px(b);
    var d = pb - pa;
    let len = length(d);
    d = select(vec2<f32>(1.0, 0.0), d / len, len > 1e-6);
    let n = vec2<f32>(-d.y, d.x) * (u.params.z * 0.5);
    // Extend slightly along the direction so joints between segments have no gaps.
    let e = d * (u.params.z * 0.5);
    let c = corner(vi);
    let along = mix(pa - e, pb + e, c.x);
    let p = along + n * (c.y * 2.0 - 1.0);
    var out: VsOut;
    out.pos = vec4<f32>(px_to_ndc(p), 0.0, 1.0);
    out.color = shade(u.color);
    return out;
}

// Filled area from segment ii..ii+1 down to the pane bottom.
@vertex
fn vs_area(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    let a = pts[ii];
    if (!(finite(a.y) && finite(pts[ii + 1u].y)) || a.x - u.rect.z >= u.anim.z) {
        return degenerate();
    }
    let b = traced(a, pts[ii + 1u]);
    let c = corner(vi);
    let x = mix(x_ndc(a.x), x_ndc(b.x), c.x);
    let top = mix(y_ndc(a.y), y_ndc(b.y), c.x);
    let y = mix(-1.0, top, c.y);
    // Fade towards the bottom.
    var out: VsOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.color = shade(vec4<f32>(u.color.rgb, u.color.a * mix(0.15, 1.0, c.y)));
    return out;
}

// Filled area from segment ii..ii+1 to the baseline (extra.w), e.g. oscillator waves. A
// segment that crosses the baseline folds into two triangles that meet at the crossing.
@vertex
fn vs_fill(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    let a = pts[ii];
    if (!(finite(a.y) && finite(pts[ii + 1u].y)) || a.x - u.rect.z >= u.anim.z) {
        return degenerate();
    }
    let b = traced(a, pts[ii + 1u]);
    let c = corner(vi);
    let x = mix(x_ndc(a.x), x_ndc(b.x), c.x);
    let top = mix(y_ndc(a.y), y_ndc(b.y), c.x);
    var out: VsOut;
    out.pos = vec4<f32>(x, mix(y_ndc(u.extra.w), top, c.y), 0.0, 1.0);
    out.color = shade(u.color);
    return out;
}

// Histogram column for point ii from the baseline (e.g. MACD histogram).
@vertex
fn vs_hist(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    let a = pts[ii];
    let g = grow(a.x);
    if (!finite(a.y) || g <= 0.0) {
        return degenerate();
    }
    let color = select(u.down, u.up, a.y >= u.extra.w);
    let half = max(u.extra.z * 0.5 / u.view.x * 2.0, px_to_ndc_x(0.5));
    let cx = x_ndc(a.x);
    return rect(vi, cx - half, cx + half, y_ndc(u.extra.w), y_ndc(mix(u.extra.w, a.y, g)), color);
}

// Flat dash at point ii's level across the bars of its group (extra.z bars wide, one when
// not zoomed out). The point is its group's last bar, so the dash runs from the group's
// start to that bar; a short final group stays short. Consecutive levels never join, so a
// session's pivot starts afresh instead of stepping up from the last one.
@vertex
fn vs_steps(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    let a = pts[ii];
    let x1 = a.x + 0.5;
    let x0 = floor(a.x / u.extra.z) * u.extra.z;
    if (!finite(a.y) || x0 - u.rect.z >= u.anim.z) {
        return degenerate();
    }
    let hy = px_to_ndc_y(u.params.z * 0.5);
    let cy = y_ndc(a.y);
    return rect(vi, x_ndc(x0), x_ndc(min(x1, u.rect.z + u.anim.z)), cy - hy, cy + hy, u.color);
}

// Square marker centred on point ii (e.g. parabolic SAR), sized from the line thickness.
// Markers grow in behind the reveal front.
@vertex
fn vs_dots(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    let a = pts[ii];
    let g = grow(a.x);
    if (!finite(a.y) || g <= 0.0) {
        return degenerate();
    }
    let hx = px_to_ndc_x(u.params.z * 1.25 * g);
    let hy = px_to_ndc_y(u.params.z * 1.25 * g);
    let cx = x_ndc(a.x);
    let cy = y_ndc(a.y);
    return rect(vi, cx - hx, cx + hx, cy - hy, cy + hy, u.color);
}
