// ABOUTME: Fullscreen-quad shader for the photo viewport.
// ABOUTME: Handles view transform, compare split, loupe magnifier, overlays, dither.

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) tex_coords: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

struct Settings {
    view_matrix: mat3x3<f32>,
    grayscale: f32,
    overlay_opacity: f32,
    grid_size: f32,
    overlay_mode: u32,
    compare_divider: f32,
    compare_active: u32,
    loupe_active: u32,
    loupe_zoom: f32,
    loupe_center_uv: vec2<f32>,
    loupe_center_screen: vec2<f32>,
    loupe_radius: f32,
    dither: u32,
    split_tone_active: u32,
    split_tone_amount: f32,
    clipping_warning: u32,
    posterize_active: u32,
    posterize_levels: u32,
    annotation_active: u32,
    grid_active: u32,
    grid_count: u32,
    screen_aspect: f32,
    manual_rotation: u32,
    tile_image_aspects: vec4<f32>,
    shadow_tint: vec3<f32>,
    highlight_tint: vec3<f32>,
};

@group(0) @binding(0) var t_diffuse: texture_2d<f32>;
@group(0) @binding(1) var s_diffuse: sampler;
@group(0) @binding(2) var<uniform> settings: Settings;
@group(0) @binding(3) var t_compare: texture_2d<f32>;
@group(0) @binding(4) var t_annotation: texture_2d<f32>;
@group(0) @binding(5) var t_grid_b: texture_2d<f32>;
@group(0) @binding(6) var t_grid_c: texture_2d<f32>;
@group(0) @binding(7) var t_grid_d: texture_2d<f32>;

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let pos = settings.view_matrix * vec3<f32>(model.position, 1.0);
    out.clip_position = vec4<f32>(pos.xy, 0.0, 1.0);
    out.tex_coords = model.tex_coords;
    return out;
}

fn line_alpha(coord: f32, at: f32) -> f32 {
    let half_w = 0.001;
    return 1.0 - smoothstep(half_w * 0.5, half_w, abs(coord - at));
}

fn diag_alpha(uv: vec2<f32>) -> f32 {
    let half_w = 0.0015;
    let d1 = abs(uv.x - uv.y);
    let d2 = abs(uv.x - (1.0 - uv.y));
    let a1 = 1.0 - smoothstep(half_w * 0.5, half_w, d1);
    let a2 = 1.0 - smoothstep(half_w * 0.5, half_w, d2);
    return max(a1, a2);
}

// Cheap PRNG-ish noise for dither, ~triangular distribution in [-0.5, 0.5].
fn tri_noise(p: vec2<f32>) -> f32 {
    let n = fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    return n - 0.5;
}

/// Map a display-orientation UV to the source texture's UV given a 90-deg
/// clockwise rotation count (0..=3). The quad's geometry already uses the
/// display-orientation aspect, so this just remaps which texel each
/// fragment samples from.
fn rotate_uv(uv: vec2<f32>, rot: u32) -> vec2<f32> {
    if (rot == 1u) { return vec2<f32>(uv.y, 1.0 - uv.x); }
    if (rot == 2u) { return vec2<f32>(1.0 - uv.x, 1.0 - uv.y); }
    if (rot == 3u) { return vec2<f32>(1.0 - uv.y, uv.x); }
    return uv;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Loupe: magnify around the cursor by remapping UV inside a screen-space disc.
    var sample_uv = in.tex_coords;
    if (settings.loupe_active != 0u) {
        let frag = in.clip_position.xy;
        let d = distance(frag, settings.loupe_center_screen);
        if (d < settings.loupe_radius) {
            sample_uv = settings.loupe_center_uv
                + (in.tex_coords - settings.loupe_center_uv) / settings.loupe_zoom;
        }
    }

    // Grid mode: partition the displayed UV into N tiles, sample each tile's
    // own texture with a remapped local UV that aspect-fits the tile so the
    // image isn't stretched. Letterbox area is filled with dark grey.
    var color: vec4<f32>;
    var on_tile_border: bool = false;
    if (settings.grid_active != 0u) {
        let count = settings.grid_count;
        var tile_idx: u32 = 0u;
        var local_uv = sample_uv;
        var tile_w_frac = 1.0;
        var tile_h_frac = 1.0;
        if (count == 4u) {
            // 2x2 grid.
            if (sample_uv.x >= 0.5) { tile_idx = tile_idx + 1u; }
            if (sample_uv.y >= 0.5) { tile_idx = tile_idx + 2u; }
            local_uv = vec2<f32>(fract(sample_uv.x * 2.0), fract(sample_uv.y * 2.0));
            tile_w_frac = 0.5;
            tile_h_frac = 0.5;
            // Border highlight: ~1px in UV terms approximated with smoothstep
            // on distance to the centre split lines.
            on_tile_border = abs(sample_uv.x - 0.5) < 0.0015
                          || abs(sample_uv.y - 0.5) < 0.0015;
        } else {
            // 1xN horizontal strip (count = 2 or 3).
            let n = f32(count);
            tile_idx = u32(floor(sample_uv.x * n));
            if (tile_idx >= count) { tile_idx = count - 1u; }
            local_uv = vec2<f32>(fract(sample_uv.x * n), sample_uv.y);
            tile_w_frac = 1.0 / n;
            on_tile_border = false;
            if (count >= 2u) {
                on_tile_border = abs(sample_uv.x - 1.0 / n) < 0.0015;
            }
            if (count >= 3u) {
                on_tile_border = on_tile_border || abs(sample_uv.x - 2.0 / n) < 0.0015;
            }
        }
        // Each tile's pixel aspect = (screen_aspect * tile_w_frac) / tile_h_frac.
        let tile_aspect = settings.screen_aspect * tile_w_frac / tile_h_frac;
        let img_aspect = settings.tile_image_aspects[tile_idx];
        var img_uv = local_uv;
        if (img_aspect > tile_aspect) {
            // Image wider than tile -> letterbox top/bottom.
            let scale = tile_aspect / img_aspect;
            img_uv.y = (local_uv.y - 0.5) / scale + 0.5;
        } else {
            let scale = img_aspect / tile_aspect;
            img_uv.x = (local_uv.x - 0.5) / scale + 0.5;
        }
        if (img_uv.x < 0.0 || img_uv.x > 1.0 || img_uv.y < 0.0 || img_uv.y > 1.0) {
            // Letterbox region.
            color = vec4<f32>(0.08, 0.08, 0.08, 1.0);
        } else {
            switch tile_idx {
                case 0u: { color = textureSample(t_diffuse, s_diffuse, img_uv); }
                case 1u: { color = textureSample(t_grid_b, s_diffuse, img_uv); }
                case 2u: { color = textureSample(t_grid_c, s_diffuse, img_uv); }
                default: { color = textureSample(t_grid_d, s_diffuse, img_uv); }
            }
        }
    } else if (settings.compare_active != 0u && sample_uv.x > settings.compare_divider) {
        let tex_uv = rotate_uv(sample_uv, settings.manual_rotation);
        color = textureSample(t_compare, s_diffuse, tex_uv);
    } else {
        let tex_uv = rotate_uv(sample_uv, settings.manual_rotation);
        color = textureSample(t_diffuse, s_diffuse, tex_uv);
    }

    let gray = dot(color.rgb, vec3<f32>(0.299, 0.587, 0.114));
    var final_color = mix(color.rgb, vec3<f32>(gray), settings.grayscale);

    // Value study: posterize luma to N bands. Always grayscale (artist
    // convention) and overrides the grayscale slider while active.
    if (settings.posterize_active != 0u && settings.posterize_levels >= 2u) {
        let levels = f32(settings.posterize_levels);
        let q = floor(gray * levels) / (levels - 1.0);
        final_color = vec3<f32>(clamp(q, 0.0, 1.0));
    }

    // Composition overlays (use the original UV so they stay anchored to the
    // image, not the magnified loupe region).
    var overlay = 0.0;
    if (settings.overlay_opacity > 0.0) {
        let ov = in.tex_coords;
        switch settings.overlay_mode {
            case 1u: {
                let g = step(vec2<f32>(0.98), fract(ov * settings.grid_size));
                overlay = max(g.x, g.y);
            }
            case 2u: {
                overlay = max(
                    max(line_alpha(ov.x, 1.0 / 3.0), line_alpha(ov.x, 2.0 / 3.0)),
                    max(line_alpha(ov.y, 1.0 / 3.0), line_alpha(ov.y, 2.0 / 3.0)),
                );
            }
            case 3u: {
                overlay = max(
                    max(line_alpha(ov.x, 0.382), line_alpha(ov.x, 0.618)),
                    max(line_alpha(ov.y, 0.382), line_alpha(ov.y, 0.618)),
                );
            }
            case 4u: {
                overlay = diag_alpha(ov);
            }
            default: {}
        }
        overlay *= settings.overlay_opacity;
    }
    final_color = mix(final_color, vec3<f32>(1.0), overlay);

    // Compare divider line (in image UV so it stays at the split point).
    if (settings.compare_active != 0u) {
        let dl = line_alpha(in.tex_coords.x, settings.compare_divider);
        final_color = mix(final_color, vec3<f32>(1.0), dl * 0.6);
    }

    // Loupe edge ring (in screen pixels so it's a constant 1.5 px wide).
    if (settings.loupe_active != 0u) {
        let frag = in.clip_position.xy;
        let edge = abs(distance(frag, settings.loupe_center_screen) - settings.loupe_radius);
        if (edge < 1.5) {
            let t = 1.0 - edge / 1.5;
            final_color = mix(final_color, vec3<f32>(1.0), t * 0.8);
        }
    }

    // Split-tone: lerp between shadow and highlight tints by per-pixel luma,
    // multiply (tint scaled by 2 so a neutral 0.5 tint preserves brightness),
    // then mix toward original by `amount`.
    if (settings.split_tone_active != 0u && settings.split_tone_amount > 0.0) {
        let luma = dot(final_color, vec3<f32>(0.299, 0.587, 0.114));
        let tint = mix(settings.shadow_tint, settings.highlight_tint, luma);
        let toned = final_color * tint * 2.0;
        final_color = mix(final_color, toned, settings.split_tone_amount);
    }

    // Dither to break up subtle banding from grayscale + overlay blends when
    // quantizing back down to 8-bit at scanout. Cost: one sin + one fract.
    if (settings.dither != 0u) {
        let n = tri_noise(in.clip_position.xy) / 255.0;
        final_color = final_color + vec3<f32>(n);
    }

    // Annotation overlay: sampled at the same image-UV as the photo (so it
    // tracks zoom, pan, flip, and is magnified inside the loupe), src-over
    // blended on top of all colour effects so pen strokes stay legible
    // regardless of grayscale/posterize/split-tone.
    if (settings.annotation_active != 0u) {
        let ann_uv = rotate_uv(sample_uv, settings.manual_rotation);
        let ann = textureSample(t_annotation, s_diffuse, ann_uv);
        final_color = mix(final_color, ann.rgb, ann.a);
    }

    // Grid mode tile borders: subtle white seam between adjacent tiles so the
    // user can see where one image ends and the next begins.
    if (settings.grid_active != 0u && on_tile_border) {
        final_color = mix(final_color, vec3<f32>(1.0), 0.4);
    }

    // Clipping warning: any channel at the extremes flagged with hi-vis colors.
    // Magenta = highlight clip, cyan = shadow clip. Sampled from the original
    // texture color so split-tone/grayscale don't false-trigger it.
    if (settings.clipping_warning != 0u) {
        let hi = max(max(color.r, color.g), color.b);
        let lo = min(min(color.r, color.g), color.b);
        if (hi >= 254.0 / 255.0) {
            final_color = vec3<f32>(1.0, 0.0, 1.0);
        } else if (lo <= 1.0 / 255.0) {
            final_color = vec3<f32>(0.0, 1.0, 1.0);
        }
    }

    return vec4<f32>(final_color, color.a);
}
