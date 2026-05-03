// ABOUTME: Fullscreen-quad shader for the photo viewport.
// ABOUTME: Applies a 3x3 view matrix and optional grayscale + composition overlays.

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
};

@group(0) @binding(0) var t_diffuse: texture_2d<f32>;
@group(0) @binding(1) var s_diffuse: sampler;
@group(0) @binding(2) var<uniform> settings: Settings;

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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(t_diffuse, s_diffuse, in.tex_coords);

    let gray = dot(color.rgb, vec3<f32>(0.299, 0.587, 0.114));
    let final_color = mix(color.rgb, vec3<f32>(gray), settings.grayscale);

    var overlay = 0.0;
    if (settings.overlay_opacity > 0.0) {
        switch settings.overlay_mode {
            // 1 = Grid (user-controlled spacing).
            case 1u: {
                let g = step(vec2<f32>(0.98), fract(in.tex_coords * settings.grid_size));
                overlay = max(g.x, g.y);
            }
            // 2 = Rule of thirds.
            case 2u: {
                let v1 = line_alpha(in.tex_coords.x, 1.0 / 3.0);
                let v2 = line_alpha(in.tex_coords.x, 2.0 / 3.0);
                let h1 = line_alpha(in.tex_coords.y, 1.0 / 3.0);
                let h2 = line_alpha(in.tex_coords.y, 2.0 / 3.0);
                overlay = max(max(v1, v2), max(h1, h2));
            }
            // 3 = Golden ratio (~0.382 / 0.618).
            case 3u: {
                let v1 = line_alpha(in.tex_coords.x, 0.382);
                let v2 = line_alpha(in.tex_coords.x, 0.618);
                let h1 = line_alpha(in.tex_coords.y, 0.382);
                let h2 = line_alpha(in.tex_coords.y, 0.618);
                overlay = max(max(v1, v2), max(h1, h2));
            }
            // 4 = Diagonal cross.
            case 4u: {
                overlay = diag_alpha(in.tex_coords);
            }
            default: {}
        }
        overlay *= settings.overlay_opacity;
    }

    return vec4<f32>(mix(final_color, vec3<f32>(1.0), overlay), color.a);
}
