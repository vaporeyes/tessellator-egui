// ABOUTME: Fullscreen-quad shader for the photo viewport.
// ABOUTME: Applies a 3x3 view matrix and optional grayscale + grid overlays.

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
    grid_opacity: f32,
    grid_size: f32,
    padding: f32,
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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(t_diffuse, s_diffuse, in.tex_coords);

    let gray = dot(color.rgb, vec3<f32>(0.299, 0.587, 0.114));
    let final_color = mix(color.rgb, vec3<f32>(gray), settings.grayscale);

    var grid_effect = 0.0;
    if (settings.grid_opacity > 0.0) {
        let grid = step(vec2<f32>(0.98), fract(in.tex_coords * settings.grid_size));
        grid_effect = max(grid[0], grid[1]) * settings.grid_opacity;
    }

    return vec4<f32>(mix(final_color, vec3<f32>(1.0), grid_effect), color.a);
}
