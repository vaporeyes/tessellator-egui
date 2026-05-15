// ABOUTME: WGPU pipeline, vertex buffer, sampler, and texture upload for the viewer.
// ABOUTME: ShaderSettings is the uniform buffer matching the WGSL Settings struct.

use crate::io::DecodedImage;
use eframe::wgpu;
use wgpu::util::DeviceExt;

const SHADER_SOURCE: &str = include_str!("../shader.wgsl");

const BLIT_SHADER_SOURCE: &str = "
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

@vertex
fn vs_blit(@location(0) position: vec2<f32>, @location(1) tex_coords: vec2<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(position, 0.0, 1.0);
    out.tex_coords = tex_coords;
    return out;
}

@group(0) @binding(0) var t_src: texture_2d<f32>;
@group(0) @binding(1) var s_src: sampler;

@fragment
fn fs_blit(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_src, s_src, in.tex_coords);
}
";

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShaderSettings {
    pub view_matrix: [f32; 12],
    pub grayscale: f32,
    pub overlay_opacity: f32,
    pub grid_size: f32,
    /// 0 = none, 1 = grid, 2 = rule of thirds, 3 = golden ratio, 4 = diagonal.
    pub overlay_mode: u32,
    pub compare_divider: f32,
    pub compare_active: u32,
    pub loupe_active: u32,
    pub loupe_zoom: f32,
    pub loupe_center_uv: [f32; 2],
    pub loupe_center_screen: [f32; 2],
    pub loupe_radius: f32,
    pub dither: u32,
    pub split_tone_active: u32,
    pub split_tone_amount: f32,
    pub clipping_warning: u32,
    pub posterize_active: u32,
    pub posterize_levels: u32,
    pub annotation_active: u32,
    /// Multi-image grid (A/B/C/D side-by-side compare). 0 = off, else number
    /// of tiles (2, 3, or 4).
    pub grid_active: u32,
    pub grid_count: u32,
    /// Width / height of the displayed viewport, used so the shader can
    /// derive each tile's aspect ratio for letterboxing.
    pub screen_aspect: f32,
    /// Manual image rotation in 90-degree clockwise quarters: 0/1/2/3 =
    /// 0/90/180/270 deg. Source UVs are remapped in-shader so the quad's
    /// aspect (computed CPU-side) stays in display orientation.
    pub manual_rotation: u32,
    /// w/h of each tile's source image. Index 0 is the main (current) image,
    /// 1..3 are the user-picked grid additions. Unused slots are 0.0.
    pub tile_image_aspects: [f32; 4],
    /// 4th component is WGSL vec3 padding; ignored by the shader.
    pub shadow_tint: [f32; 4],
    pub highlight_tint: [f32; 4],
}

// Layout matches WGSL Settings: 48B mat3x3 + scalars + grid block + two 16B
// vec3 blocks. A future field reorder that breaks WGSL alignment will fail
// to compile here. Per-field offset asserts below catch silent reorders that
// keep the same total size but shift uniforms by 4 bytes - a class of bug
// that previously could only be detected by visual inspection of the shader
// output.
const _: () = assert!(std::mem::size_of::<ShaderSettings>() == 192);
const _: () = assert!(std::mem::offset_of!(ShaderSettings, view_matrix) == 0);
const _: () = assert!(std::mem::offset_of!(ShaderSettings, grayscale) == 48);
const _: () = assert!(std::mem::offset_of!(ShaderSettings, overlay_opacity) == 52);
const _: () = assert!(std::mem::offset_of!(ShaderSettings, grid_size) == 56);
const _: () = assert!(std::mem::offset_of!(ShaderSettings, overlay_mode) == 60);
const _: () = assert!(std::mem::offset_of!(ShaderSettings, compare_divider) == 64);
const _: () = assert!(std::mem::offset_of!(ShaderSettings, compare_active) == 68);
const _: () = assert!(std::mem::offset_of!(ShaderSettings, loupe_active) == 72);
const _: () = assert!(std::mem::offset_of!(ShaderSettings, loupe_zoom) == 76);
const _: () = assert!(std::mem::offset_of!(ShaderSettings, loupe_center_uv) == 80);
const _: () = assert!(std::mem::offset_of!(ShaderSettings, loupe_center_screen) == 88);
const _: () = assert!(std::mem::offset_of!(ShaderSettings, loupe_radius) == 96);
const _: () = assert!(std::mem::offset_of!(ShaderSettings, dither) == 100);
const _: () = assert!(std::mem::offset_of!(ShaderSettings, split_tone_active) == 104);
const _: () = assert!(std::mem::offset_of!(ShaderSettings, split_tone_amount) == 108);
const _: () = assert!(std::mem::offset_of!(ShaderSettings, clipping_warning) == 112);
const _: () = assert!(std::mem::offset_of!(ShaderSettings, posterize_active) == 116);
const _: () = assert!(std::mem::offset_of!(ShaderSettings, posterize_levels) == 120);
const _: () = assert!(std::mem::offset_of!(ShaderSettings, annotation_active) == 124);
const _: () = assert!(std::mem::offset_of!(ShaderSettings, grid_active) == 128);
const _: () = assert!(std::mem::offset_of!(ShaderSettings, grid_count) == 132);
const _: () = assert!(std::mem::offset_of!(ShaderSettings, screen_aspect) == 136);
const _: () = assert!(std::mem::offset_of!(ShaderSettings, tile_image_aspects) == 144);
const _: () = assert!(std::mem::offset_of!(ShaderSettings, shadow_tint) == 160);
const _: () = assert!(std::mem::offset_of!(ShaderSettings, highlight_tint) == 176);

pub struct TessellatorResources {
    pipeline: wgpu::RenderPipeline,
    /// Sibling of `pipeline`, identical except the color target is Rgba8Unorm
    /// for offscreen render-to-buffer used by the Export feature.
    export_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    vertex_buffer: wgpu::Buffer,
    settings_buffer: wgpu::Buffer,

    blit_pipeline: wgpu::RenderPipeline,
    blit_bind_group_layout: wgpu::BindGroupLayout,

    main_texture: Option<wgpu::Texture>,
    compare_texture: Option<wgpu::Texture>,
    annotation_texture: Option<wgpu::Texture>,
    /// 1x1 transparent fallback so bind slot 4 always has something valid
    /// when no annotation layer exists.
    annotation_placeholder: wgpu::Texture,
    /// Grid mode tiles B / C / D (slot 0 = main_texture). Each falls back to
    /// the main texture when grid mode is off so the bindings always sample
    /// something valid.
    grid_b_texture: Option<wgpu::Texture>,
    grid_c_texture: Option<wgpu::Texture>,
    grid_d_texture: Option<wgpu::Texture>,
    bind_group: Option<wgpu::BindGroup>,
}

#[derive(Clone)]
pub struct ExportResources {
    export_pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    settings_buffer: wgpu::Buffer,
}

impl TessellatorResources {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        log::debug!("Creating TessellatorResources with format {:?}", format);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tessellator.shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });

        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tessellator.blit_shader"),
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER_SOURCE.into()),
        });

        let blit_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("tessellator.blit_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tessellator.blit_pipeline_layout"),
            bind_group_layouts: &[Some(&blit_bind_group_layout)],
            ..Default::default()
        });

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tessellator.blit_pipeline"),
            layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_blit"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 16,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 8,
                            shader_location: 1,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_blit"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tessellator.bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tessellator.pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            ..Default::default()
        });

        // Vertex layout shared between the swapchain and export pipelines.
        let vertex_attrs = [
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 8,
                shader_location: 1,
            },
        ];
        let make_pipeline = |label: &str, target_format: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: 16,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &vertex_attrs,
                    }],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline = make_pipeline("tessellator.pipeline", format);
        let export_pipeline = make_pipeline(
            "tessellator.export_pipeline",
            wgpu::TextureFormat::Rgba8Unorm,
        );

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let vertices: [f32; 24] = [
            -1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0,
            -1.0, -1.0, 0.0, 1.0, 1.0, -1.0, 1.0, 1.0,
        ];
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tessellator.vertex_buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let settings_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tessellator.settings_buffer"),
            size: std::mem::size_of::<ShaderSettings>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let annotation_placeholder = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tessellator.annotation_placeholder"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // Initialise the placeholder to fully transparent. write_texture is
        // safe before the first frame because the queue runs at submit time.
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &annotation_placeholder,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[0u8, 0, 0, 0],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        Self {
            pipeline,
            export_pipeline,
            bind_group_layout,
            sampler,
            vertex_buffer,
            settings_buffer,
            main_texture: None,
            compare_texture: None,
            annotation_texture: None,
            annotation_placeholder,
            grid_b_texture: None,
            grid_c_texture: None,
            grid_d_texture: None,
            bind_group: None,
            blit_pipeline,
            blit_bind_group_layout,
        }
    }

    pub fn set_main_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: &DecodedImage,
    ) {
        self.main_texture = Some(self.create_image_texture(device, queue, image, "main"));
        self.rebuild_bind_group(device);
    }

    pub fn set_compare_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: Option<&DecodedImage>,
    ) {
        self.compare_texture =
            image.map(|img| self.create_image_texture(device, queue, img, "compare"));
        self.rebuild_bind_group(device);
    }

    /// Create a fresh transparent annotation texture of the given size and
    /// optionally upload initial pixels (full RGBA8). Replaces any prior
    /// annotation texture.
    pub fn set_annotation_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tessellator.annotation_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.annotation_texture = Some(texture);
        self.rebuild_bind_group(device);
    }

    /// Patch a sub-region of the existing annotation texture. No-op if no
    /// annotation texture is currently bound (the app is expected to call
    /// `set_annotation_texture` first when entering annotation mode).
    pub fn patch_annotation_region(
        &self,
        queue: &wgpu::Queue,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) {
        let Some(texture) = &self.annotation_texture else {
            return;
        };
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    pub fn clear_annotation_texture(&mut self, device: &wgpu::Device) {
        if self.annotation_texture.take().is_some() {
            self.rebuild_bind_group(device);
        }
    }

    /// Set / clear an additional grid-mode tile (slot 1 = B, 2 = C, 3 = D;
    /// slot 0 is always main_texture). Passing `None` clears the slot.
    pub fn set_grid_tile(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        slot: u32,
        image: Option<&DecodedImage>,
    ) {
        let label = match slot {
            1 => "grid_b",
            2 => "grid_c",
            3 => "grid_d",
            _ => return,
        };
        let texture = image.map(|img| self.create_image_texture(device, queue, img, label));
        match slot {
            1 => self.grid_b_texture = texture,
            2 => self.grid_c_texture = texture,
            3 => self.grid_d_texture = texture,
            _ => unreachable!(),
        }
        self.rebuild_bind_group(device);
    }

    /// Rebuild the bind group from the current main + (optional) compare
    /// textures. The compare slot falls back to the main texture so the
    /// shader's binding always has something to sample, even when compare
    /// mode is off.
    fn rebuild_bind_group(&mut self, device: &wgpu::Device) {
        let Some(main) = &self.main_texture else {
            self.bind_group = None;
            return;
        };
        let main_view = main.create_view(&wgpu::TextureViewDescriptor::default());
        let compare_view = self
            .compare_texture
            .as_ref()
            .unwrap_or(main)
            .create_view(&wgpu::TextureViewDescriptor::default());
        let annotation_view = self
            .annotation_texture
            .as_ref()
            .unwrap_or(&self.annotation_placeholder)
            .create_view(&wgpu::TextureViewDescriptor::default());
        let grid_b_view = self
            .grid_b_texture
            .as_ref()
            .unwrap_or(main)
            .create_view(&wgpu::TextureViewDescriptor::default());
        let grid_c_view = self
            .grid_c_texture
            .as_ref()
            .unwrap_or(main)
            .create_view(&wgpu::TextureViewDescriptor::default());
        let grid_d_view = self
            .grid_d_texture
            .as_ref()
            .unwrap_or(main)
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tessellator.bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&main_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.settings_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&compare_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&annotation_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&grid_b_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&grid_c_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&grid_d_view),
                },
            ],
        }));
    }

    pub fn update_settings(&self, queue: &wgpu::Queue, settings: ShaderSettings) {
        queue.write_buffer(&self.settings_buffer, 0, bytemuck::cast_slice(&[settings]));
    }

    pub fn export_resources(&self) -> Option<ExportResources> {
        Some(ExportResources {
            export_pipeline: self.export_pipeline.clone(),
            bind_group: self.bind_group.as_ref()?.clone(),
            vertex_buffer: self.vertex_buffer.clone(),
            settings_buffer: self.settings_buffer.clone(),
        })
    }

    pub fn pipeline(&self) -> &wgpu::RenderPipeline {
        &self.pipeline
    }

    pub fn vertex_buffer(&self) -> &wgpu::Buffer {
        &self.vertex_buffer
    }

    pub fn current_bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.bind_group.as_ref()
    }

    fn create_image_texture(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: &DecodedImage,
        label_suffix: &str,
    ) -> wgpu::Texture {
        let width = image.width;
        let height = image.height;
        let mip_level_count = (width.max(height).max(1)).ilog2() + 1;

        log::info!(
            "Uploading {} texture: {}x{} ({} mips, GPU generated)",
            label_suffix,
            width,
            height,
            mip_level_count
        );

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tessellator.image_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        // Upload level 0.
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &image.rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        // Generate mips via blit.
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        for level in 1..mip_level_count {
            let view_src = texture.create_view(&wgpu::TextureViewDescriptor {
                label: None,
                format: None,
                dimension: None,
                usage: None,
                aspect: wgpu::TextureAspect::All,
                base_mip_level: level - 1,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: None,
            });
            let view_dst = texture.create_view(&wgpu::TextureViewDescriptor {
                label: None,
                format: None,
                dimension: None,
                usage: None,
                aspect: wgpu::TextureAspect::All,
                base_mip_level: level,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: None,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.blit_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view_src),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });

            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view_dst,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                rpass.set_pipeline(&self.blit_pipeline);
                rpass.set_bind_group(0, &bind_group, &[]);
                rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                rpass.draw(0..6, 0..1);
            }
        }

        queue.submit(Some(encoder.finish()));

        texture
    }
}

pub fn render_export_resources_to_rgba8(
    resources: &ExportResources,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    settings: ShaderSettings,
) -> Result<Vec<u8>, &'static str> {
    let unpadded = 4 * width;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = unpadded.div_ceil(align) * align;
    let buffer_size = (padded as u64) * (height as u64);

    let color_target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("tessellator.export_target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color_target.create_view(&wgpu::TextureViewDescriptor::default());

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tessellator.export_readback"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    queue.write_buffer(
        &resources.settings_buffer,
        0,
        bytemuck::cast_slice(&[settings]),
    );

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("tessellator.export_encoder"),
    });
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("tessellator.export_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(&resources.export_pipeline);
        rpass.set_bind_group(0, &resources.bind_group, &[]);
        rpass.set_vertex_buffer(0, resources.vertex_buffer.slice(..));
        rpass.draw(0..6, 0..1);
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &color_target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|_| "device.poll failed")?;
    rx.recv()
        .map_err(|_| "map_async result channel dropped")?
        .map_err(|_| "buffer map failed")?;

    let data = slice.get_mapped_range();
    let mut out = Vec::with_capacity((unpadded as usize) * (height as usize));
    for row in 0..height as usize {
        let start = row * padded as usize;
        let end = start + unpadded as usize;
        out.extend_from_slice(&data[start..end]);
    }
    drop(data);
    readback.unmap();

    Ok(out)
}
