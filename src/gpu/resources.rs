// ABOUTME: WGPU pipeline, vertex buffer, sampler, and texture upload for the viewer.
// ABOUTME: ShaderSettings is the uniform buffer matching the WGSL Settings struct.

use crate::io::DecodedImage;
use eframe::wgpu;
use wgpu::util::DeviceExt;

const SHADER_SOURCE: &str = include_str!("../shader.wgsl");

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
    pub _pad0: f32,
    pub _pad1: f32,
}

// Layout matches WGSL Settings: 48B mat3x3 + four 16B blocks of scalars/vec2s.
// A future field reorder that breaks WGSL alignment will fail to compile.
const _: () = assert!(std::mem::size_of::<ShaderSettings>() == 112);

pub struct TessellatorResources {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    vertex_buffer: wgpu::Buffer,
    settings_buffer: wgpu::Buffer,
    main_texture: Option<wgpu::Texture>,
    compare_texture: Option<wgpu::Texture>,
    bind_group: Option<wgpu::BindGroup>,
}

impl TessellatorResources {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        log::debug!("Creating TessellatorResources with format {:?}", format);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tessellator.shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tessellator.pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tessellator.pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
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
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let vertices: [f32; 24] = [
            -1.0,  1.0,  0.0, 0.0,
            -1.0, -1.0,  0.0, 1.0,
             1.0,  1.0,  1.0, 0.0,
             1.0,  1.0,  1.0, 0.0,
            -1.0, -1.0,  0.0, 1.0,
             1.0, -1.0,  1.0, 1.0,
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

        Self {
            pipeline,
            bind_group_layout,
            sampler,
            vertex_buffer,
            settings_buffer,
            main_texture: None,
            compare_texture: None,
            bind_group: None,
        }
    }

    pub fn set_main_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: &DecodedImage,
    ) {
        self.main_texture = Some(create_image_texture(device, queue, image, "main"));
        self.rebuild_bind_group(device);
    }

    pub fn set_compare_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: Option<&DecodedImage>,
    ) {
        self.compare_texture =
            image.map(|img| create_image_texture(device, queue, img, "compare"));
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
            ],
        }));
    }

    pub fn update_settings(&self, queue: &wgpu::Queue, settings: ShaderSettings) {
        queue.write_buffer(&self.settings_buffer, 0, bytemuck::cast_slice(&[settings]));
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
}

fn create_image_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image: &DecodedImage,
    label_suffix: &str,
) -> wgpu::Texture {
    let width = image.width;
    let height = image.height;
    let mip_level_count = image.mips.len() as u32;
    log::info!(
        "Uploading {} texture: {}x{} ({} mips)",
        label_suffix,
        width,
        height,
        mip_level_count
    );
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("tessellator.image_texture"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (level, mip) in image.mips.iter().enumerate() {
        upload_mip(queue, &texture, level as u32, &mip.rgba, mip.width, mip.height);
    }
    texture
}

fn upload_mip(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    level: u32,
    rgba: &[u8],
    width: u32,
    height: u32,
) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: level,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * width),
            rows_per_image: None,
        },
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );
}
