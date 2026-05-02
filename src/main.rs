// ABOUTME: Main entry point for Tessellator, a high-performance photo viewer.
// ABOUTME: Handles UI layout, state management, and WGPU integration.

use eframe::egui;
use eframe::wgpu;
use std::path::{Path, PathBuf};
use crossbeam_channel::{Receiver, Sender};
use std::sync::Arc;
use std::collections::HashSet;

// ABOUTME: Main entry point for Tessellator, a high-performance photo viewer.
// ABOUTME: Handles UI layout, state management, and WGPU integration.

fn main() -> eframe::Result {
    env_logger::init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 720.0])
            .with_drag_and_drop(true),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    eframe::run_native(
        "Tessellator",
        options,
        Box::new(|cc| Ok(Box::new(TessellatorApp::new(cc)))),
    )
}

const SHADER_SOURCE: &str = "
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

@group(0) @binding(2) var<uniform> settings: Settings;

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let pos = settings.view_matrix * vec3<f32>(model.position, 1.0);
    out.clip_position = vec4<f32>(pos.xy, 0.0, 1.0);
    out.tex_coords = model.tex_coords;
    return out;
}

@group(0) @binding(0) var t_diffuse: texture_2d<f32>;
@group(0) @binding(1) var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Simplified shader to verify rendering
    let color = textureSample(t_diffuse, s_diffuse, in.tex_coords);
    
    // Grayscale
    let gray = dot(color.rgb, vec3<f32>(0.299, 0.587, 0.114));
    let final_color = mix(color.rgb, vec3<f32>(gray), settings.grayscale);
    
    // Grid
    var grid_effect = 0.0;
    if (settings.grid_opacity > 0.0) {
        let grid = step(vec2<f32>(0.98), fract(in.tex_coords * settings.grid_size));
        grid_effect = max(grid[0], grid[1]) * settings.grid_opacity;
    }
    
    return vec4<f32>(mix(final_color, vec3<f32>(1.0), grid_effect), color.a);
}
";

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ShaderSettings {
    view_matrix: [f32; 12], // 3x3 matrix (padded to 4-float alignment)
    grayscale: f32,
    grid_opacity: f32,
    grid_size: f32,
    padding: f32,
}

struct TessellatorResources {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    vertex_buffer: wgpu::Buffer,
    settings_buffer: wgpu::Buffer,
    current_texture: Option<(wgpu::Texture, wgpu::BindGroup, [u32; 2])>,
}

impl TessellatorResources {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        println!("Creating TessellatorResources with format {:?}", format);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
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
        use wgpu::util::DeviceExt;
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let settings_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Settings Buffer"),
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
            current_texture: None,
        }
    }

    fn update_texture(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, image: &image::DynamicImage) {
        println!("Uploading high-res texture: {}x{}", image.width(), image.height());
        let rgba = image.to_rgba8();
        let (width, height) = rgba.dimensions();
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("High-Res Texture"),
            size,
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
            &rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: None,
            },
            size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.settings_buffer.as_entire_binding(),
                },
            ],
        });

        self.current_texture = Some((texture, bind_group, [width, height]));
    }

    fn update_settings(&self, queue: &wgpu::Queue, settings: ShaderSettings) {
        queue.write_buffer(&self.settings_buffer, 0, bytemuck::cast_slice(&[settings]));
    }
}

struct TessellatorCallback {
    image: Option<Arc<image::DynamicImage>>,
    settings: ShaderSettings,
    format: wgpu::TextureFormat,
}

impl egui_wgpu::CallbackTrait for TessellatorCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if !resources.contains::<TessellatorResources>() {
            println!("Inserting TessellatorResources");
            resources.insert(TessellatorResources::new(device, self.format));
        }
        
        let tess_resources = resources.get_mut::<TessellatorResources>().unwrap();
        
        if let Some(image) = &self.image {
            println!("Callback prepare: uploading pending image");
            tess_resources.update_texture(device, queue, image);
        }
        
        tess_resources.update_settings(queue, self.settings);
        
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let tess_resources = resources.get::<TessellatorResources>().unwrap();
        if let Some((_, bind_group, _)) = &tess_resources.current_texture {
            render_pass.set_pipeline(&tess_resources.pipeline);
            render_pass.set_bind_group(0, bind_group, &[]);
            render_pass.set_vertex_buffer(0, tess_resources.vertex_buffer.slice(..));
            render_pass.draw(0..6, 0..1);
        }
    }
}

#[derive(Clone)]
struct FileEntry {
    path: PathBuf,
    name: String,
    thumbnail: Option<egui::TextureHandle>,
}

enum Message {
    FilesFound(Vec<PathBuf>),
    ThumbnailLoaded { path: PathBuf, image: egui::ColorImage },
    HighResLoaded { image: Arc<image::DynamicImage> },
}

struct TessellatorApp {
    folder_path: Option<PathBuf>,
    files: Vec<FileEntry>,
    selected_index: Option<usize>,
    receiver: Receiver<Message>,
    sender: Sender<Message>,
    pending_thumbnails: HashSet<PathBuf>,
    current_high_res: Option<PathBuf>,
    wgpu_state: Arc<egui_wgpu::RenderState>,

    // Artist tool settings
    grayscale: f32,
    grid_opacity: f32,
    grid_size: f32,
    zoom: f32,
    pan: egui::Vec2,
    
    // Pending high-res image to be uploaded to GPU
    pending_high_res: Option<Arc<image::DynamicImage>>,
}

impl TessellatorApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let wgpu_render_state = cc.wgpu_render_state.clone().expect("WGPU render state not available");
        let (sender, receiver) = crossbeam_channel::unbounded();

        Self {
            folder_path: None,
            files: Vec::new(),
            selected_index: None,
            receiver,
            sender,
            pending_thumbnails: HashSet::new(),
            current_high_res: None,
            wgpu_state: Arc::new(wgpu_render_state),
            grayscale: 0.0,
            grid_opacity: 0.0,
            grid_size: 10.0,
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            pending_high_res: None,
        }
    }

    fn scan_folder(&mut self, path: PathBuf, ctx: egui::Context) {
        self.files.clear();
        self.selected_index = None;
        self.pending_thumbnails.clear();
        
        let sender = self.sender.clone();
        let path_clone = path.clone();
        
        std::thread::spawn(move || {
            let mut found = Vec::new();
            for entry in walkdir::WalkDir::new(path_clone)
                .max_depth(2)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().is_file()) 
            {
                if is_image_file(entry.path()) {
                    found.push(entry.path().to_path_buf());
                }
            }
            let _ = sender.send(Message::FilesFound(found));
            ctx.request_repaint();
        });
    }

    fn request_thumbnail(&mut self, path: PathBuf, ctx: egui::Context) {
        if self.pending_thumbnails.contains(&path) {
            return;
        }
        self.pending_thumbnails.insert(path.clone());
        
        let sender = self.sender.clone();
        rayon::spawn(move || {
            if let Ok(img) = image::open(&path) {
                let thumbnail = img.thumbnail(128, 128);
                let size = [thumbnail.width() as usize, thumbnail.height() as usize];
                let pixels = thumbnail.to_rgba8();
                let color_image = egui::ColorImage::from_rgba_unmultiplied(size, pixels.as_flat_samples().as_slice());
                let _ = sender.send(Message::ThumbnailLoaded { path, image: color_image });
                ctx.request_repaint();
            }
        });
    }

    fn request_high_res(&mut self, path: PathBuf, ctx: egui::Context) {
        if self.current_high_res == Some(path.clone()) {
            println!("High-res request ignored: already selected {:?}", path.file_name().unwrap_or_default());
            return;
        }
        self.current_high_res = Some(path.clone());
        println!("Starting high-res load for: {:?}", path.file_name().unwrap_or_default());
        
        let sender = self.sender.clone();
        std::thread::spawn(move || {
            println!("Background thread: Opening image {:?}", path.file_name().unwrap_or_default());
            match image::open(&path) {
                Ok(img) => {
                    println!("Background thread: Image opened successfully, sending message");
                    if let Err(e) = sender.send(Message::HighResLoaded { image: Arc::new(img) }) {
                        println!("Background thread: Failed to send high-res message: {}", e);
                    } else {
                        println!("Background thread: High-res message sent");
                        ctx.request_repaint();
                    }
                }
                Err(e) => {
                    println!("Background thread: Failed to open high-res image {:?}: {}", path, e);
                }
            }
        });
    }
}

fn is_image_file(path: &Path) -> bool {
    let extensions = ["jpg", "jpeg", "png", "webp", "bmp", "tiff"];
    path.extension()
        .and_then(|s| s.to_str())
        .map(|s| extensions.contains(&s.to_lowercase().as_str()))
        .unwrap_or(false)
}

impl eframe::App for TessellatorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        while let Ok(msg) = self.receiver.try_recv() {
            match msg {
                Message::FilesFound(paths) => {
                    self.files = paths.into_iter().map(|p| FileEntry {
                        name: p.file_name().unwrap_or_default().to_string_lossy().to_string(),
                        path: p,
                        thumbnail: None,
                    }).collect();
                }
                Message::ThumbnailLoaded { path, image } => {
                    self.pending_thumbnails.remove(&path);
                    if let Some(entry) = self.files.iter_mut().find(|f| f.path == path) {
                        let handle = ctx.load_texture(
                            path.to_string_lossy(),
                            image,
                            egui::TextureOptions::default()
                        );
                        entry.thumbnail = Some(handle);
                    }
                }
                Message::HighResLoaded { image, .. } => {
                    println!("High-res image loaded, triggering GPU upload");
                    let mut renderer = self.wgpu_state.renderer.write();
                    if let Some(tess_resources) = renderer.callback_resources.get_mut::<TessellatorResources>() {
                        tess_resources.update_texture(&self.wgpu_state.device, &self.wgpu_state.queue, &image);
                    } else {
                        // If resources don't exist yet, we'll store them for the first paint to handle.
                        self.pending_high_res = Some(image);
                    }
                    self.zoom = 1.0;
                    self.pan = egui::Vec2::ZERO;
                    ctx.request_repaint();
                }
            }
        }

        egui::SidePanel::left("sidebar")
            .resizable(true)
            .default_width(250.0)
            .show(ctx, |ui| {
                ui.heading("Tessellator");
                if ui.button("Open Folder...").clicked() {
                    if let Some(path) = rfd::FileDialog::new().pick_folder() {
                        self.folder_path = Some(path.clone());
                        self.scan_folder(path, ctx.clone());
                    }
                }
                ui.separator();
                
                let row_height = 40.0;
                egui::ScrollArea::vertical().show_rows(ui, row_height, self.files.len(), |ui, row_range| {
                    for i in row_range {
                        let entry = &self.files[i];
                        let is_selected = self.selected_index == Some(i);
                        
                        let response = ui.horizontal(|ui| {
                            if let Some(texture) = &entry.thumbnail {
                                ui.image((texture.id(), egui::vec2(32.0, 32.0)));
                            } else {
                                ui.allocate_space(egui::vec2(32.0, 32.0));
                            }
                            ui.selectable_label(is_selected, &entry.name)
                        }).inner;

                        if response.clicked() {
                            println!("Selected image: {}", entry.name);
                            self.selected_index = Some(i);
                            self.request_high_res(entry.path.clone(), ctx.clone());
                        }
                    }
                });

                let missing_thumbs: Vec<PathBuf> = self.files.iter()
                    .filter(|f| f.thumbnail.is_none() && !self.pending_thumbnails.contains(&f.path))
                    .take(10)
                    .map(|f| f.path.clone())
                    .collect();
                for path in missing_thumbs {
                    self.request_thumbnail(path, ctx.clone());
                }
            });

        egui::TopBottomPanel::top("tools").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Grayscale:");
                ui.add(egui::Slider::new(&mut self.grayscale, 0.0..=1.0));
                ui.separator();
                ui.label("Grid Opacity:");
                ui.add(egui::Slider::new(&mut self.grid_opacity, 0.0..=1.0));
                ui.label("Grid Size:");
                ui.add(egui::Slider::new(&mut self.grid_size, 1.0..=50.0));
                ui.separator();
                if ui.button("Reset View").clicked() {
                    self.zoom = 1.0;
                    self.pan = egui::Vec2::ZERO;
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(_) = self.selected_index {
                let rect = ui.max_rect();
                let response = ui.interact(rect, ui.id(), egui::Sense::drag());

                // Zoom & Pan
                if response.dragged() {
                    self.pan += response.drag_delta() / (rect.size() * 0.5 * self.zoom);
                }
                let scroll_delta = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll_delta != 0.0 {
                    self.zoom *= (scroll_delta * 0.01).exp();
                }

                // Prepare matrix: scale * translate
                // Column-major mat3x3 (48 bytes)
                let mut matrix = [0.0f32; 12];
                matrix[0] = self.zoom;   // Col 0, X
                matrix[5] = self.zoom;   // Col 1, Y
                matrix[8] = self.pan.x * self.zoom; // Col 2, X
                matrix[9] = -self.pan.y * self.zoom; // Col 2, Y (NDC Y is up)
                matrix[10] = 1.0;        // Col 2, Z

                let settings = ShaderSettings {
                    view_matrix: matrix,
                    grayscale: self.grayscale,
                    grid_opacity: self.grid_opacity,
                    grid_size: self.grid_size,
                    padding: 0.0,
                };

                let callback = egui_wgpu::Callback::new_paint_callback(
                    rect,
                    TessellatorCallback {
                        image: self.pending_high_res.take(),
                        settings,
                        format: self.wgpu_state.target_format,
                    },
                );
                ui.painter().with_clip_rect(rect).add(callback);
            } else {
                ui.heading("Viewer");
                ui.label("Select an image to view.");
            }
        });
    }
}
