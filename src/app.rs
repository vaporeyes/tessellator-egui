// ABOUTME: Main application state and egui update loop for the photo viewer.
// ABOUTME: Owns the file list, view transform, artist tools, and routes I/O messages.

use eframe::egui;
use eframe::wgpu;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender};

use crate::gpu::{ShaderSettings, TessellatorCallback};
use crate::io::{self, Message};
use crate::view::{view_matrix, ViewState};

pub struct FileEntry {
    pub path: PathBuf,
    pub name: String,
    pub thumbnail: Option<egui::TextureHandle>,
}

pub struct TessellatorApp {
    folder_path: Option<PathBuf>,
    files: Vec<FileEntry>,
    selected_index: Option<usize>,

    receiver: Receiver<Message>,
    sender: Sender<Message>,

    /// Paths we have already issued a thumbnail decode for, regardless of
    /// outcome. Prevents repeated retries of broken files.
    requested_thumbnails: HashSet<PathBuf>,

    /// Monotonic counter incremented on every high-res request. Decoded results
    /// arriving with a stale generation are discarded.
    high_res_generation: u64,
    last_load_error: Option<String>,

    target_format: wgpu::TextureFormat,

    grayscale: f32,
    grid_opacity: f32,
    grid_size: f32,
    view: ViewState,

    /// Image waiting to be uploaded by the next paint callback.
    pending_high_res: Option<Arc<image::DynamicImage>>,
    current_image_size: Option<[u32; 2]>,
}

impl TessellatorApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let render_state = cc
            .wgpu_render_state
            .as_ref()
            .expect("WGPU render state not available - eframe must be built with the wgpu renderer");
        let target_format = render_state.target_format;
        let (sender, receiver) = crossbeam_channel::unbounded();

        Self {
            folder_path: None,
            files: Vec::new(),
            selected_index: None,
            receiver,
            sender,
            requested_thumbnails: HashSet::new(),
            high_res_generation: 0,
            last_load_error: None,
            target_format,
            grayscale: 0.0,
            grid_opacity: 0.0,
            grid_size: 10.0,
            view: ViewState::identity(),
            pending_high_res: None,
            current_image_size: None,
        }
    }

    fn select_image(&mut self, index: usize, ctx: &egui::Context) {
        let path = self.files[index].path.clone();
        log::info!("Selected: {}", self.files[index].name);
        self.selected_index = Some(index);
        self.high_res_generation = self.high_res_generation.wrapping_add(1);
        self.last_load_error = None;
        io::request_high_res(
            path,
            self.high_res_generation,
            self.sender.clone(),
            ctx.clone(),
        );
    }

    fn drain_messages(&mut self, ctx: &egui::Context) {
        while let Ok(msg) = self.receiver.try_recv() {
            match msg {
                Message::FilesFound(paths) => {
                    self.files = paths
                        .into_iter()
                        .map(|p| FileEntry {
                            name: p.file_name().unwrap_or_default().to_string_lossy().to_string(),
                            path: p,
                            thumbnail: None,
                        })
                        .collect();
                }
                Message::ThumbnailLoaded { path, image } => {
                    if let Some(entry) = self.files.iter_mut().find(|f| f.path == path) {
                        let handle = ctx.load_texture(
                            path.to_string_lossy(),
                            image,
                            egui::TextureOptions::default(),
                        );
                        entry.thumbnail = Some(handle);
                    }
                }
                Message::ThumbnailFailed => {
                    // Path stays in requested_thumbnails so we don't retry.
                }
                Message::HighResLoaded { generation, image } => {
                    if generation != self.high_res_generation {
                        log::debug!(
                            "Discarding stale high-res result (gen {} != {})",
                            generation,
                            self.high_res_generation
                        );
                        continue;
                    }
                    log::debug!("High-res ready: {}x{}", image.width(), image.height());
                    self.current_image_size = Some([image.width(), image.height()]);
                    self.pending_high_res = Some(image);
                    self.view = ViewState::FitOnNextFrame;
                }
                Message::HighResFailed { generation, path, error } => {
                    if generation != self.high_res_generation {
                        continue;
                    }
                    self.last_load_error = Some(format!(
                        "Failed to load {}: {}",
                        path.file_name().unwrap_or_default().to_string_lossy(),
                        error
                    ));
                }
            }
        }
    }
}

impl eframe::App for TessellatorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_messages(ctx);
        self.show_sidebar(ctx);
        self.show_tools_panel(ctx);
        self.show_viewport(ctx);
    }
}

impl TessellatorApp {
    fn show_sidebar(&mut self, ctx: &egui::Context) {
        let mut clicked: Option<usize> = None;
        let mut to_request: Vec<PathBuf> = Vec::new();
        let mut open_folder: Option<PathBuf> = None;

        egui::SidePanel::left("sidebar")
            .resizable(true)
            .default_width(250.0)
            .show(ctx, |ui| {
                ui.heading("Tessellator");
                if ui.button("Open Folder...").clicked()
                    && let Some(path) = rfd::FileDialog::new().pick_folder()
                {
                    open_folder = Some(path);
                }
                ui.separator();

                let row_height = 40.0;
                egui::ScrollArea::vertical().show_rows(
                    ui,
                    row_height,
                    self.files.len(),
                    |ui, row_range| {
                        for i in row_range.clone() {
                            let entry = &self.files[i];
                            let is_selected = self.selected_index == Some(i);
                            let response = ui
                                .horizontal(|ui| {
                                    if let Some(texture) = &entry.thumbnail {
                                        ui.image((texture.id(), egui::vec2(32.0, 32.0)));
                                    } else {
                                        ui.allocate_space(egui::vec2(32.0, 32.0));
                                    }
                                    ui.selectable_label(is_selected, &entry.name)
                                })
                                .inner;
                            if response.clicked() {
                                clicked = Some(i);
                            }
                        }
                        // Only request thumbnails for currently-visible rows.
                        for i in row_range {
                            let entry = &self.files[i];
                            if entry.thumbnail.is_none()
                                && !self.requested_thumbnails.contains(&entry.path)
                            {
                                to_request.push(entry.path.clone());
                            }
                        }
                    },
                );
            });

        if let Some(path) = open_folder {
            self.folder_path = Some(path.clone());
            self.files.clear();
            self.selected_index = None;
            self.requested_thumbnails.clear();
            io::scan_folder(path, self.sender.clone(), ctx.clone());
        }

        for path in to_request {
            self.requested_thumbnails.insert(path.clone());
            io::request_thumbnail(path, self.sender.clone(), ctx.clone());
        }

        if let Some(i) = clicked {
            self.select_image(i, ctx);
        }
    }

    fn show_tools_panel(&mut self, ctx: &egui::Context) {
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
                    self.view = ViewState::identity();
                }
            });
        });
    }

    fn show_viewport(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(err) = &self.last_load_error {
                ui.colored_label(egui::Color32::LIGHT_RED, err);
            }

            if self.selected_index.is_none() {
                ui.heading("Viewer");
                ui.label("Select an image to view.");
                return;
            }

            let Some([img_w, img_h]) = self.current_image_size else {
                return;
            };
            let img_w = img_w as f32;
            let img_h = img_h as f32;

            let rect = ui.max_rect();
            let response = ui.interact(rect, ui.id(), egui::Sense::drag());
            let screen = rect.size();

            // Resolve fit-to-screen now that we know the viewport size.
            if matches!(self.view, ViewState::FitOnNextFrame) {
                let zoom = (screen.x / img_w).min(screen.y / img_h);
                self.view = ViewState::Manual { zoom, pan: egui::Vec2::ZERO };
            }

            let ViewState::Manual { mut zoom, mut pan } = self.view else {
                return;
            };

            // Only consume scroll when the pointer is over the viewport so the
            // file list keeps its own scroll input.
            if response.hovered() {
                let scroll_delta = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll_delta != 0.0 {
                    let zoom_factor = (scroll_delta * 0.01).exp();
                    if let Some(mouse_pos) = ui.input(|i| i.pointer.hover_pos()) {
                        let mouse_ndc = egui::vec2(
                            (mouse_pos.x - rect.center().x) / (rect.width() * 0.5),
                            (mouse_pos.y - rect.center().y) / (rect.height() * 0.5),
                        );
                        // Pan_new = Mouse_ndc + (Pan_old - Mouse_ndc) * zoom_factor
                        pan = mouse_ndc + (pan - mouse_ndc) * zoom_factor;
                    }
                    zoom *= zoom_factor;
                }
            }

            if response.dragged() {
                pan += response.drag_delta() / (screen * 0.5);
            }

            self.view = ViewState::Manual { zoom, pan };

            let scale = egui::vec2((img_w * zoom) / screen.x, (img_h * zoom) / screen.y);
            let settings = ShaderSettings {
                view_matrix: view_matrix(scale, pan),
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
                    format: self.target_format,
                },
            );
            ui.painter().with_clip_rect(rect).add(callback);
        });
    }
}
