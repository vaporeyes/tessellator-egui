// ABOUTME: Main application state and egui update loop for the photo viewer.
// ABOUTME: Owns the file list, view transform, artist tools, and routes I/O messages.

use eframe::egui;
use eframe::wgpu;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender};

use crate::cache::ImageCache;
use crate::gpu::{ShaderSettings, TessellatorCallback};
use crate::io::{self, ImagePurpose, Message};
use crate::view::{view_matrix, ViewState};

/// Maximum bytes of decoded image data held in the LRU cache (~512 MB).
const CACHE_CAP_BYTES: usize = 512 * 1024 * 1024;

pub struct FileEntry {
    pub path: PathBuf,
    pub name: String,
    pub size_bytes: u64,
    pub thumbnail: Option<egui::TextureHandle>,
}

/// Default folder-scan recursion depth. 1 = the folder itself only.
const DEFAULT_RECURSION_DEPTH: usize = 2;

pub struct TessellatorApp {
    folder_path: Option<PathBuf>,
    files: Vec<FileEntry>,
    selected_index: Option<usize>,

    receiver: Receiver<Message>,
    sender: Sender<Message>,

    /// Paths we have already issued a thumbnail decode for, regardless of
    /// outcome. Prevents repeated retries of broken files.
    requested_thumbnails: HashSet<PathBuf>,

    /// Paths with a high-res decode currently in flight. Used to avoid
    /// duplicate Preload requests; Display requests are issued unconditionally
    /// and disambiguated by generation.
    pending_image_requests: HashSet<PathBuf>,

    /// Decoded images cached for instant re-display on neighbor navigation.
    image_cache: ImageCache,

    /// Monotonic counter incremented on every Display request. Decoded results
    /// arriving with a stale generation are not shown (but are still cached).
    high_res_generation: u64,
    last_load_error: Option<String>,

    /// Sidebar should scroll to center this row index on the next frame.
    pending_scroll_to_index: Option<usize>,

    target_format: wgpu::TextureFormat,

    grayscale: f32,
    grid_opacity: f32,
    grid_size: f32,
    view: ViewState,

    recursion_depth: usize,

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
            pending_image_requests: HashSet::new(),
            image_cache: ImageCache::new(CACHE_CAP_BYTES),
            high_res_generation: 0,
            last_load_error: None,
            pending_scroll_to_index: None,
            target_format,
            grayscale: 0.0,
            grid_opacity: 0.0,
            grid_size: 10.0,
            view: ViewState::identity(),
            recursion_depth: DEFAULT_RECURSION_DEPTH,
            pending_high_res: None,
            current_image_size: None,
        }
    }

    fn open_folder(&mut self, path: PathBuf, ctx: &egui::Context) {
        self.folder_path = Some(path.clone());
        self.files.clear();
        self.selected_index = None;
        self.requested_thumbnails.clear();
        self.pending_image_requests.clear();
        self.current_image_size = None;
        self.pending_high_res = None;
        io::scan_folder(path, self.recursion_depth, self.sender.clone(), ctx.clone());
    }

    /// Display the given image in the viewport (and reset to fit-to-screen).
    fn show_image(&mut self, image: Arc<image::DynamicImage>) {
        self.current_image_size = Some([image.width(), image.height()]);
        self.pending_high_res = Some(image);
        self.view = ViewState::FitOnNextFrame;
    }

    fn select_image(&mut self, index: usize, ctx: &egui::Context) {
        if index >= self.files.len() {
            return;
        }
        let path = self.files[index].path.clone();
        log::info!("Selected: {}", self.files[index].name);
        self.selected_index = Some(index);
        self.high_res_generation = self.high_res_generation.wrapping_add(1);
        self.last_load_error = None;
        self.pending_scroll_to_index = Some(index);

        if let Some(image) = self.image_cache.get(&path) {
            log::debug!("Cache hit: {:?}", path.file_name().unwrap_or_default());
            self.show_image(image);
        } else {
            self.pending_image_requests.insert(path.clone());
            io::request_image(
                path,
                ImagePurpose::Display { generation: self.high_res_generation },
                self.sender.clone(),
                ctx.clone(),
            );
        }

        self.preload_neighbors(ctx);
    }

    /// Issue Preload requests for the images immediately before and after the
    /// current selection, skipping anything already cached or in flight.
    fn preload_neighbors(&mut self, ctx: &egui::Context) {
        let Some(current) = self.selected_index else { return };
        let neighbors = [current.checked_sub(1), Some(current + 1).filter(|&i| i < self.files.len())];
        for n in neighbors.into_iter().flatten() {
            let path = self.files[n].path.clone();
            if self.image_cache.contains(&path) || self.pending_image_requests.contains(&path) {
                continue;
            }
            self.pending_image_requests.insert(path.clone());
            io::request_image(path, ImagePurpose::Preload, self.sender.clone(), ctx.clone());
        }
    }

    fn drain_messages(&mut self, ctx: &egui::Context) {
        while let Ok(msg) = self.receiver.try_recv() {
            match msg {
                Message::FilesFound(scanned) => {
                    self.files = scanned
                        .into_iter()
                        .map(|s| FileEntry {
                            name: s
                                .path
                                .file_name()
                                .unwrap_or_default()
                                .to_string_lossy()
                                .to_string(),
                            path: s.path,
                            size_bytes: s.size_bytes,
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
                Message::ImageDecoded { path, image, purpose } => {
                    self.pending_image_requests.remove(&path);
                    self.image_cache.insert(path.clone(), image.clone());
                    match purpose {
                        ImagePurpose::Display { generation } => {
                            if generation != self.high_res_generation {
                                log::debug!(
                                    "Discarding stale Display result (gen {} != {})",
                                    generation,
                                    self.high_res_generation
                                );
                                continue;
                            }
                            log::debug!("High-res ready: {}x{}", image.width(), image.height());
                            self.show_image(image);
                        }
                        ImagePurpose::Preload => {
                            // Cached above; if the user has since navigated to
                            // this image, surface it now.
                            if let Some(i) = self.selected_index
                                && self.files.get(i).map(|f| &f.path) == Some(&path)
                                && self.current_image_size.is_none_or(|sz| {
                                    sz != [image.width(), image.height()]
                                })
                            {
                                self.show_image(image);
                            }
                        }
                    }
                }
                Message::ImageFailed { path, error, purpose } => {
                    self.pending_image_requests.remove(&path);
                    if let ImagePurpose::Display { generation } = purpose {
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

    /// Read keyboard navigation and view-mode keys and act on them. Uses
    /// `consume_key` so a focused widget doesn't also see the press.
    fn handle_keyboard_nav(&mut self, ctx: &egui::Context) {
        let (left, right, home, end, page_up, page_down, space, fit, fill, one_to_one) =
            ctx.input_mut(|i| {
                (
                    i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowLeft),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowRight),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::Home),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::End),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::PageUp),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::PageDown),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::Space),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::F),
                    i.consume_key(egui::Modifiers::SHIFT, egui::Key::F),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::Num1),
                )
            });

        if fit {
            self.view = ViewState::FitOnNextFrame;
        } else if fill {
            self.view = ViewState::FillOnNextFrame;
        } else if one_to_one {
            self.view = ViewState::one_to_one();
        }

        if self.files.is_empty() {
            return;
        }
        let last = self.files.len() - 1;
        let current = self.selected_index;

        let target = if home {
            Some(0)
        } else if end {
            Some(last)
        } else if left {
            Some(current.map_or(0, |i| i.saturating_sub(1)))
        } else if right || space {
            Some(current.map_or(0, |i| (i + 1).min(last)))
        } else if page_up {
            Some(current.map_or(0, |i| i.saturating_sub(10)))
        } else if page_down {
            Some(current.map_or(0, |i| (i + 10).min(last)))
        } else {
            None
        };

        if let Some(i) = target
            && Some(i) != current
        {
            self.select_image(i, ctx);
        }
    }

    /// Resolve any folders or files dropped onto the window. A dropped folder
    /// is opened as-is; a dropped file opens its parent directory.
    fn handle_dropped_files(&mut self, ctx: &egui::Context) {
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        if dropped.is_empty() {
            return;
        }
        for file in dropped {
            let Some(path) = file.path else { continue };
            let folder = if path.is_dir() {
                Some(path)
            } else if path.is_file() {
                path.parent().map(|p| p.to_path_buf())
            } else {
                None
            };
            if let Some(folder) = folder {
                self.open_folder(folder, ctx);
                break;
            }
        }
    }
}

impl eframe::App for TessellatorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_messages(ctx);
        self.handle_dropped_files(ctx);
        self.handle_keyboard_nav(ctx);
        self.show_sidebar(ctx);
        self.show_tools_panel(ctx);
        self.show_status_bar(ctx);
        self.show_viewport(ctx);
        self.show_drop_overlay(ctx);
    }
}

impl TessellatorApp {
    fn show_sidebar(&mut self, ctx: &egui::Context) {
        let mut clicked: Option<usize> = None;
        let mut to_request: Vec<PathBuf> = Vec::new();
        let mut open_folder: Option<PathBuf> = None;
        let mut depth_changed = false;
        let pending_scroll = self.pending_scroll_to_index.take();

        egui::SidePanel::left("sidebar")
            .resizable(true)
            .default_width(250.0)
            .show(ctx, |ui| {
                ui.heading("Tessellator");
                ui.horizontal(|ui| {
                    if ui.button("Open Folder...").clicked()
                        && let Some(path) = rfd::FileDialog::new().pick_folder()
                    {
                        open_folder = Some(path);
                    }
                    ui.label("Depth:");
                    let r = ui.add(
                        egui::DragValue::new(&mut self.recursion_depth)
                            .range(1..=16)
                            .speed(0.1),
                    );
                    if r.changed() {
                        depth_changed = true;
                    }
                });
                ui.separator();

                let row_height = 40.0;
                let viewport_h = ui.available_height();

                let mut scroll = egui::ScrollArea::vertical();
                if let Some(i) = pending_scroll {
                    let target = (i as f32) * row_height
                        - viewport_h * 0.5
                        + row_height * 0.5;
                    scroll = scroll.vertical_scroll_offset(target.max(0.0));
                }

                scroll.show_rows(ui, row_height, self.files.len(), |ui, row_range| {
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
                });
            });

        if let Some(path) = open_folder {
            self.open_folder(path, ctx);
        } else if depth_changed && let Some(path) = self.folder_path.clone() {
            // Re-scan with the new depth.
            self.open_folder(path, ctx);
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
                if ui.button("Fit").on_hover_text("Fit image to viewport (F)").clicked() {
                    self.view = ViewState::FitOnNextFrame;
                }
                if ui.button("Fill").on_hover_text("Fill viewport, may crop (Shift+F)").clicked() {
                    self.view = ViewState::FillOnNextFrame;
                }
                if ui.button("100%").on_hover_text("Display at native pixel size (1)").clicked() {
                    self.view = ViewState::one_to_one();
                }
            });
        });
    }

    fn show_status_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let entry = self.selected_index.and_then(|i| self.files.get(i));
                match entry {
                    Some(entry) => {
                        ui.label(&entry.name);
                        ui.separator();
                        if let Some([w, h]) = self.current_image_size {
                            ui.label(format!("{} x {}", w, h));
                            ui.separator();
                        }
                        ui.label(format_bytes(entry.size_bytes));
                        ui.separator();
                    }
                    None => {
                        ui.label(format!("{} images", self.files.len()));
                        ui.separator();
                    }
                }
                ui.label(format!("Zoom: {}", format_zoom(self.view)));
                if let Some(folder) = &self.folder_path {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(folder.to_string_lossy());
                    });
                }
            });
        });
    }

    fn show_drop_overlay(&mut self, ctx: &egui::Context) {
        let hovering = ctx.input(|i| !i.raw.hovered_files.is_empty());
        if !hovering {
            return;
        }
        let screen = ctx.screen_rect();
        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("drop_overlay"),
        ));
        painter.rect_filled(screen, 0.0, egui::Color32::from_black_alpha(160));
        painter.text(
            screen.center(),
            egui::Align2::CENTER_CENTER,
            "Drop folder to open",
            egui::FontId::proportional(32.0),
            egui::Color32::WHITE,
        );
    }

    fn show_viewport(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(err) = &self.last_load_error {
                ui.colored_label(egui::Color32::LIGHT_RED, err);
            }

            if self.selected_index.is_none() {
                ui.heading("Viewer");
                ui.label("Select an image to view, or use the arrow keys to navigate.");
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

            match self.view {
                ViewState::FitOnNextFrame => {
                    let zoom = (screen.x / img_w).min(screen.y / img_h);
                    self.view = ViewState::Manual { zoom, pan: egui::Vec2::ZERO };
                }
                ViewState::FillOnNextFrame => {
                    let zoom = (screen.x / img_w).max(screen.y / img_h);
                    self.view = ViewState::Manual { zoom, pan: egui::Vec2::ZERO };
                }
                ViewState::Manual { .. } => {}
            }

            let ViewState::Manual { mut zoom, mut pan } = self.view else {
                return;
            };

            if response.hovered() {
                let scroll_delta = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll_delta != 0.0 {
                    let zoom_factor = (scroll_delta * 0.01).exp();
                    if let Some(mouse_pos) = ui.input(|i| i.pointer.hover_pos()) {
                        let mouse_ndc = egui::vec2(
                            (mouse_pos.x - rect.center().x) / (rect.width() * 0.5),
                            (mouse_pos.y - rect.center().y) / (rect.height() * 0.5),
                        );
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

fn format_bytes(n: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut value = n as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} {}", n, UNITS[0])
    } else {
        format!("{:.1} {}", value, UNITS[unit])
    }
}

fn format_zoom(view: ViewState) -> String {
    match view {
        ViewState::FitOnNextFrame => "Fit".to_string(),
        ViewState::FillOnNextFrame => "Fill".to_string(),
        ViewState::Manual { zoom, .. } => format!("{:.0}%", zoom * 100.0),
    }
}
