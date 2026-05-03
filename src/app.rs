// ABOUTME: Main application state and egui update loop for the photo viewer.
// ABOUTME: Owns the file list, view transform, artist tools, and routes I/O messages.

use eframe::egui;
use eframe::wgpu;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};

use crate::cache::ImageCache;
use crate::gpu::{ShaderSettings, TessellatorCallback};
use crate::io::{self, DecodedImage, ImagePurpose, Message};
use crate::view::{view_matrix, ViewState};
use crate::watcher::FolderWatcher;

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

/// Window of inactivity after a watcher event before triggering a re-scan.
const FOLDER_REFRESH_DEBOUNCE: Duration = Duration::from_millis(500);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OverlayMode {
    None,
    Grid,
    RuleOfThirds,
    GoldenRatio,
    Diagonal,
}

impl OverlayMode {
    fn label(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Grid => "Grid",
            Self::RuleOfThirds => "Rule of Thirds",
            Self::GoldenRatio => "Golden Ratio",
            Self::Diagonal => "Diagonal",
        }
    }

    /// Discriminant used by the WGSL shader (must match the `switch` cases).
    fn shader_id(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Grid => 1,
            Self::RuleOfThirds => 2,
            Self::GoldenRatio => 3,
            Self::Diagonal => 4,
        }
    }
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
    overlay_opacity: f32,
    grid_size: f32,
    overlay_mode: OverlayMode,
    view: ViewState,

    recursion_depth: usize,

    /// Currently displayed image (kept for eyedropper sampling). `None` means
    /// no image is shown.
    current_image: Option<Arc<DecodedImage>>,
    /// `current_image` has changed; the next paint callback should re-upload.
    needs_upload: bool,
    current_image_size: Option<[u32; 2]>,

    /// Pixel under the cursor in the viewport (image coords + RGBA), updated
    /// each frame. Drives the eyedropper readout in the status bar.
    cursor_pixel: Option<CursorSample>,

    /// Folder filesystem watcher; replaced when the folder changes.
    folder_watcher: Option<FolderWatcher>,
    /// Deadline for the next debounced folder rescan.
    folder_refresh_at: Option<Instant>,
    /// On the next FilesFound, restore selection to this path if present.
    restore_selection: Option<PathBuf>,
}

#[derive(Clone, Copy)]
struct CursorSample {
    x: u32,
    y: u32,
    rgba: [u8; 4],
}

#[derive(Default, Serialize, Deserialize)]
struct PersistentState {
    folder_path: Option<PathBuf>,
    recursion_depth: Option<usize>,
    overlay_mode: Option<OverlayMode>,
    overlay_opacity: Option<f32>,
    grid_size: Option<f32>,
    grayscale: Option<f32>,
}

impl TessellatorApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let render_state = cc
            .wgpu_render_state
            .as_ref()
            .expect("WGPU render state not available - eframe must be built with the wgpu renderer");
        let target_format = render_state.target_format;
        let (sender, receiver) = crossbeam_channel::unbounded();

        let saved: PersistentState = cc
            .storage
            .and_then(|s| eframe::get_value(s, eframe::APP_KEY))
            .unwrap_or_default();

        let mut app = Self {
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
            grayscale: saved.grayscale.unwrap_or(0.0),
            overlay_opacity: saved.overlay_opacity.unwrap_or(0.5),
            grid_size: saved.grid_size.unwrap_or(10.0),
            overlay_mode: saved.overlay_mode.unwrap_or(OverlayMode::None),
            view: ViewState::identity(),
            recursion_depth: saved.recursion_depth.unwrap_or(DEFAULT_RECURSION_DEPTH),
            current_image: None,
            needs_upload: false,
            current_image_size: None,
            cursor_pixel: None,
            folder_watcher: None,
            folder_refresh_at: None,
            restore_selection: None,
        };

        if let Some(path) = saved.folder_path
            && path.is_dir()
        {
            app.open_folder(path, &cc.egui_ctx);
        }

        app
    }

    fn open_folder(&mut self, path: PathBuf, ctx: &egui::Context) {
        self.folder_path = Some(path.clone());
        self.files.clear();
        self.selected_index = None;
        self.requested_thumbnails.clear();
        self.pending_image_requests.clear();
        self.current_image_size = None;
        self.current_image = None;
        self.needs_upload = false;
        self.cursor_pixel = None;
        io::scan_folder(
            path.clone(),
            self.recursion_depth,
            self.sender.clone(),
            ctx.clone(),
        );

        // (Re)attach the watcher. Drop the old one first so we don't leak a
        // background thread on each folder change.
        self.folder_watcher = None;
        match FolderWatcher::new(
            &path,
            self.recursion_depth > 1,
            self.sender.clone(),
            ctx.clone(),
        ) {
            Ok(w) => self.folder_watcher = Some(w),
            Err(e) => log::warn!("Failed to start folder watcher for {:?}: {}", path, e),
        }
    }

    /// Display the given image in the viewport (and reset to fit-to-screen).
    fn show_image(&mut self, image: Arc<DecodedImage>) {
        self.current_image_size = Some([image.width, image.height]);
        self.current_image = Some(image);
        self.needs_upload = true;
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
                    if let Some(target) = self.restore_selection.take()
                        && let Some(i) = self.files.iter().position(|f| f.path == target)
                    {
                        self.selected_index = Some(i);
                        self.pending_scroll_to_index = Some(i);
                    }
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
                            log::debug!("High-res ready: {}x{}", image.width, image.height);
                            self.show_image(image);
                        }
                        ImagePurpose::Preload => {
                            // Cached above; if the user has since navigated to
                            // this image, surface it now.
                            if let Some(i) = self.selected_index
                                && self.files.get(i).map(|f| &f.path) == Some(&path)
                                && self.current_image_size.is_none_or(|sz| {
                                    sz != [image.width, image.height]
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
                Message::FolderChanged => {
                    let deadline = Instant::now() + FOLDER_REFRESH_DEBOUNCE;
                    self.folder_refresh_at = Some(deadline);
                    ctx.request_repaint_after(FOLDER_REFRESH_DEBOUNCE + Duration::from_millis(50));
                }
            }
        }
    }

    /// Trigger a folder rescan once watcher events have quieted.
    fn process_folder_refresh(&mut self, ctx: &egui::Context) {
        let Some(deadline) = self.folder_refresh_at else { return };
        if Instant::now() < deadline {
            return;
        }
        self.folder_refresh_at = None;
        let Some(folder) = self.folder_path.clone() else { return };
        self.restore_selection = self
            .selected_index
            .and_then(|i| self.files.get(i))
            .map(|f| f.path.clone());
        log::info!("Watcher triggered re-scan of {:?}", folder);
        self.open_folder(folder, ctx);
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
        self.process_folder_refresh(ctx);
        self.handle_dropped_files(ctx);
        self.handle_keyboard_nav(ctx);
        self.show_sidebar(ctx);
        self.show_tools_panel(ctx);
        self.show_status_bar(ctx);
        self.show_viewport(ctx);
        self.show_drop_overlay(ctx);
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        let state = PersistentState {
            folder_path: self.folder_path.clone(),
            recursion_depth: Some(self.recursion_depth),
            overlay_mode: Some(self.overlay_mode),
            overlay_opacity: Some(self.overlay_opacity),
            grid_size: Some(self.grid_size),
            grayscale: Some(self.grayscale),
        };
        eframe::set_value(storage, eframe::APP_KEY, &state);
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
                ui.label("Overlay:");
                egui::ComboBox::from_id_salt("overlay_mode")
                    .selected_text(self.overlay_mode.label())
                    .show_ui(ui, |ui| {
                        for mode in [
                            OverlayMode::None,
                            OverlayMode::Grid,
                            OverlayMode::RuleOfThirds,
                            OverlayMode::GoldenRatio,
                            OverlayMode::Diagonal,
                        ] {
                            ui.selectable_value(&mut self.overlay_mode, mode, mode.label());
                        }
                    });
                if self.overlay_mode != OverlayMode::None {
                    ui.label("Opacity:");
                    ui.add(egui::Slider::new(&mut self.overlay_opacity, 0.0..=1.0));
                    if self.overlay_mode == OverlayMode::Grid {
                        ui.label("Size:");
                        ui.add(egui::Slider::new(&mut self.grid_size, 1.0..=50.0));
                    }
                }
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
                if let Some(s) = self.cursor_pixel {
                    ui.separator();
                    let [r, g, b, _a] = s.rgba;
                    let swatch_size = egui::vec2(14.0, 14.0);
                    let (rect, _) = ui.allocate_exact_size(swatch_size, egui::Sense::hover());
                    ui.painter().rect_filled(
                        rect,
                        2.0,
                        egui::Color32::from_rgb(r, g, b),
                    );
                    ui.label(format!(
                        "({}, {})  rgb({}, {}, {})  #{:02X}{:02X}{:02X}",
                        s.x, s.y, r, g, b, r, g, b
                    ));
                }
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

            // Eyedropper: sample the pixel under the cursor.
            self.cursor_pixel = if response.hovered() {
                ui.input(|i| i.pointer.hover_pos()).and_then(|mouse| {
                    sample_pixel_at(
                        mouse,
                        rect,
                        scale,
                        pan,
                        self.current_image.as_deref()?,
                    )
                })
            } else {
                None
            };

            let settings = ShaderSettings {
                view_matrix: view_matrix(scale, pan),
                grayscale: self.grayscale,
                overlay_opacity: self.overlay_opacity,
                grid_size: self.grid_size,
                overlay_mode: self.overlay_mode.shader_id(),
            };

            let upload = if self.needs_upload {
                self.needs_upload = false;
                self.current_image.clone()
            } else {
                None
            };

            let callback = egui_wgpu::Callback::new_paint_callback(
                rect,
                TessellatorCallback {
                    image: upload,
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

/// Inverse-transform a screen-space cursor position into image pixel coords
/// using the same scale/pan that drives the shader's view matrix, then sample
/// mip 0. Returns `None` if the cursor is outside the image bounds.
fn sample_pixel_at(
    mouse: egui::Pos2,
    rect: egui::Rect,
    scale: egui::Vec2,
    pan: egui::Vec2,
    image: &DecodedImage,
) -> Option<CursorSample> {
    let center = rect.center();
    // Mouse → NDC' (screen-space NDC; egui Y is down so negate to get NDC up).
    let ndc_x = (mouse.x - center.x) / (rect.width() * 0.5);
    let ndc_y = -(mouse.y - center.y) / (rect.height() * 0.5);
    // Inverse of view_matrix: matrix applies (scale * v) + (pan.x, -pan.y).
    let vx = (ndc_x - pan.x) / scale.x;
    let vy = (ndc_y + pan.y) / scale.y;
    // Quad vertex NDC [-1, 1] → tex coords [0, 1] (Y flipped).
    let u = (vx + 1.0) * 0.5;
    let v = (1.0 - vy) * 0.5;
    if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
        return None;
    }
    let mip0 = image.mips.first()?;
    let x = ((u * image.width as f32) as u32).min(image.width.saturating_sub(1));
    let y = ((v * image.height as f32) as u32).min(image.height.saturating_sub(1));
    let idx = (y as usize * image.width as usize + x as usize) * 4;
    let rgba = [
        *mip0.rgba.get(idx)?,
        *mip0.rgba.get(idx + 1)?,
        *mip0.rgba.get(idx + 2)?,
        *mip0.rgba.get(idx + 3)?,
    ];
    Some(CursorSample { x, y, rgba })
}
