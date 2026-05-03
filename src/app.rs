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
use crate::gpu::{CompareUpload, ShaderSettings, TessellatorCallback};
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

#[cfg(target_os = "macos")]
const OPEN_SHORTCUT_LABEL: &str = "Cmd+O";
#[cfg(not(target_os = "macos"))]
const OPEN_SHORTCUT_LABEL: &str = "Ctrl+O";

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

    /// Right-side image for compare mode (None = compare off).
    compare_image: Option<Arc<DecodedImage>>,
    compare_path: Option<PathBuf>,
    compare_divider: f32,
    compare_generation: u64,
    /// Pending compare-slot change to ship with the next paint callback.
    compare_dirty: bool,

    loupe_zoom: f32,
    loupe_radius: f32,
    dither: bool,

    /// Multiplier to apply to the current zoom on the next viewport frame
    /// (set by keyboard zoom-step shortcuts).
    pending_zoom_step: Option<f32>,

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
    compare_divider: Option<f32>,
    loupe_zoom: Option<f32>,
    loupe_radius: Option<f32>,
    dither: Option<bool>,
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
            compare_image: None,
            compare_path: None,
            compare_divider: saved.compare_divider.unwrap_or(0.5),
            compare_generation: 0,
            compare_dirty: false,
            loupe_zoom: saved.loupe_zoom.unwrap_or(4.0),
            loupe_radius: saved.loupe_radius.unwrap_or(120.0),
            dither: saved.dither.unwrap_or(true),
            pending_zoom_step: None,
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
        self.clear_compare();
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

    fn start_compare(&mut self, path: PathBuf, ctx: &egui::Context) {
        self.compare_generation = self.compare_generation.wrapping_add(1);
        self.compare_path = Some(path.clone());
        if let Some(image) = self.image_cache.get(&path) {
            self.compare_image = Some(image);
            self.compare_dirty = true;
        } else {
            self.compare_image = None;
            self.compare_dirty = true;
            self.pending_image_requests.insert(path.clone());
            io::request_image(
                path,
                ImagePurpose::Compare { generation: self.compare_generation },
                self.sender.clone(),
                ctx.clone(),
            );
        }
    }

    fn clear_compare(&mut self) {
        if self.compare_image.is_some() || self.compare_path.is_some() {
            self.compare_dirty = true;
        }
        self.compare_image = None;
        self.compare_path = None;
        // Bump generation so any in-flight Compare result is discarded.
        self.compare_generation = self.compare_generation.wrapping_add(1);
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
                        ImagePurpose::Compare { generation } => {
                            if generation == self.compare_generation
                                && self.compare_path.as_deref() == Some(&path)
                            {
                                self.compare_image = Some(image);
                                self.compare_dirty = true;
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
        let p = read_pressed_keys(ctx);

        // App-wide commands.
        if p.open
            && let Some(path) = rfd::FileDialog::new().pick_folder()
        {
            self.open_folder(path, ctx);
        }
        if p.refresh
            && let Some(folder) = self.folder_path.clone()
        {
            self.restore_selection = self
                .selected_index
                .and_then(|i| self.files.get(i))
                .map(|f| f.path.clone());
            self.open_folder(folder, ctx);
        }
        if p.escape {
            self.clear_compare();
        }

        // View mode.
        if p.fit {
            self.view = ViewState::FitOnNextFrame;
        } else if p.fill {
            self.view = ViewState::FillOnNextFrame;
        } else if p.one_to_one {
            self.view = ViewState::one_to_one();
        }
        if p.zoom_in {
            let acc = self.pending_zoom_step.unwrap_or(1.0) * 1.1;
            self.pending_zoom_step = Some(acc);
        }
        if p.zoom_out {
            let acc = self.pending_zoom_step.unwrap_or(1.0) / 1.1;
            self.pending_zoom_step = Some(acc);
        }

        // Tools.
        if p.toggle_grayscale {
            self.grayscale = if self.grayscale > 0.5 { 0.0 } else { 1.0 };
        }

        // Navigation through file list.
        if self.files.is_empty() {
            return;
        }
        let last = self.files.len() - 1;
        let current = self.selected_index;
        let target = if p.home {
            Some(0)
        } else if p.end {
            Some(last)
        } else if p.left {
            Some(current.map_or(0, |i| i.saturating_sub(1)))
        } else if p.right {
            Some(current.map_or(0, |i| (i + 1).min(last)))
        } else if p.page_up {
            Some(current.map_or(0, |i| i.saturating_sub(10)))
        } else if p.page_down {
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
            compare_divider: Some(self.compare_divider),
            loupe_zoom: Some(self.loupe_zoom),
            loupe_radius: Some(self.loupe_radius),
            dither: Some(self.dither),
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
                    let btn = ui
                        .button("Open Folder...")
                        .on_hover_text(format!("Open folder ({})", OPEN_SHORTCUT_LABEL));
                    if btn.clicked()
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
                ui.separator();
                self.show_compare_controls(ui, ctx);
                ui.separator();
                ui.checkbox(&mut self.dither, "Dither")
                    .on_hover_text("Add 1-bit noise to remove gradient banding");
            });
        });
    }

    fn show_compare_controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        match self.compare_path.clone() {
            None => {
                if ui
                    .button("Compare...")
                    .on_hover_text("Pick a second image to A/B compare")
                    .clicked()
                    && let Some(path) = rfd::FileDialog::new()
                        .add_filter("Images", &["jpg", "jpeg", "png", "webp", "bmp", "tiff"])
                        .pick_file()
                {
                    self.start_compare(path, ctx);
                }
            }
            Some(p) => {
                let name = p.file_name().unwrap_or_default().to_string_lossy().to_string();
                ui.label(format!("vs {}", name));
                if ui.button("X").on_hover_text("Stop comparing").clicked() {
                    self.clear_compare();
                }
                ui.add(
                    egui::Slider::new(&mut self.compare_divider, 0.0..=1.0)
                        .text("Split")
                        .clamping(egui::SliderClamping::Always),
                );
            }
        }
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

            // Keyboard zoom step (=/+/-) - applied around screen center, so
            // pan stays as-is.
            if let Some(factor) = self.pending_zoom_step.take() {
                zoom *= factor;
            }

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
            let mouse_pos = ui.input(|i| i.pointer.hover_pos());

            // Eyedropper: sample the pixel under the cursor.
            self.cursor_pixel = if response.hovered() {
                mouse_pos.and_then(|mouse| {
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

            // Loupe: active when Alt is held and the cursor is over the image.
            let alt_held = ctx.input(|i| i.modifiers.alt);
            let (loupe_active, loupe_center_screen, loupe_center_uv) = if alt_held
                && response.hovered()
                && let Some(m) = mouse_pos
                && let Some(uv) = screen_to_uv(m, rect, scale, pan)
            {
                (1u32, [m.x, m.y], [uv.x, uv.y])
            } else {
                (0u32, [0.0, 0.0], [0.0, 0.0])
            };

            let settings = ShaderSettings {
                view_matrix: view_matrix(scale, pan),
                grayscale: self.grayscale,
                overlay_opacity: self.overlay_opacity,
                grid_size: self.grid_size,
                overlay_mode: self.overlay_mode.shader_id(),
                compare_divider: self.compare_divider,
                compare_active: u32::from(self.compare_image.is_some()),
                loupe_active,
                loupe_zoom: self.loupe_zoom,
                loupe_center_uv,
                loupe_center_screen,
                loupe_radius: self.loupe_radius,
                dither: u32::from(self.dither),
                _pad0: 0.0,
                _pad1: 0.0,
            };

            let upload = if self.needs_upload {
                self.needs_upload = false;
                self.current_image.clone()
            } else {
                None
            };

            let compare_upload = if self.compare_dirty {
                self.compare_dirty = false;
                match &self.compare_image {
                    Some(img) => CompareUpload::Set(img.clone()),
                    None => CompareUpload::Clear,
                }
            } else {
                CompareUpload::NoChange
            };

            let callback = egui_wgpu::Callback::new_paint_callback(
                rect,
                TessellatorCallback {
                    image: upload,
                    compare: compare_upload,
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

/// One-frame snapshot of which shortcuts fired this frame. Centralising the
/// `consume_key` calls here keeps the dispatcher in `handle_keyboard_nav`
/// readable and ensures one consistent set of modifiers.
#[derive(Default)]
struct Pressed {
    // Navigation
    left: bool,
    right: bool,
    home: bool,
    end: bool,
    page_up: bool,
    page_down: bool,
    // App
    open: bool,
    refresh: bool,
    escape: bool,
    // View
    fit: bool,
    fill: bool,
    one_to_one: bool,
    zoom_in: bool,
    zoom_out: bool,
    // Tools
    toggle_grayscale: bool,
}

fn read_pressed_keys(ctx: &egui::Context) -> Pressed {
    use egui::{Key, Modifiers};
    ctx.input_mut(|i| {
        let space = i.consume_key(Modifiers::NONE, Key::Space);
        Pressed {
            left: i.consume_key(Modifiers::NONE, Key::ArrowLeft),
            right: i.consume_key(Modifiers::NONE, Key::ArrowRight) || space,
            home: i.consume_key(Modifiers::NONE, Key::Home),
            end: i.consume_key(Modifiers::NONE, Key::End),
            page_up: i.consume_key(Modifiers::NONE, Key::PageUp),
            page_down: i.consume_key(Modifiers::NONE, Key::PageDown),

            open: i.consume_key(Modifiers::COMMAND, Key::O),
            refresh: i.consume_key(Modifiers::COMMAND, Key::R),
            escape: i.consume_key(Modifiers::NONE, Key::Escape),

            fit: i.consume_key(Modifiers::NONE, Key::F)
                || i.consume_key(Modifiers::NONE, Key::Num0),
            fill: i.consume_key(Modifiers::SHIFT, Key::F),
            one_to_one: i.consume_key(Modifiers::NONE, Key::Num1),
            // Accept "=" (no shift), "+" (shifted "="), and the dedicated Plus
            // key on layouts that have one.
            zoom_in: i.consume_key(Modifiers::NONE, Key::Equals)
                || i.consume_key(Modifiers::SHIFT, Key::Equals)
                || i.consume_key(Modifiers::NONE, Key::Plus),
            zoom_out: i.consume_key(Modifiers::NONE, Key::Minus),

            toggle_grayscale: i.consume_key(Modifiers::NONE, Key::G),
        }
    })
}

fn format_zoom(view: ViewState) -> String {
    match view {
        ViewState::FitOnNextFrame => "Fit".to_string(),
        ViewState::FillOnNextFrame => "Fill".to_string(),
        ViewState::Manual { zoom, .. } => format!("{:.0}%", zoom * 100.0),
    }
}

/// Inverse-transform a screen-space cursor position into image UV coords using
/// the same scale/pan that drives the shader's view matrix. Returns `None` if
/// the cursor is outside the image bounds.
fn screen_to_uv(
    mouse: egui::Pos2,
    rect: egui::Rect,
    scale: egui::Vec2,
    pan: egui::Vec2,
) -> Option<egui::Vec2> {
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
    Some(egui::vec2(u, v))
}

fn sample_pixel_at(
    mouse: egui::Pos2,
    rect: egui::Rect,
    scale: egui::Vec2,
    pan: egui::Vec2,
    image: &DecodedImage,
) -> Option<CursorSample> {
    let uv = screen_to_uv(mouse, rect, scale, pan)?;
    let mip0 = image.mips.first()?;
    let x = ((uv.x * image.width as f32) as u32).min(image.width.saturating_sub(1));
    let y = ((uv.y * image.height as f32) as u32).min(image.height.saturating_sub(1));
    let idx = (y as usize * image.width as usize + x as usize) * 4;
    let rgba = [
        *mip0.rgba.get(idx)?,
        *mip0.rgba.get(idx + 1)?,
        *mip0.rgba.get(idx + 2)?,
        *mip0.rgba.get(idx + 3)?,
    ];
    Some(CursorSample { x, y, rgba })
}
