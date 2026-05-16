// ABOUTME: Main application state and egui update loop for the photo viewer.
// ABOUTME: Owns the file list, view transform, artist tools, and routes I/O messages.

use eframe::egui;
use eframe::wgpu;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};

use crate::annotation::AnnotationLayer;
use crate::cache::ImageCache;
use crate::gpu::{
    AnnotationUpload, CompareUpload, GridUpload, ShaderSettings, TessellatorCallback,
};
use crate::io::{self, CancelToken, DecodedImage, ImagePurpose, Message};
use crate::pinboard::{self, DragState, Pinboard, PinnedItem};
use crate::view::{ViewState, view_matrix};
use crate::watcher::FolderWatcher;

/// Dynamic cache size: 25% of total system RAM, but at least 256MB and at most 8GB.
fn compute_cache_cap() -> usize {
    let total = io::get_total_memory();
    let quarter = total / 4;
    quarter.clamp(256 * 1024 * 1024, 8192 * 1024 * 1024)
}

pub struct FileEntry {
    pub path: PathBuf,
    pub name: String,
    pub size_bytes: u64,
    pub thumbnail: Option<egui::TextureHandle>,
    /// User-marked favourite. Persisted as a sidecar `<image>.tess.json`
    /// next to the file (presence == starred, for now).
    pub starred: bool,
    /// Source-image dimensions, populated when the thumbnail decode
    /// completes. None until the row has been thumbnailed.
    pub source_dims: Option<(u32, u32)>,
}

/// Sidecar path for star/tag metadata. Designed to coexist with the annotation
/// sidecar (.tess.png) under a shared `.tess.*` namespace. Future fields
/// (color labels, tags, comments) extend the JSON without breaking the format.
fn star_sidecar_path(image_path: &Path) -> PathBuf {
    let mut s = image_path.as_os_str().to_owned();
    s.push(".tess.json");
    PathBuf::from(s)
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
pub enum CropRatio {
    None,
    Square,
    FourFive,
    SixteenNine,
    Golden,
}

impl CropRatio {
    fn label(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Square => "Square (1:1)",
            Self::FourFive => "4:5",
            Self::SixteenNine => "16:9",
            Self::Golden => "Golden (1.618:1)",
        }
    }

    /// Aspect ratio as width / height. `None` means no crop overlay.
    fn aspect(self) -> Option<f32> {
        match self {
            Self::None => None,
            Self::Square => Some(1.0),
            Self::FourFive => Some(4.0 / 5.0),
            Self::SixteenNine => Some(16.0 / 9.0),
            Self::Golden => Some(1.6180339),
        }
    }
}

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

    /// Paths with a high-res decode currently in flight, mapped to a
    /// cancellation handle. Used to avoid duplicate work and to abort decodes
    /// the user has navigated past.
    pending_image_requests: HashMap<PathBuf, CancelToken>,

    /// Wall-clock of the last `select_image` call, for direction tracking.
    last_select_time: Option<Instant>,
    /// Signed counter of consecutive same-direction selections within the
    /// debounce window. Positive = forward streak, negative = backward,
    /// zero = stationary or recently paused.
    direction_streak: i32,

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
    /// Path that the currently-uploaded `current_image` was decoded from.
    /// Used to detect "selection has moved past the displayed image" so
    /// the viewport can show a spinner instead of stale content.
    current_image_path: Option<PathBuf>,
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

    /// Mirror the displayed image horizontally. Toggled with `H`. Affects
    /// view, eyedropper, loupe, and crop overlay coherently.
    flip_h: bool,
    /// Posterize luma into N bands ("value study mode"). Overrides grayscale
    /// while active. Toggled with `V`.
    value_study: bool,
    /// Number of value bands when `value_study` is on. Slider range 2..=8.
    value_levels: u32,
    /// Crop preview overlay aspect ratio. `None` disables.
    crop_ratio: CropRatio,

    /// Annotation mode active: drag paints instead of pans.
    annotating: bool,
    /// Active brush color (RGB; alpha is taken from the image's pen alpha
    /// which is always 255 for opaque pen).
    brush_color: [f32; 3],
    /// Brush radius in image pixels.
    brush_radius: f32,
    /// Eraser mode toggle (within annotation mode).
    erasing: bool,
    /// Annotation layer for the currently-displayed image. `None` until the
    /// user paints something or a sidecar is found on selection.
    annotation: Option<AnnotationLayer>,
    /// Path of the image whose annotation is currently in `annotation`. Used
    /// to save the sidecar back to the right file when the user navigates.
    annotation_path: Option<PathBuf>,
    /// Last image-pixel position painted, for line-segment interpolation
    /// between mouse samples on the same drag.
    last_paint_pos: Option<(f32, f32)>,
    /// True if the GPU annotation texture needs to be (re)created next frame
    /// to match `annotation`'s dimensions.
    annotation_needs_full_upload: bool,

    /// Window pinned above other apps + decorations stripped. Toggled with
    /// `T`. Lets the viewer float over a painting app as a reference.
    reference_mode: bool,
    /// Hide sidebar / tools / status panels for a chrome-free image. `\`.
    compact: bool,
    /// Sidebar filter: only show starred entries.
    show_only_starred: bool,
    /// Sidebar text filter. Matched case-insensitively against filename.
    /// A leading `.` is treated as an extension filter (e.g. `.jpg`).
    filter_query: String,

    /// Multi-image grid compare. Tile 0 is the currently-selected image
    /// (drawn from `current_image`); tiles 1..=3 are user-picked extras.
    /// When non-empty, the viewport renders a 1xN strip or 2x2 layout
    /// depending on `grid_paths.len()` (0=off, 1=N/A, 2=1x2, 3=1x3, 4=2x2).
    grid_paths: Vec<Option<PathBuf>>,
    grid_images: Vec<Option<Arc<DecodedImage>>>,
    /// Set when any tile slot's content has changed; drives the per-frame
    /// upload (one `GridUpload` entry per slot 1..=3).
    grid_dirty: [bool; 3],
    /// Bumped each time grid mode is started; in-flight decodes that arrive
    /// stale (i.e. with a different generation) are dropped.
    grid_generation: u64,

    /// Pinboard / moodboard state. Always-allocated; `pinboard_mode`
    /// controls whether the viewport renders it. Items persist across
    /// pinboard_mode toggles within a session but are not saved to disk.
    pinboard: Pinboard,
    pinboard_mode: bool,

    /// Manual rotation override in 90-degree clockwise quarters: 0/1/2/3 =
    /// 0/90/180/270 deg. Layered on top of EXIF auto-orient. Per-session
    /// (not saved to disk) for now.
    manual_rotation: u32,

    /// True while a viewport drag is moving the A/B compare divider line.
    /// Captured at drag-start when the press was within a few pixels of the
    /// divider in screen space; held until drag_stopped. Pre-empts the pan
    /// and annotation drag paths so the slider can be tweaked directly on
    /// the image without chasing it down to the toolbar slider.
    compare_divider_drag: bool,

    show_histogram: bool,
    show_exif: bool,
    show_color_panel: bool,
    clipping_warning: bool,
    split_tone_amount: f32,
    shadow_tint: [f32; 3],
    highlight_tint: [f32; 3],

    /// Multiplier to apply to the current zoom on the next viewport frame
    /// (set by keyboard zoom-step shortcuts).
    pending_zoom_step: Option<f32>,

    /// Most-recently-opened folders, MRU first, capped at `RECENT_FOLDERS_MAX`.
    recent_folders: Vec<PathBuf>,

    /// Pixel under the cursor in the viewport (image coords + RGBA), updated
    /// each frame. Drives the eyedropper readout in the status bar.
    cursor_pixel: Option<CursorSample>,

    /// Set by the toolbar Export button. Consumed in `ui()` where the wgpu
    /// frame is in scope. The path is the destination chosen by the user;
    /// extension determines JPEG vs PNG encoding.
    pending_export: Option<PathBuf>,
    export_in_progress: bool,

    /// Folder filesystem watcher; replaced when the folder changes.
    folder_watcher: Option<FolderWatcher>,
    /// Deadline for the next debounced folder rescan.
    folder_refresh_at: Option<Instant>,
    /// On the next FilesFound, restore selection to this path if present.
    restore_selection: Option<PathBuf>,
    scan_generation: u64,
    scan_cancel: Option<CancelToken>,
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
    #[serde(default)]
    recent_folders: Vec<PathBuf>,
    #[serde(default)]
    show_histogram: bool,
    #[serde(default)]
    show_exif: bool,
    #[serde(default)]
    clipping_warning: bool,
    #[serde(default)]
    split_tone_amount: Option<f32>,
    #[serde(default)]
    shadow_tint: Option<[f32; 3]>,
    #[serde(default)]
    highlight_tint: Option<[f32; 3]>,
    #[serde(default)]
    flip_h: Option<bool>,
    #[serde(default)]
    value_study: Option<bool>,
    #[serde(default)]
    value_levels: Option<u32>,
    #[serde(default)]
    crop_ratio: Option<CropRatio>,
    #[serde(default)]
    brush_color: Option<[f32; 3]>,
    #[serde(default)]
    brush_radius: Option<f32>,
    #[serde(default)]
    reference_mode: Option<bool>,
    #[serde(default)]
    compact: Option<bool>,
    #[serde(default)]
    show_only_starred: Option<bool>,
}

const RECENT_FOLDERS_MAX: usize = 8;

impl TessellatorApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let render_state = cc.wgpu_render_state.as_ref().expect(
            "WGPU render state not available - eframe must be built with the wgpu renderer",
        );
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
            pending_image_requests: HashMap::new(),
            last_select_time: None,
            direction_streak: 0,
            image_cache: ImageCache::new(compute_cache_cap()),
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
            current_image_path: None,
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
            flip_h: saved.flip_h.unwrap_or(false),
            value_study: saved.value_study.unwrap_or(false),
            value_levels: saved.value_levels.unwrap_or(4).clamp(2, 8),
            crop_ratio: saved.crop_ratio.unwrap_or(CropRatio::None),
            annotating: false,
            brush_color: saved.brush_color.unwrap_or([1.0, 0.15, 0.15]),
            brush_radius: saved.brush_radius.unwrap_or(8.0).clamp(1.0, 200.0),
            erasing: false,
            annotation: None,
            annotation_path: None,
            last_paint_pos: None,
            annotation_needs_full_upload: false,
            reference_mode: saved.reference_mode.unwrap_or(false),
            compact: saved.compact.unwrap_or(false),
            show_only_starred: saved.show_only_starred.unwrap_or(false),
            filter_query: String::new(),
            grid_paths: Vec::new(),
            grid_images: Vec::new(),
            grid_dirty: [false; 3],
            grid_generation: 0,
            pinboard: Pinboard::default(),
            pinboard_mode: false,
            compare_divider_drag: false,
            manual_rotation: 0,
            show_histogram: saved.show_histogram,
            show_exif: saved.show_exif,
            show_color_panel: false,
            clipping_warning: saved.clipping_warning,
            split_tone_amount: saved.split_tone_amount.unwrap_or(0.0),
            // Cinematic teal-and-orange defaults: visible immediately when
            // the user drags the amount slider above 0.
            shadow_tint: saved.shadow_tint.unwrap_or([0.20, 0.40, 0.55]),
            highlight_tint: saved.highlight_tint.unwrap_or([0.95, 0.65, 0.30]),
            pending_zoom_step: None,
            recent_folders: saved.recent_folders,
            cursor_pixel: None,
            pending_export: None,
            export_in_progress: false,
            folder_watcher: None,
            folder_refresh_at: None,
            restore_selection: None,
            scan_generation: 0,
            scan_cancel: None,
        };

        if let Some(path) = saved.folder_path
            && path.is_dir()
        {
            app.open_folder(path, &cc.egui_ctx);
        }

        // Re-apply persisted window mode on startup so a relaunch keeps the
        // user's "always on top" preference.
        if app.reference_mode {
            app.apply_reference_mode(&cc.egui_ctx);
        }

        app
    }

    /// Push the current `reference_mode` flag through the viewport: pinned
    /// above other windows + OS chrome removed when on, normal otherwise.
    fn apply_reference_mode(&self, ctx: &egui::Context) {
        let level = if self.reference_mode {
            egui::WindowLevel::AlwaysOnTop
        } else {
            egui::WindowLevel::Normal
        };
        ctx.send_viewport_cmd(egui::ViewportCommand::WindowLevel(level));
        ctx.send_viewport_cmd(egui::ViewportCommand::Decorations(!self.reference_mode));
    }

    fn open_folder(&mut self, path: PathBuf, ctx: &egui::Context) {
        // Save any pending annotation before swapping folders. Otherwise the
        // outgoing layer would be dropped on the floor.
        self.save_current_annotation();
        self.annotation = None;
        self.annotation_path = None;
        self.annotation_needs_full_upload = false;
        self.folder_path = Some(path.clone());
        self.push_recent_folder(path.clone());
        self.files.clear();
        self.selected_index = None;
        self.requested_thumbnails.clear();
        self.cancel_all_image_requests();
        self.last_select_time = None;
        self.direction_streak = 0;
        self.current_image_size = None;
        self.current_image = None;
        self.current_image_path = None;
        self.needs_upload = false;
        self.cursor_pixel = None;
        self.clear_compare();
        ctx.send_viewport_cmd(egui::ViewportCommand::Title("Tessellator".to_string()));
        self.start_folder_scan(path.clone(), ctx);

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

    /// Move `path` to the front of the MRU list, deduping and capping length.
    fn push_recent_folder(&mut self, path: PathBuf) {
        self.recent_folders.retain(|p| p != &path);
        self.recent_folders.insert(0, path);
        self.recent_folders.truncate(RECENT_FOLDERS_MAX);
    }

    fn start_folder_scan(&mut self, path: PathBuf, ctx: &egui::Context) {
        if let Some(token) = self.scan_cancel.take() {
            token.cancel();
        }
        self.scan_generation = self.scan_generation.wrapping_add(1);
        let generation = self.scan_generation;
        let token = CancelToken::new();
        self.scan_cancel = Some(token.clone());
        io::scan_folder(
            path,
            self.recursion_depth,
            generation,
            token,
            self.sender.clone(),
            ctx.clone(),
        );
    }

    fn start_compare(&mut self, path: PathBuf, ctx: &egui::Context) {
        // Cancel a prior compare-target if it differs from the new pick.
        if let Some(prev) = self.compare_path.take()
            && prev != path
        {
            self.cancel_image_request(&prev);
        }
        self.compare_generation = self.compare_generation.wrapping_add(1);
        self.compare_path = Some(path.clone());
        if let Some(image) = self.image_cache.get(&path) {
            self.compare_image = Some(image);
            self.compare_dirty = true;
        } else {
            self.compare_image = None;
            self.compare_dirty = true;
            // Cancel any prior in-flight request for this same path so we get
            // fresh work tagged with the current Compare generation.
            self.cancel_image_request(&path);
            self.dispatch_image_request(
                path,
                ImagePurpose::Compare {
                    generation: self.compare_generation,
                },
                ctx,
            );
        }
    }

    /// Begin a multi-image grid compare. `paths` is the full set of tiles
    /// including the primary; tile 0 is implicit (the currently-selected
    /// image), tiles 1..=3 come from `paths`. Caller passes 1..=3 paths.
    /// Mutually exclusive with the slider compare mode.
    fn start_grid(&mut self, paths: Vec<PathBuf>, ctx: &egui::Context) {
        if paths.is_empty() {
            return;
        }
        self.clear_compare();
        // Bump generation first so any in-flight grid decodes from a prior
        // start_grid get discarded when they arrive.
        self.grid_generation = self.grid_generation.wrapping_add(1);
        self.clear_grid_internal();
        let n = paths.len().min(3);
        self.grid_paths = paths[..n].iter().cloned().map(Some).collect();
        self.grid_images = vec![None; n];
        self.grid_dirty = [false; 3];
        // Snapshot (slot_index, path) pairs before mutating self in the loop;
        // dispatch_image_request needs &mut self.
        let plan: Vec<(usize, PathBuf)> = self
            .grid_paths
            .iter()
            .enumerate()
            .filter_map(|(i, p)| p.clone().map(|p| (i, p)))
            .collect();
        for (i, p) in plan {
            let tile_slot = (i as u32) + 1;
            if let Some(img) = self.image_cache.get(&p) {
                self.grid_images[i] = Some(img);
                self.grid_dirty[i] = true;
            } else {
                self.cancel_image_request(&p);
                self.dispatch_image_request(
                    p,
                    ImagePurpose::Grid {
                        slot: tile_slot,
                        generation: self.grid_generation,
                    },
                    ctx,
                );
            }
        }
    }

    fn clear_grid_internal(&mut self) {
        // Cancel any pending grid decodes.
        let pending: Vec<PathBuf> = self.grid_paths.iter().filter_map(|p| p.clone()).collect();
        for p in pending {
            self.cancel_image_request(&p);
        }
        // Mark every slot dirty so the GPU clears its texture next frame.
        for i in 0..3 {
            self.grid_dirty[i] = true;
        }
        self.grid_paths.clear();
        self.grid_images.clear();
    }

    fn clear_grid(&mut self) {
        self.grid_generation = self.grid_generation.wrapping_add(1);
        self.clear_grid_internal();
    }

    fn clear_compare(&mut self) {
        if let Some(prev) = self.compare_path.take() {
            self.cancel_image_request(&prev);
            self.compare_dirty = true;
        }
        if self.compare_image.take().is_some() {
            self.compare_dirty = true;
        }
        // Bump generation so any in-flight Compare result is discarded.
        self.compare_generation = self.compare_generation.wrapping_add(1);
    }

    /// Issue a high-res decode for `path`, replacing any prior in-flight
    /// request for the same path (the prior gets cancelled).
    fn dispatch_image_request(
        &mut self,
        path: PathBuf,
        purpose: ImagePurpose,
        ctx: &egui::Context,
    ) {
        if let Some(prior) = self
            .pending_image_requests
            .insert(path.clone(), CancelToken::new())
        {
            prior.cancel();
        }
        let token = self
            .pending_image_requests
            .get(&path)
            .expect("just inserted")
            .clone();
        io::request_image(path, purpose, token, self.sender.clone(), ctx.clone());
    }

    fn cancel_image_request(&mut self, path: &Path) {
        if let Some(tok) = self.pending_image_requests.remove(path) {
            tok.cancel();
        }
    }

    fn cancel_all_image_requests(&mut self) {
        for (_, tok) in self.pending_image_requests.drain() {
            tok.cancel();
        }
    }

    /// Display the given image in the viewport (and reset to fit-to-screen).
    /// Also runs the annotation handoff: saves any outgoing layer and tries
    /// to load a sidecar for the new image.
    fn show_image(&mut self, image: Arc<DecodedImage>) {
        let (w, h) = (image.width, image.height);
        let is_preview = image.is_preview;
        self.current_image_size = Some([w, h]);
        self.current_image = Some(image);
        self.needs_upload = true;
        self.view = ViewState::FitOnNextFrame;

        let new_path = self
            .selected_index
            .and_then(|i| self.files.get(i))
            .map(|f| f.path.clone());
        // Record which path's pixels are now loaded so the viewport can tell
        // "still loading" from "displayed" without inspecting in-flight queues.
        self.current_image_path = new_path.clone();
        if new_path != self.annotation_path {
            self.save_current_annotation();
            self.annotation = if is_preview {
                None
            } else {
                new_path
                    .as_deref()
                    .and_then(|p| AnnotationLayer::load_sidecar(p, w, h))
            };
            self.annotation_needs_full_upload = self.annotation.is_some();
            self.annotation_path = if is_preview { None } else { new_path };
            self.last_paint_pos = None;
            self.erasing = false;
        }
    }

    /// Write the current annotation layer to its sidecar PNG if it has any
    /// strokes (`touched`) and the in-memory state diverges from disk
    /// (`unsaved`). The actual encode + write runs on a one-off background
    /// thread; PNG encoding a multi-MP RGBA buffer can take seconds and
    /// would otherwise block the UI on every drag-stop.
    /// Send the currently-selected image to the OS trash, with a confirm
    /// dialog. Trash semantics (not raw unlink) so a misfired hotkey is
    /// recoverable. After deletion, advance the selection to the next
    /// surviving entry and trigger a rescan to pick up sidecar changes.
    fn delete_selected_to_trash(&mut self, ctx: &egui::Context) {
        let Some(i) = self.selected_index else { return };
        let Some(entry) = self.files.get(i) else {
            return;
        };
        let path = entry.path.clone();
        let name = entry.name.clone();
        let confirm = rfd::MessageDialog::new()
            .set_level(rfd::MessageLevel::Warning)
            .set_title("Move to Trash")
            .set_description(format!("Move \"{}\" to the Trash?", name))
            .set_buttons(rfd::MessageButtons::OkCancel)
            .show();
        if confirm != rfd::MessageDialogResult::Ok {
            return;
        }
        // trash::delete handles the platform-specific OS call (Recycle Bin
        // on Windows, ~/.Trash on macOS, gio trash on Linux).
        match trash::delete(&path) {
            Ok(()) => {
                log::info!("Trashed {:?}", path);
                // Move selection forward (or back if we deleted the last).
                let next = if i + 1 < self.files.len() {
                    Some(i)
                } else if i > 0 {
                    Some(i - 1)
                } else {
                    None
                };
                self.files.remove(i);
                self.selected_index = next;
                if let Some(idx) = next
                    && let Some(e) = self.files.get(idx)
                {
                    let p = e.path.clone();
                    self.high_res_generation = self.high_res_generation.wrapping_add(1);
                    if let Some(image) = self.image_cache.get(&p) {
                        self.show_image(image);
                    } else {
                        self.dispatch_image_request(
                            p,
                            ImagePurpose::Display {
                                generation: self.high_res_generation,
                            },
                            ctx,
                        );
                    }
                } else {
                    self.current_image = None;
                    self.current_image_path = None;
                    self.current_image_size = None;
                }
                // The folder watcher will fire its own rescan, but kick one
                // off immediately so the sidebar updates without waiting.
                self.rescan_current_folder(ctx);
            }
            Err(e) => {
                log::warn!("Failed to trash {:?}: {}", path, e);
                rfd::MessageDialog::new()
                    .set_level(rfd::MessageLevel::Error)
                    .set_title("Delete failed")
                    .set_description(format!("Could not trash file: {}", e))
                    .show();
            }
        }
    }

    /// Render the current image with effects baked in (grayscale, dither,
    /// posterize, split-tone, clipping warning, annotation, manual rotation,
    /// flip-h) at native source resolution and write to `path` as PNG/JPEG
    /// based on extension. Skips loupe, compare, grid, overlays, and crop.
    fn run_export(
        &mut self,
        path: &std::path::Path,
        frame: &mut eframe::Frame,
        ctx: &egui::Context,
    ) {
        if self.export_in_progress {
            self.last_load_error = Some("Export already in progress".to_string());
            return;
        }
        let Some(image) = self.current_image.clone() else {
            self.last_load_error = Some("Export: no image loaded".to_string());
            return;
        };
        if image.is_preview {
            self.last_load_error =
                Some("Export: full-resolution image is still loading".to_string());
            return;
        }
        let Some(render_state) = frame.wgpu_render_state() else {
            self.last_load_error = Some("Export: no wgpu render state".to_string());
            return;
        };

        // Output dimensions follow displayed orientation: 90/270 swap axes.
        let rotated_quarter = self.manual_rotation == 1 || self.manual_rotation == 3;
        let (out_w, out_h) = if rotated_quarter {
            (image.height, image.width)
        } else {
            (image.width, image.height)
        };

        // Build identity view (fills the NDC quad), with flip_h folded in.
        let scale = egui::vec2(if self.flip_h { -1.0 } else { 1.0 }, 1.0);
        let view = crate::view::view_matrix(scale, egui::vec2(0.0, 0.0));

        let settings = crate::gpu::ShaderSettings {
            view_matrix: view,
            grayscale: self.grayscale,
            // Composition overlays are intentionally excluded from export.
            overlay_opacity: 0.0,
            grid_size: self.grid_size,
            overlay_mode: 0,
            compare_divider: 0.5,
            compare_active: 0,
            loupe_active: 0,
            loupe_zoom: 1.0,
            loupe_center_uv: [0.5, 0.5],
            loupe_center_screen: [0.0, 0.0],
            loupe_radius: 0.0,
            dither: u32::from(self.dither),
            split_tone_active: u32::from(
                self.shadow_tint != [0.5; 3] || self.highlight_tint != [0.5; 3],
            ),
            split_tone_amount: self.split_tone_amount,
            clipping_warning: u32::from(self.clipping_warning),
            posterize_active: u32::from(self.value_study),
            posterize_levels: self.value_levels,
            annotation_active: u32::from(self.annotation.is_some()),
            grid_active: 0,
            grid_count: 0,
            screen_aspect: out_w as f32 / out_h.max(1) as f32,
            manual_rotation: self.manual_rotation,
            tile_image_aspects: [
                image.width as f32 / image.height.max(1) as f32,
                0.0,
                0.0,
                0.0,
            ],
            shadow_tint: [
                self.shadow_tint[0],
                self.shadow_tint[1],
                self.shadow_tint[2],
                0.0,
            ],
            highlight_tint: [
                self.highlight_tint[0],
                self.highlight_tint[1],
                self.highlight_tint[2],
                0.0,
            ],
        };

        let device = render_state.device.clone();
        let queue = render_state.queue.clone();
        let renderer = render_state.renderer.read();
        let Some(resources) = renderer
            .callback_resources
            .get::<crate::gpu::TessellatorResources>()
            .and_then(|tess| tess.export_resources())
        else {
            self.last_load_error = Some("Export: render resources not initialised yet".to_string());
            return;
        };
        drop(renderer);

        let path = path.to_path_buf();
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();
        self.export_in_progress = true;
        let sender = self.sender.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = (|| -> Result<(), String> {
                let rgba = crate::gpu::render_export_resources_to_rgba8(
                    &resources, &device, &queue, out_w, out_h, settings,
                )
                .map_err(|e| format!("GPU readback failed: {}", e))?;
                let img = image::RgbaImage::from_raw(out_w, out_h, rgba)
                    .ok_or_else(|| "buffer size mismatch".to_string())?;
                match ext.as_str() {
                    "jpg" | "jpeg" => {
                        let rgb = image::DynamicImage::ImageRgba8(img).to_rgb8();
                        rgb.save_with_format(&path, image::ImageFormat::Jpeg)
                    }
                    _ => img.save_with_format(&path, image::ImageFormat::Png),
                }
                .map_err(|e| format!("write failed: {}", e))
            })();
            let _ = sender.send(Message::ExportFinished { path, result });
            ctx.request_repaint();
        });
    }

    /// Rename the currently-selected image. We piggyback on rfd's native
    /// save dialog (which shows a filename field) since rfd has no plain
    /// "text input" dialog. The dialog's parent is forced to the image's
    /// folder; whatever the user types becomes the new filename.
    fn rename_selected(&mut self, ctx: &egui::Context) {
        let Some(i) = self.selected_index else { return };
        let Some(entry) = self.files.get(i) else {
            return;
        };
        let path = entry.path.clone();
        let current_name = entry.name.clone();
        let parent = match path.parent() {
            Some(p) => p.to_path_buf(),
            None => return,
        };
        let picked = match rfd::FileDialog::new()
            .set_directory(&parent)
            .set_file_name(&current_name)
            .set_title("Rename image")
            .save_file()
        {
            Some(p) => p,
            None => return,
        };
        // Force same-folder rename: keep only the basename. Rejects empty
        // basenames (just in case the user erased everything).
        let Some(new_name_os) = picked.file_name() else {
            return;
        };
        let new_name = new_name_os.to_string_lossy().into_owned();
        if new_name.is_empty() {
            return;
        }
        let new_path = parent.join(&new_name);
        if new_path == path {
            return;
        }
        if new_path.exists() {
            rfd::MessageDialog::new()
                .set_level(rfd::MessageLevel::Error)
                .set_title("Rename failed")
                .set_description(format!("\"{}\" already exists.", new_name))
                .show();
            return;
        }
        match std::fs::rename(&path, &new_path) {
            Ok(()) => {
                log::info!("Renamed {:?} -> {:?}", path, new_path);
                // Also rename any sidecars so they stay attached.
                let _ = std::fs::rename(
                    AnnotationLayer::sidecar_path(&path),
                    AnnotationLayer::sidecar_path(&new_path),
                );
                let _ = std::fs::rename(star_sidecar_path(&path), star_sidecar_path(&new_path));
                self.restore_selection = Some(new_path);
                self.rescan_current_folder(ctx);
            }
            Err(e) => {
                log::warn!("Rename failed: {}", e);
                rfd::MessageDialog::new()
                    .set_level(rfd::MessageLevel::Error)
                    .set_title("Rename failed")
                    .set_description(format!("Could not rename: {}", e))
                    .show();
            }
        }
    }

    /// Pin the currently-selected sidebar image to the moodboard. Default
    /// world size is derived from the loaded image's aspect (or the
    /// thumbnail's aspect, or 1:1) at a fixed 240-unit long edge. Items
    /// stack near `view_center` with a small per-item offset so a rapid
    /// run of `B` presses fans them out instead of hiding under each other.
    fn pin_selected_to_board(&mut self) {
        let Some(i) = self.selected_index else { return };
        let Some(entry) = self.files.get(i) else {
            return;
        };
        let path = entry.path.clone();
        // Derive aspect: prefer the full-image dims if currently loaded
        // (most accurate), fall back to thumbnail dims, then 1:1.
        let aspect = if Some(&path) == self.current_image_path.as_ref()
            && let Some([w, h]) = self.current_image_size
        {
            (w as f32) / (h.max(1) as f32)
        } else if let Some(tex) = entry.thumbnail.as_ref() {
            let [w, h] = tex.size();
            (w as f32) / (h.max(1) as f32)
        } else {
            1.0
        };
        let long_edge = 240.0;
        let size = if aspect >= 1.0 {
            egui::vec2(long_edge, long_edge / aspect)
        } else {
            egui::vec2(long_edge * aspect, long_edge)
        };
        // Cascade-offset each new pin so they don't perfectly overlap.
        let n = self.pinboard.items.len() as f32;
        let cascade = egui::vec2(20.0 * (n % 8.0), 20.0 * (n % 8.0));
        let pos = self.pinboard.view_center + cascade - size * 0.5;
        self.pinboard.add(PinnedItem { path, pos, size });
    }

    /// Flip the starred state of the currently-selected file and write the
    /// matching sidecar to disk on a background thread. Sidecar presence is
    /// the source of truth; toggling off removes the file.
    fn toggle_star_for_selected(&mut self) {
        let Some(i) = self.selected_index else { return };
        let Some(entry) = self.files.get_mut(i) else {
            return;
        };
        entry.starred = !entry.starred;
        let path = entry.path.clone();
        let starred = entry.starred;
        std::thread::spawn(move || {
            let sidecar = star_sidecar_path(&path);
            if starred {
                // Minimal, forward-compatible JSON. Future versions can add
                // fields (rating, color label, tags) without rewriting old
                // sidecars: any unknown field is ignored on read.
                if let Err(e) = std::fs::write(&sidecar, b"{\"starred\":true}") {
                    log::warn!("Failed to write {:?}: {}", sidecar, e);
                }
            } else {
                match std::fs::remove_file(&sidecar) {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => log::warn!("Failed to remove {:?}: {}", sidecar, e),
                }
            }
        });
    }

    /// Indices into `self.files` that pass the active filter. When the
    /// filter is off this is just `0..files.len()`.
    fn visible_indices(&self) -> Vec<usize> {
        let query = self.filter_query.trim().to_lowercase();
        let ext_filter = query.strip_prefix('.').map(|s| s.to_string());
        self.files
            .iter()
            .enumerate()
            .filter(|(_, f)| {
                if self.show_only_starred && !f.starred {
                    return false;
                }
                if !query.is_empty() {
                    let name_lc = f.name.to_lowercase();
                    return match &ext_filter {
                        Some(ext) => name_lc.ends_with(&format!(".{}", ext)),
                        None => name_lc.contains(&query),
                    };
                }
                true
            })
            .map(|(i, _)| i)
            .collect()
    }

    /// Drop the in-memory annotation layer and delete its sidecar PNG from
    /// disk (if any). Leaves annotation mode toggled as-is so the user can
    /// immediately start a new layer if they want.
    fn remove_current_annotation(&mut self) {
        let path = self.annotation_path.clone();
        self.annotation = None;
        self.last_paint_pos = None;
        // The GPU still holds the prior annotation texture; flag a refresh so
        // the next frame swaps it for the transparent placeholder. Without
        // this, the strokes would linger on screen until something else
        // triggered a bind-group rebuild.
        self.annotation_needs_full_upload = true;
        if let Some(p) = path {
            std::thread::spawn(move || {
                let sidecar = AnnotationLayer::sidecar_path(&p);
                match std::fs::remove_file(&sidecar) {
                    Ok(()) => log::debug!("Removed sidecar: {:?}", sidecar),
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => log::warn!("Failed to remove sidecar {:?}: {}", sidecar, e),
                }
            });
        }
    }

    fn save_current_annotation(&mut self) {
        let Some(layer) = self.annotation.as_mut() else {
            return;
        };
        if !layer.touched || !layer.unsaved {
            return;
        }
        let Some(path) = self.annotation_path.clone() else {
            return;
        };
        layer.unsaved = false;

        // Empty layer -> delete the sidecar so erasing everything actually
        // cleans up the folder rather than leaving a blank PNG behind.
        if layer.is_empty() {
            std::thread::spawn(move || {
                let sidecar = AnnotationLayer::sidecar_path(&path);
                match std::fs::remove_file(&sidecar) {
                    Ok(()) => log::debug!("Removed empty sidecar: {:?}", sidecar),
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => log::warn!("Failed to remove sidecar {:?}: {}", sidecar, e),
                }
            });
            return;
        }

        // Clone the buffer so the worker can encode without borrowing self.
        // This is a Vec memcpy of width*height*4 bytes; far cheaper than the
        // PNG encode that follows it.
        let buf = layer.rgba.clone();
        std::thread::spawn(move || {
            let sidecar = AnnotationLayer::sidecar_path(&path);
            match buf.save(&sidecar) {
                Ok(()) => log::debug!("Saved sidecar: {:?}", sidecar),
                Err(e) => log::warn!("Failed to save sidecar for {:?}: {}", path, e),
            }
        });
    }

    fn select_image(&mut self, index: usize, ctx: &egui::Context) {
        if index >= self.files.len() {
            return;
        }
        // Capture prev_index up front so subsequent reorderings of this
        // function can't accidentally read selected_index after it's been
        // overwritten - the streak math depends on the *previous* value.
        let prev_index = self.selected_index;
        self.update_direction_streak(prev_index, index);

        let path = self.files[index].path.clone();
        log::info!(
            "Selected: {} (streak={})",
            self.files[index].name,
            self.direction_streak
        );
        self.selected_index = Some(index);
        self.high_res_generation = self.high_res_generation.wrapping_add(1);
        self.last_load_error = None;
        self.pending_scroll_to_index = Some(index);
        ctx.send_viewport_cmd(egui::ViewportCommand::Title(format!(
            "Tessellator - {}",
            self.files[index].name
        )));

        if let Some(image) = self.image_cache.get(&path) {
            log::debug!("Cache hit: {:?}", path.file_name().unwrap_or_default());
            self.show_image(image);
        } else if self.pending_image_requests.contains_key(&path) {
            // An in-flight Preload (or earlier request) will cover this. The
            // preload-completes-for-current-path branch in drain_messages will
            // surface it when ready.
            log::debug!(
                "Awaiting in-flight request: {:?}",
                path.file_name().unwrap_or_default()
            );
        } else {
            self.dispatch_image_request(
                path,
                ImagePurpose::Display {
                    generation: self.high_res_generation,
                },
                ctx,
            );
        }

        self.preload_neighbors(ctx);
    }

    /// Update `direction_streak` based on the new selection vs the previous.
    /// Same direction extends the streak; reversal or pause resets it.
    /// `prev_index` is passed explicitly so this function can't be silently
    /// broken by a reorder that reads `selected_index` after assignment.
    fn update_direction_streak(&mut self, prev_index: Option<usize>, new_index: usize) {
        const STREAK_DEBOUNCE: Duration = Duration::from_millis(500);
        let now = Instant::now();
        let recent = self
            .last_select_time
            .is_some_and(|t| now.duration_since(t) < STREAK_DEBOUNCE);

        let signed_step: i32 = match prev_index {
            Some(prev) if recent => match new_index.cmp(&prev) {
                std::cmp::Ordering::Greater => 1,
                std::cmp::Ordering::Less => -1,
                std::cmp::Ordering::Equal => 0,
            },
            _ => 0,
        };

        self.direction_streak = next_streak(self.direction_streak, signed_step);
        self.last_select_time = Some(now);
    }

    /// Compute the active preload window based on traversal direction and
    /// then (1) cancel anything pending that's outside it, (2) dispatch
    /// preloads for empty slots inside it.
    fn preload_neighbors(&mut self, ctx: &egui::Context) {
        let Some(current) = self.selected_index else {
            return;
        };
        let neighbors = self.compute_preload_window(current);

        // Build the protected-paths set: the preload window plus the
        // currently-displayed image (whose Display request may still be in
        // flight) plus the active compare image.
        let mut keep: HashSet<PathBuf> = neighbors
            .iter()
            .filter_map(|&i| self.files.get(i).map(|f| f.path.clone()))
            .collect();
        if let Some(p) = self.files.get(current).map(|f| f.path.clone()) {
            keep.insert(p);
        }
        if let Some(p) = self.compare_path.clone() {
            keep.insert(p);
        }

        // Cancel any pending request outside the protected set. Collected
        // first so we don't mutate the map while iterating it.
        let to_cancel: Vec<PathBuf> = self
            .pending_image_requests
            .keys()
            .filter(|p| !keep.contains(p.as_path()))
            .cloned()
            .collect();
        for path in to_cancel {
            log::debug!(
                "Cancelling stale request: {:?}",
                path.file_name().unwrap_or_default()
            );
            self.cancel_image_request(&path);
        }

        // Issue preloads for slots not already cached or in flight.
        for n in neighbors {
            let path = self.files[n].path.clone();
            if self.image_cache.contains(&path) || self.pending_image_requests.contains_key(&path) {
                continue;
            }
            self.dispatch_image_request(path, ImagePurpose::Preload, ctx);
        }
    }

    /// YACReader-style preload window: up to four images before and four
    /// after the current one. Ordered by distance so adjacent images warm
    /// first, with forward navigation getting the first slot at each distance.
    fn compute_preload_window(&self, current: usize) -> Vec<usize> {
        compute_preload_window(current, self.files.len())
    }

    fn drain_messages(&mut self, ctx: &egui::Context) {
        while let Ok(msg) = self.receiver.try_recv() {
            match msg {
                Message::FilesFound { generation, files } => {
                    if generation != self.scan_generation {
                        log::debug!(
                            "Discarding stale folder scan (gen {} != {})",
                            generation,
                            self.scan_generation
                        );
                        continue;
                    }
                    if self
                        .scan_cancel
                        .as_ref()
                        .is_some_and(|token| token.is_cancelled())
                    {
                        continue;
                    }
                    self.scan_cancel = None;
                    self.files = files
                        .into_iter()
                        .map(|s| FileEntry {
                            name: s
                                .path
                                .file_name()
                                .unwrap_or_default()
                                .to_string_lossy()
                                .to_string(),
                            // Sidecar presence is the source of truth for the
                            // starred flag; cheap stat() call per file.
                            starred: star_sidecar_path(&s.path).is_file(),
                            path: s.path,
                            size_bytes: s.size_bytes,
                            thumbnail: None,
                            source_dims: None,
                        })
                        .collect();
                    if let Some(target) = self.restore_selection.take()
                        && let Some(i) = self.files.iter().position(|f| f.path == target)
                    {
                        self.selected_index = Some(i);
                        self.pending_scroll_to_index = Some(i);
                    }
                }
                Message::ThumbnailLoaded {
                    path,
                    image,
                    source_dims,
                } => {
                    if let Some(entry) = self.files.iter_mut().find(|f| f.path == path) {
                        let handle = ctx.load_texture(
                            path.to_string_lossy(),
                            image,
                            egui::TextureOptions::default(),
                        );
                        entry.thumbnail = Some(handle);
                        entry.source_dims = Some(source_dims);
                    }
                }
                Message::ThumbnailFailed => {
                    // Path stays in requested_thumbnails so we don't retry.
                }
                Message::ImageDecoded {
                    path,
                    image,
                    purpose,
                } => {
                    match purpose {
                        ImagePurpose::PreviewDisplay { generation } => {
                            if generation != self.high_res_generation {
                                continue;
                            }
                            let is_current_selection = self
                                .selected_index
                                .and_then(|i| self.files.get(i))
                                .map(|f| f.path == path)
                                .unwrap_or(false);
                            if is_current_selection {
                                self.show_image(image);
                            }
                        }
                        ImagePurpose::Display { generation } => {
                            self.pending_image_requests.remove(&path);
                            self.image_cache.insert(path.clone(), image.clone());
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
                            self.pending_image_requests.remove(&path);
                            self.image_cache.insert(path.clone(), image.clone());
                            // Cached above; if the user has since navigated to
                            // this image, surface it now. Compare by path
                            // (not by dimensions) - two images in the same
                            // folder often share dimensions, and a dim-only
                            // check would silently skip the display, leaving
                            // the loading spinner stuck up forever.
                            if let Some(i) = self.selected_index
                                && self.files.get(i).map(|f| &f.path) == Some(&path)
                                && self.current_image_path.as_deref() != Some(&path)
                            {
                                self.show_image(image);
                            }
                        }
                        ImagePurpose::Compare { generation } => {
                            self.pending_image_requests.remove(&path);
                            self.image_cache.insert(path.clone(), image.clone());
                            if generation == self.compare_generation
                                && self.compare_path.as_deref() == Some(&path)
                            {
                                self.compare_image = Some(image);
                                self.compare_dirty = true;
                            }
                        }
                        ImagePurpose::Grid { slot, generation } => {
                            self.pending_image_requests.remove(&path);
                            self.image_cache.insert(path.clone(), image.clone());
                            if generation == self.grid_generation {
                                let i = (slot.saturating_sub(1)) as usize;
                                if let Some(slot_path) =
                                    self.grid_paths.get(i).and_then(|p| p.clone())
                                    && slot_path == path
                                {
                                    self.grid_images[i] = Some(image);
                                    self.grid_dirty[i] = true;
                                }
                            }
                        }
                    }
                }
                Message::ImageFailed {
                    path,
                    error,
                    purpose,
                } => {
                    self.pending_image_requests.remove(&path);
                    let is_current_selection = self
                        .selected_index
                        .and_then(|i| self.files.get(i))
                        .map(|f| f.path == path)
                        .unwrap_or(false);
                    match purpose {
                        ImagePurpose::Display { generation } => {
                            if generation != self.high_res_generation {
                                continue;
                            }
                            self.last_load_error = Some(format!(
                                "Failed to load {}: {}",
                                path.file_name().unwrap_or_default().to_string_lossy(),
                                error
                            ));
                        }
                        ImagePurpose::PreviewDisplay { .. } => {}
                        ImagePurpose::Preload if is_current_selection => {
                            // The preload that select_image was relying on
                            // failed - the user is staring at a loading
                            // spinner with no in-flight work. Surface the
                            // failure as a real load error so they aren't
                            // stuck.
                            self.last_load_error = Some(format!(
                                "Failed to load {}: {}",
                                path.file_name().unwrap_or_default().to_string_lossy(),
                                error
                            ));
                        }
                        _ => {}
                    }
                }
                Message::FolderChanged => {
                    let deadline = Instant::now() + FOLDER_REFRESH_DEBOUNCE;
                    self.folder_refresh_at = Some(deadline);
                    ctx.request_repaint_after(FOLDER_REFRESH_DEBOUNCE + Duration::from_millis(50));
                }
                Message::ExportFinished { path, result } => {
                    self.export_in_progress = false;
                    match result {
                        Ok(()) => {
                            log::info!("Exported {}", path.display());
                        }
                        Err(e) => {
                            self.last_load_error = Some(format!("Export failed: {}", e));
                        }
                    }
                }
            }
        }
    }

    /// Trigger a folder rescan once watcher events have quieted.
    fn process_folder_refresh(&mut self, ctx: &egui::Context) {
        let Some(deadline) = self.folder_refresh_at else {
            return;
        };
        if Instant::now() < deadline {
            return;
        }
        self.folder_refresh_at = None;
        log::info!("Watcher triggered re-scan");
        self.rescan_current_folder(ctx);
    }

    /// Re-scan the current folder while keeping decoded images and in-flight
    /// requests alive. Used by the watcher (file added / removed externally)
    /// and Cmd+R. Distinct from `open_folder` which rebuilds everything from
    /// scratch - that path threw away ~512 MB of decoded mip chains for what
    /// is usually just a one-file delta.
    fn rescan_current_folder(&mut self, ctx: &egui::Context) {
        let Some(folder) = self.folder_path.clone() else {
            return;
        };
        // Pin selection to the current file so the scan resolves it back
        // when the new file list arrives.
        self.restore_selection = self
            .selected_index
            .and_then(|i| self.files.get(i))
            .map(|f| f.path.clone());
        // The new file list will repopulate these. We keep image_cache,
        // pending_image_requests, current_image / current_image_path so
        // navigation back to recently-viewed images stays instant.
        self.files.clear();
        self.selected_index = None;
        self.requested_thumbnails.clear();
        self.start_folder_scan(folder.clone(), ctx);
        // Reattach the watcher: the prior one may have been invalidated by
        // an unmount or rename of the watched path.
        self.folder_watcher = None;
        match FolderWatcher::new(
            &folder,
            self.recursion_depth > 1,
            self.sender.clone(),
            ctx.clone(),
        ) {
            Ok(w) => self.folder_watcher = Some(w),
            Err(e) => log::warn!("Failed to restart folder watcher for {:?}: {}", folder, e),
        }
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
        if p.refresh && self.folder_path.is_some() {
            self.rescan_current_folder(ctx);
        }
        if p.escape {
            self.clear_compare();
            self.clear_grid();
        }
        if p.copy_path
            && let Some(entry) = self.selected_index.and_then(|i| self.files.get(i))
        {
            let s = entry.path.to_string_lossy().into_owned();
            log::debug!("Copied path to clipboard: {}", s);
            ctx.copy_text(s);
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
        if p.toggle_flip_h {
            self.flip_h = !self.flip_h;
        }
        if p.toggle_value_study {
            self.value_study = !self.value_study;
        }
        if p.toggle_annotate {
            self.annotating = !self.annotating;
        }
        if p.toggle_reference_mode {
            self.reference_mode = !self.reference_mode;
            self.apply_reference_mode(ctx);
        }
        if p.toggle_compact {
            self.compact = !self.compact;
        }
        if p.toggle_star {
            self.toggle_star_for_selected();
        }
        if p.toggle_starred_filter {
            self.show_only_starred = !self.show_only_starred;
        }
        if p.toggle_pinboard {
            self.pinboard_mode = !self.pinboard_mode;
        }
        if p.pin_selected {
            self.pin_selected_to_board();
        }
        if p.delete_pinned && self.pinboard_mode {
            self.pinboard.remove_selected();
        }
        if p.rotate_cw {
            self.manual_rotation = (self.manual_rotation + 1) & 3;
        }
        if p.rotate_ccw {
            self.manual_rotation = (self.manual_rotation + 3) & 3;
        }
        if p.delete_image {
            self.delete_selected_to_trash(ctx);
        }

        // Navigation steps through the *visible* (post-filter) list so a
        // "starred only" view doesn't silently skip past hidden entries.
        let visible = self.visible_indices();
        if visible.is_empty() {
            return;
        }
        let last = visible.len() - 1;
        let current_pos = self
            .selected_index
            .and_then(|i| visible.iter().position(|v| *v == i));
        let target_pos = if p.home {
            Some(0)
        } else if p.end {
            Some(last)
        } else if p.left {
            Some(current_pos.map_or(0, |i| i.saturating_sub(1)))
        } else if p.right {
            Some(current_pos.map_or(0, |i| (i + 1).min(last)))
        } else if p.page_up {
            Some(current_pos.map_or(0, |i| i.saturating_sub(10)))
        } else if p.page_down {
            Some(current_pos.map_or(0, |i| (i + 10).min(last)))
        } else {
            None
        };
        if let Some(pos) = target_pos {
            let i = visible[pos];
            if Some(i) != self.selected_index {
                self.select_image(i, ctx);
            }
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
            let (folder, target) = if path.is_dir() {
                (Some(path), None)
            } else if path.is_file() {
                let parent = path.parent().map(|p| p.to_path_buf());
                // Stash the dropped file's path so the next FilesFound
                // handler selects it after the scan completes.
                (parent, Some(path))
            } else {
                (None, None)
            };
            if let Some(folder) = folder {
                if target.is_some() {
                    self.restore_selection = target;
                }
                self.open_folder(folder, ctx);
                break;
            }
        }
    }
}

impl eframe::App for TessellatorApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        if let Some(path) = self.pending_export.take() {
            self.run_export(&path, frame, ui.ctx());
        }
        self.drain_messages(ui.ctx());
        self.process_folder_refresh(ui.ctx());
        self.handle_dropped_files(ui.ctx());
        self.handle_keyboard_nav(ui.ctx());
        // Compact mode strips chrome down to just the viewport, for a clean
        // floating reference. Reference mode (`T`) controls always-on-top
        // and OS decorations independently.
        if !self.compact {
            self.show_sidebar(ui);
            self.show_status_bar(ui);
        }
        if self.pinboard_mode {
            self.show_pinboard(ui);
        } else {
            self.show_viewport(ui);
        }
        if !self.compact {
            self.show_settings_window(ui.ctx());
        }
        self.show_color_window(ui.ctx());
        self.show_exif_window(ui.ctx());
        self.show_drop_overlay(ui.ctx());
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
            recent_folders: self.recent_folders.clone(),
            show_histogram: self.show_histogram,
            show_exif: self.show_exif,
            clipping_warning: self.clipping_warning,
            split_tone_amount: Some(self.split_tone_amount),
            shadow_tint: Some(self.shadow_tint),
            highlight_tint: Some(self.highlight_tint),
            flip_h: Some(self.flip_h),
            value_study: Some(self.value_study),
            value_levels: Some(self.value_levels),
            crop_ratio: Some(self.crop_ratio),
            brush_color: Some(self.brush_color),
            brush_radius: Some(self.brush_radius),
            reference_mode: Some(self.reference_mode),
            compact: Some(self.compact),
            show_only_starred: Some(self.show_only_starred),
        };
        eframe::set_value(storage, eframe::APP_KEY, &state);
    }
}

impl TessellatorApp {
    fn show_sidebar(&mut self, ui: &mut egui::Ui) {
        let mut clicked: Option<usize> = None;
        let mut to_request: Vec<PathBuf> = Vec::new();
        let mut open_folder: Option<PathBuf> = None;
        let mut depth_changed = false;
        let pending_scroll = self.pending_scroll_to_index.take();

        egui::Panel::left("sidebar")
            .resizable(true)
            .default_size(250.0)
            .show_inside(ui, |ui| {
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
                    if !self.recent_folders.is_empty() {
                        ui.menu_button("Recent", |ui| {
                            for path in self.recent_folders.clone() {
                                let label = path
                                    .file_name()
                                    .map(|n| n.to_string_lossy().to_string())
                                    .unwrap_or_else(|| path.display().to_string());
                                let resp =
                                    ui.button(label).on_hover_text(path.display().to_string());
                                if resp.clicked() {
                                    open_folder = Some(path);
                                    ui.close();
                                }
                            }
                        });
                    }
                    ui.label("Depth:");
                    let r = ui.add(
                        egui::DragValue::new(&mut self.recursion_depth)
                            .range(1..=16)
                            .speed(0.1),
                    );
                    // Trigger a rescan only on commit (lost focus or Enter),
                    // not while typing - otherwise typing "10" rescans at
                    // "1" and again at "10".
                    if r.lost_focus() || r.drag_stopped() {
                        depth_changed = true;
                    }
                });
                ui.horizontal(|ui| {
                    let starred_count = self.files.iter().filter(|f| f.starred).count();
                    ui.checkbox(&mut self.show_only_starred, "Starred only")
                        .on_hover_text("Filter to favourites (Shift+S)");
                    ui.label(format!("({}/{})", starred_count, self.files.len()));
                });
                ui.horizontal(|ui| {
                    ui.label("Search:");
                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut self.filter_query)
                            .hint_text("name or .ext")
                            .desired_width(f32::INFINITY),
                    );
                    if resp.changed() {
                        // The visible-row count is about to change; the
                        // current selection's row position will shift, so
                        // request a scroll to keep it visible if it's still
                        // in the filtered set.
                        if let Some(i) = self.selected_index {
                            self.pending_scroll_to_index = Some(i);
                        }
                    }
                });
                ui.separator();

                // Row layout: 44px tall, 4px outer gap. Big enough for a
                // 36px rounded thumbnail with breathing room on both sides.
                let row_height = 44.0;
                let row_inner_h = row_height - 4.0;
                let thumb_size = 36.0;
                let viewport_h = ui.available_height();

                // visible: position-in-list -> index into self.files. When
                // the filter is off this is the identity. Centralising it
                // makes scrolling, click translation, and thumbnail requests
                // all agree on what a row is.
                // Leading `.` switches to suffix-match mode (extension search),
                // anything else is a case-insensitive substring match. See
                // `visible_indices` for the canonical implementation.
                let visible: Vec<usize> = self.visible_indices();

                let mut star_toggle: Option<usize> = None;
                let mut scroll = egui::ScrollArea::vertical();
                if let Some(i) = pending_scroll
                    && let Some(pos) = visible.iter().position(|v| *v == i)
                {
                    let target = (pos as f32) * row_height - viewport_h * 0.5 + row_height * 0.5;
                    scroll = scroll.vertical_scroll_offset(target.max(0.0));
                }

                scroll.show_rows(ui, row_height, visible.len(), |ui, row_range| {
                    for pos in row_range.clone() {
                        let i = visible[pos];
                        let entry = &self.files[i];
                        let is_selected = self.selected_index == Some(i);

                        // Reserve the full row rectangle so we can paint a
                        // selection / hover background under everything,
                        // then lay out the children on top.
                        let avail_w = ui.available_width();
                        let (row_rect, row_resp) = ui.allocate_exact_size(
                            egui::vec2(avail_w, row_inner_h),
                            egui::Sense::click(),
                        );
                        let visuals = ui.style().visuals.clone();
                        if is_selected {
                            ui.painter()
                                .rect_filled(row_rect, 6.0, visuals.selection.bg_fill);
                        } else if row_resp.hovered() {
                            ui.painter().rect_filled(
                                row_rect,
                                6.0,
                                visuals.widgets.hovered.bg_fill,
                            );
                        }

                        // Children laid out inside the row rect with a small
                        // inset so the rounded fill has padding.
                        let inner_rect = row_rect.shrink2(egui::vec2(6.0, 4.0));
                        let mut child = ui.new_child(
                            egui::UiBuilder::new()
                                .max_rect(inner_rect)
                                .layout(egui::Layout::left_to_right(egui::Align::Center)),
                        );

                        // Thumbnail (or a faint placeholder while loading).
                        let thumb_vec = egui::vec2(thumb_size, thumb_size);
                        if let Some(texture) = &entry.thumbnail {
                            child.add(
                                egui::Image::new((texture.id(), thumb_vec)).corner_radius(4.0),
                            );
                        } else {
                            let (rect, _) =
                                child.allocate_exact_size(thumb_vec, egui::Sense::hover());
                            child
                                .painter()
                                .rect_filled(rect, 4.0, visuals.faint_bg_color);
                        }

                        child.add_space(8.0);

                        // Reserve space for the star at the right edge so the
                        // filename has a definite max width to truncate
                        // against (otherwise the row's available width would
                        // be unbounded and Truncate degenerates to Extend).
                        let star_glyph = if entry.starred { "★" } else { "☆" };
                        let star_w = 22.0;
                        let name_max_w = (child.available_width() - star_w - 4.0).max(0.0);

                        // Filename + dimensions, stacked. Bold name on the
                        // top line (truncated with ellipsis), small dim
                        // subtitle below showing source resolution. The
                        // subtitle is suppressed until the thumbnail decode
                        // populates source_dims so the row doesn't reflow.
                        let name_color = if is_selected {
                            visuals.selection.stroke.color
                        } else {
                            visuals.text_color()
                        };
                        let dim_color = if is_selected {
                            visuals.selection.stroke.color.linear_multiply(0.75)
                        } else {
                            visuals.weak_text_color()
                        };
                        child.allocate_ui_with_layout(
                            egui::vec2(name_max_w, row_inner_h),
                            egui::Layout::top_down(egui::Align::LEFT),
                            |ui| {
                                ui.add_space(2.0);
                                let name_text =
                                    egui::RichText::new(&entry.name).strong().color(name_color);
                                ui.add(egui::Label::new(name_text).truncate().selectable(false))
                                    .on_hover_text(&entry.name);
                                if let Some((w, h)) = entry.source_dims {
                                    ui.add(
                                        egui::Label::new(
                                            egui::RichText::new(format!("{} x {}", w, h))
                                                .small()
                                                .color(dim_color),
                                        )
                                        .truncate()
                                        .selectable(false),
                                    );
                                }
                            },
                        );

                        // Star button pinned to the right of the row.
                        child.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let star = ui.small_button(star_glyph).on_hover_text("Toggle star (S)");
                            if star.clicked() {
                                star_toggle = Some(i);
                            }
                        });

                        // The whole row is a single click target for selection.
                        // Star clicks are absorbed by the button above (which
                        // doesn't propagate), so a click here means "select
                        // this row".
                        if row_resp.clicked() {
                            clicked = Some(i);
                        }
                    }
                    // Only request thumbnails for currently-visible rows.
                    for pos in row_range {
                        let i = visible[pos];
                        let entry = &self.files[i];
                        if entry.thumbnail.is_none()
                            && !self.requested_thumbnails.contains(&entry.path)
                        {
                            to_request.push(entry.path.clone());
                        }
                    }
                });

                if let Some(i) = star_toggle {
                    self.selected_index = Some(i);
                    self.toggle_star_for_selected();
                }
            });

        if let Some(path) = open_folder {
            self.open_folder(path, ui.ctx());
        } else if depth_changed && let Some(path) = self.folder_path.clone() {
            // Re-scan with the new depth.
            self.open_folder(path, ui.ctx());
        }

        for path in to_request {
            self.requested_thumbnails.insert(path.clone());
            io::request_thumbnail(path, self.sender.clone(), ui.ctx().clone());
        }

        if let Some(i) = clicked {
            self.select_image(i, ui.ctx());
        }
    }

    fn show_settings_window(&mut self, ctx: &egui::Context) {
        egui::Window::new("Settings")
            .id(egui::Id::new("settings_window"))
            .default_pos(egui::pos2(310.0, 64.0))
            .default_width(500.0)
            .resizable(true)
            .collapsible(true)
            .vscroll(true)
            .show(ctx, |ui| {
                self.show_settings_controls(ui);
            });
    }

    fn show_settings_controls(&mut self, ui: &mut egui::Ui) {
        egui::Grid::new("settings_list")
            .num_columns(2)
            .striped(true)
            .spacing(egui::vec2(14.0, 6.0))
            .show(ui, |ui| {
                settings_row(ui, "Grayscale:", |ui| {
                    ui.add_sized(
                        egui::vec2(180.0, 0.0),
                        egui::Slider::new(&mut self.grayscale, 0.0..=1.0),
                    );
                });

                settings_row(ui, "Values:", |ui| {
                    ui.checkbox(&mut self.value_study, "Enabled")
                        .on_hover_text("Posterize to N bands for value study (V)");
                    if self.value_study {
                        ui.add(egui::Slider::new(&mut self.value_levels, 2..=8).text("bands"));
                    }
                });

                settings_row(ui, "Transform:", |ui| {
                    ui.checkbox(&mut self.flip_h, "Flip H")
                        .on_hover_text("Mirror horizontally to spot composition issues (H)");
                    if ui
                        .button(format!(
                            "Rotate {}",
                            ["0°", "90°", "180°", "270°"][self.manual_rotation as usize]
                        ))
                        .on_hover_text("Rotate clockwise (R) - hold Shift to go counter-clockwise")
                        .clicked()
                    {
                        self.manual_rotation = (self.manual_rotation + 1) & 3;
                    }
                });

                settings_row(ui, "Crop:", |ui| {
                    egui::ComboBox::from_id_salt("crop_ratio")
                        .selected_text(self.crop_ratio.label())
                        .show_ui(ui, |ui| {
                            for r in [
                                CropRatio::None,
                                CropRatio::Square,
                                CropRatio::FourFive,
                                CropRatio::SixteenNine,
                                CropRatio::Golden,
                            ] {
                                ui.selectable_value(&mut self.crop_ratio, r, r.label());
                            }
                        });
                });

                settings_row(ui, "Annotate:", |ui| {
                    ui.toggle_value(&mut self.annotating, "Enabled")
                        .on_hover_text(
                            "Drag to paint over the image; strokes save to a sidecar PNG (A)",
                        );
                    if self.annotating {
                        ui.color_edit_button_rgb(&mut self.brush_color);
                        ui.add(egui::Slider::new(&mut self.brush_radius, 1.0..=200.0).text("px"));
                        ui.toggle_value(&mut self.erasing, "Erase");
                        if ui
                            .button("Clear")
                            .on_hover_text("Remove all strokes and delete the sidecar PNG")
                            .clicked()
                        {
                            self.remove_current_annotation();
                        }
                    }
                });

                settings_row(ui, "Overlay:", |ui| {
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
                        ui.add(
                            egui::Slider::new(&mut self.overlay_opacity, 0.0..=1.0)
                                .text("Opacity"),
                        );
                        if self.overlay_mode == OverlayMode::Grid {
                            ui.add(egui::Slider::new(&mut self.grid_size, 1.0..=50.0).text("Size"));
                        }
                    }
                });

                settings_row(ui, "Grid:", |ui| {
                    let mut show_grid = self.overlay_mode == OverlayMode::Grid;
                    if ui
                        .checkbox(&mut show_grid, "Show")
                        .on_hover_text("Show the composition grid overlay")
                        .changed()
                    {
                        self.overlay_mode = if show_grid {
                            OverlayMode::Grid
                        } else {
                            OverlayMode::None
                        };
                    }
                    if self.overlay_mode == OverlayMode::Grid {
                        ui.add(egui::Slider::new(&mut self.grid_size, 1.0..=50.0).text("Size"));
                        ui.add(
                            egui::Slider::new(&mut self.overlay_opacity, 0.0..=1.0)
                                .text("Opacity"),
                        );
                    }
                });

                settings_row(ui, "View:", |ui| {
                    if ui
                        .button("Fit")
                        .on_hover_text("Fit image to viewport (F)")
                        .clicked()
                    {
                        self.view = ViewState::FitOnNextFrame;
                    }
                    if ui
                        .button("Fill")
                        .on_hover_text("Fill viewport, may crop (Shift+F)")
                        .clicked()
                    {
                        self.view = ViewState::FillOnNextFrame;
                    }
                    if ui
                        .button("100%")
                        .on_hover_text("Display at native pixel size (1)")
                        .clicked()
                    {
                        self.view = ViewState::one_to_one();
                    }
                });

                settings_row(ui, "Compare:", |ui| {
                    self.show_compare_controls(ui);
                });

                settings_row(ui, "Image Grid:", |ui| {
                    self.show_grid_controls(ui);
                });

                settings_row(ui, "Render:", |ui| {
                    ui.checkbox(&mut self.dither, "Dither")
                        .on_hover_text("Add 1-bit noise to remove gradient banding");
                    ui.checkbox(&mut self.clipping_warning, "Clip").on_hover_text(
                        "Highlight clipped pixels - magenta = blown highlights, cyan = crushed shadows",
                    );
                });

                settings_row(ui, "Panels:", |ui| {
                    ui.checkbox(&mut self.show_histogram, "Histogram")
                        .on_hover_text("RGB + luminance histogram overlay");
                    ui.checkbox(&mut self.show_exif, "EXIF")
                        .on_hover_text("Camera, lens, and capture settings panel");
                    ui.toggle_value(&mut self.show_color_panel, "Color...")
                        .on_hover_text("Open palette + split-tone controls")
                        .clicked();
                });

                settings_row(ui, "Window:", |ui| {
                    let prev_ref = self.reference_mode;
                    ui.toggle_value(&mut self.reference_mode, "On top")
                        .on_hover_text("Always-on-top + borderless (T)");
                    if self.reference_mode != prev_ref {
                        self.apply_reference_mode(ui.ctx());
                    }
                    ui.toggle_value(&mut self.compact, "Compact")
                        .on_hover_text("Hide all panels - just the image (\\)");
                });

                settings_row(ui, "Pinboard:", |ui| {
                    ui.toggle_value(&mut self.pinboard_mode, "Enabled")
                        .on_hover_text("Moodboard canvas - drag, resize, arrange (P)");
                    if self.pinboard_mode
                        && ui
                            .button("Pin selected")
                            .on_hover_text("Add the current image to the pinboard (B)")
                            .clicked()
                    {
                        self.pin_selected_to_board();
                    }
                });

                if self.selected_index.is_some() {
                    settings_row(ui, "File:", |ui| {
                        if ui
                            .button("Rename...")
                            .on_hover_text("Rename the current image (sidecars follow)")
                            .clicked()
                        {
                            self.rename_selected(ui.ctx());
                        }
                        if ui
                            .button("Trash")
                            .on_hover_text("Send the current image to the OS trash (Cmd+Delete)")
                            .clicked()
                        {
                            self.delete_selected_to_trash(ui.ctx());
                        }
                        if self.current_image.is_some()
                            && ui
                                .button("Export...")
                                .on_hover_text(
                                    "Render the image with current effects to a new JPEG/PNG",
                                )
                                .clicked()
                        {
                            let default_name = self
                                .current_image_path
                                .as_ref()
                                .and_then(|p| p.file_stem())
                                .map(|s| format!("{}_export.png", s.to_string_lossy()))
                                .unwrap_or_else(|| "export.png".to_string());
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("PNG", &["png"])
                                .add_filter("JPEG", &["jpg", "jpeg"])
                                .set_file_name(default_name)
                                .save_file()
                            {
                                self.pending_export = Some(path);
                            }
                        }
                    });
                }
            });
    }

    fn show_color_window(&mut self, ctx: &egui::Context) {
        if !self.show_color_panel {
            return;
        }
        let mut open = self.show_color_panel;
        egui::Window::new("Color")
            .open(&mut open)
            .default_width(280.0)
            .show(ctx, |ui| {
                ui.heading("Split-tone");
                ui.add(egui::Slider::new(&mut self.split_tone_amount, 0.0..=1.0).text("Amount"));
                ui.horizontal(|ui| {
                    ui.label("Shadows:");
                    ui.color_edit_button_rgb(&mut self.shadow_tint);
                    ui.label("Highlights:");
                    ui.color_edit_button_rgb(&mut self.highlight_tint);
                });
                if ui.button("Reset to teal/orange").clicked() {
                    self.shadow_tint = [0.20, 0.40, 0.55];
                    self.highlight_tint = [0.95, 0.65, 0.30];
                }

                ui.separator();
                ui.heading("Palette");
                if let Some(image) = self.current_image.as_ref() {
                    if image.palette.is_empty() {
                        ui.label("(no palette)");
                    } else {
                        let palette = image.palette.clone();
                        let mut to_copy: Option<String> = None;
                        ui.horizontal_wrapped(|ui| {
                            for [r, g, b] in &palette {
                                let hex = format!("#{:02X}{:02X}{:02X}", r, g, b);
                                let (rect, resp) = ui.allocate_exact_size(
                                    egui::vec2(36.0, 36.0),
                                    egui::Sense::click(),
                                );
                                ui.painter().rect_filled(
                                    rect,
                                    4.0,
                                    egui::Color32::from_rgb(*r, *g, *b),
                                );
                                let resp = resp.on_hover_text(format!("{} (click to copy)", hex));
                                if resp.clicked() {
                                    to_copy = Some(hex);
                                }
                            }
                        });
                        if let Some(hex) = to_copy {
                            ctx.copy_text(hex.clone());
                            log::debug!("Copied palette swatch: {}", hex);
                        }
                    }
                } else {
                    ui.label("(load an image to see its palette)");
                }
            });
        self.show_color_panel = open;
    }

    fn show_grid_controls(&mut self, ui: &mut egui::Ui) {
        if self.grid_paths.is_empty() {
            if ui.button("Pick images...").clicked()
                && let Some(paths) = rfd::FileDialog::new()
                    .add_filter("Images", &["jpg", "jpeg", "png", "webp", "bmp", "tiff"])
                    .pick_files()
            {
                // The first picked image goes into the primary slot via
                // current_image; tiles 1..=3 come from the rest. If the user
                // picked only 1, treat it as A vs current — promote to a
                // 2-image grid by using the picked file as tile 1.
                let extras: Vec<PathBuf> = paths.into_iter().take(3).collect();
                if !extras.is_empty() {
                    self.start_grid(extras, ui.ctx());
                }
            }
        } else {
            let n_extras = self.grid_paths.len();
            let count = (n_extras + 1).min(4);
            ui.label(format!("Grid: {} tiles", count));
            if ui.button("X").on_hover_text("Exit grid mode").clicked() {
                self.clear_grid();
            }
        }
    }

    fn show_compare_controls(&mut self, ui: &mut egui::Ui) {
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
                    self.start_compare(path, ui.ctx());
                }
            }
            Some(p) => {
                let name = p
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
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

    fn show_status_bar(&mut self, ui: &mut egui::Ui) {
        egui::Panel::bottom("status").show_inside(ui, |ui| {
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
                if self
                    .current_image
                    .as_ref()
                    .is_some_and(|img| img.is_preview)
                {
                    ui.separator();
                    ui.label("Preview");
                }
                if self.export_in_progress {
                    ui.separator();
                    ui.label("Exporting...");
                }
                if let Some(s) = self.cursor_pixel {
                    ui.separator();
                    let [r, g, b, _a] = s.rgba;
                    let swatch_size = egui::vec2(14.0, 14.0);
                    let (rect, _) = ui.allocate_exact_size(swatch_size, egui::Sense::hover());
                    ui.painter()
                        .rect_filled(rect, 2.0, egui::Color32::from_rgb(r, g, b));
                    ui.label(format!(
                        "({}, {})  rgb({}, {}, {})  #{:02X}{:02X}{:02X}",
                        s.x, s.y, r, g, b, r, g, b
                    ));
                }
                if let Some(folder) = &self.folder_path {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add(egui::Label::new(folder.to_string_lossy()).truncate());
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
        let screen = ctx.content_rect();
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

    fn show_pinboard(&mut self, ui: &mut egui::Ui) {
        // A lookup from path -> thumbnail texture so we can render any
        // pinned item even if the user has scrolled the sidebar past it.
        // (The sidebar only requests thumbs for visible rows, so a pin made
        // earlier might no longer have its thumbnail in the file entry.)
        let mut to_request: Vec<PathBuf> = Vec::new();
        let pinboard_id = egui::Id::new("pinboard_canvas");

        egui::CentralPanel::default().show_inside(ui, |ui| {
            let rect = ui.max_rect();
            // Soft fill so the moodboard plane is visually distinct from
            // the single-image viewport.
            ui.painter()
                .rect_filled(rect, 0.0, ui.style().visuals.extreme_bg_color);

            let response = ui.interact(rect, pinboard_id, egui::Sense::click_and_drag());
            let mouse_pos = ui.input(|i| i.pointer.hover_pos());

            // Scroll-to-zoom around the cursor, like the single-image
            // viewport. Keeping it on the same gesture means muscle memory
            // transfers between modes.
            if response.hovered() {
                let scroll_dy = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll_dy != 0.0
                    && let Some(mp) = mouse_pos
                {
                    let world_before = self.pinboard.screen_to_world(mp, rect);
                    let factor = (scroll_dy * 0.005).exp();
                    self.pinboard.view_scale =
                        (self.pinboard.view_scale * factor).clamp(0.05, 20.0);
                    let world_after = self.pinboard.screen_to_world(mp, rect);
                    self.pinboard.view_center += world_before - world_after;
                }
            }

            // Drag-start dispatch: figure out what the user grabbed and
            // record a DragState. We consume `press_origin` rather than the
            // current pointer pos so the hit-test reflects where the drag
            // actually began.
            if response.drag_started() {
                let mp = ui
                    .input(|i| i.pointer.press_origin())
                    .unwrap_or(rect.center());
                let world = self.pinboard.screen_to_world(mp, rect);
                // Corner-handle hit radius is 8 screen pixels expressed in
                // world units so it stays a constant on-screen size.
                let handle_world = 8.0 / self.pinboard.view_scale.max(0.001);
                if let Some((idx, corner)) = self.pinboard.hit_test(world, handle_world) {
                    let raised = self.pinboard.raise(idx);
                    let item = &self.pinboard.items[raised];
                    self.pinboard.drag = match corner {
                        Some(c) => {
                            let aspect = (item.size.x / item.size.y.max(1e-3)).max(0.01);
                            DragState::Resize {
                                idx: raised,
                                corner: c,
                                start_pos: item.pos,
                                start_size: item.size,
                                aspect,
                            }
                        }
                        None => DragState::Move {
                            idx: raised,
                            anchor: world - item.pos,
                        },
                    };
                } else {
                    self.pinboard.selected = None;
                    self.pinboard.drag = DragState::Pan;
                }
            }

            // Drag-in-progress dispatch.
            if response.dragged() {
                match &self.pinboard.drag {
                    DragState::Move { idx, anchor } => {
                        let idx = *idx;
                        let anchor = *anchor;
                        if let Some(mp) = mouse_pos {
                            let world = self.pinboard.screen_to_world(mp, rect);
                            if let Some(item) = self.pinboard.items.get_mut(idx) {
                                item.pos = world - anchor;
                            }
                        }
                    }
                    DragState::Resize {
                        idx,
                        corner,
                        start_pos,
                        start_size,
                        aspect,
                    } => {
                        let (idx, corner, sp, ss, asp) =
                            (*idx, *corner, *start_pos, *start_size, *aspect);
                        if let Some(mp) = mouse_pos {
                            let world = self.pinboard.screen_to_world(mp, rect);
                            let (new_pos, new_size) =
                                pinboard::resize_uniform(corner, sp, ss, asp, world);
                            if let Some(item) = self.pinboard.items.get_mut(idx) {
                                item.pos = new_pos;
                                item.size = new_size;
                            }
                        }
                    }
                    DragState::Pan => {
                        let delta = ui.input(|i| i.pointer.delta());
                        self.pinboard.view_center -= egui::vec2(
                            delta.x / self.pinboard.view_scale.max(0.001),
                            delta.y / self.pinboard.view_scale.max(0.001),
                        );
                    }
                    DragState::None => {}
                }
            }

            if response.drag_stopped() {
                self.pinboard.drag = DragState::None;
            }

            // Render. `items` is z-sorted (back-to-front by Vec order); we
            // paint in iteration order so later items occlude earlier.
            let painter = ui.painter().with_clip_rect(rect);
            let item_count = self.pinboard.items.len();
            for idx in 0..item_count {
                let item = self.pinboard.items[idx].clone();
                let world_rect = egui::Rect::from_min_size(
                    egui::pos2(item.pos.x, item.pos.y),
                    egui::vec2(item.size.x, item.size.y),
                );
                let screen_rect = self.pinboard.world_to_screen_rect(world_rect, rect);

                // Skip drawing if entirely off-screen (cheap culling).
                if !screen_rect.intersects(rect) {
                    continue;
                }

                // Drop shadow for a sleeker look.
                painter.rect_filled(
                    screen_rect.translate(egui::vec2(2.0, 4.0)),
                    4.0,
                    egui::Color32::from_black_alpha(80),
                );

                // Look up the thumbnail by path. If the sidebar hasn't
                // requested it yet (item is below the fold), queue it.
                let entry = self.files.iter().find(|f| f.path == item.path);
                let drew_image = if let Some(e) = entry
                    && let Some(tex) = e.thumbnail.as_ref()
                {
                    painter.image(
                        tex.id(),
                        screen_rect,
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        egui::Color32::WHITE,
                    );
                    true
                } else {
                    painter.rect_filled(screen_rect, 4.0, ui.style().visuals.faint_bg_color);
                    false
                };
                // Schedule a thumbnail request for items whose entry exists
                // but hasn't been thumbnailed yet (not in current view rows).
                if !drew_image
                    && let Some(e) = entry
                    && e.thumbnail.is_none()
                    && !self.requested_thumbnails.contains(&e.path)
                {
                    to_request.push(e.path.clone());
                }

                // Selection chrome: outline + four corner handles.
                if self.pinboard.selected == Some(idx) {
                    painter.rect_stroke(
                        screen_rect,
                        4.0,
                        egui::Stroke::new(2.0, egui::Color32::from_rgb(120, 180, 255)),
                        egui::StrokeKind::Outside,
                    );
                    let handle = 8.0;
                    for p in [
                        screen_rect.left_top(),
                        screen_rect.right_top(),
                        screen_rect.left_bottom(),
                        screen_rect.right_bottom(),
                    ] {
                        let r = egui::Rect::from_center_size(p, egui::vec2(handle, handle));
                        painter.rect_filled(r, 2.0, egui::Color32::WHITE);
                        painter.rect_stroke(
                            r,
                            2.0,
                            egui::Stroke::new(1.0, egui::Color32::from_gray(60)),
                            egui::StrokeKind::Outside,
                        );
                    }
                }
            }

            // Empty-state hint so a fresh pinboard isn't a confusing void.
            if self.pinboard.items.is_empty() {
                painter.text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "Pinboard - select an image and press B to pin it",
                    egui::FontId::proportional(16.0),
                    ui.style().visuals.weak_text_color(),
                );
            }
        });

        // Issue any deferred thumbnail requests for off-screen pins.
        for path in to_request {
            if !self.requested_thumbnails.contains(&path) {
                self.requested_thumbnails.insert(path.clone());
                io::request_thumbnail(path, self.sender.clone(), ui.ctx().clone());
            }
        }
    }

    fn show_viewport(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            if let Some(err) = &self.last_load_error {
                ui.colored_label(egui::Color32::LIGHT_RED, err);
            }

            if self.selected_index.is_none() {
                ui.heading("Viewer");
                ui.label("Select an image to view, or use the arrow keys to navigate.");
                return;
            }

            // If the displayed image's path doesn't match the selected file,
            // a decode is in flight or just failed. Show a spinner only
            // while a request is genuinely pending; if there's nothing
            // queued, fall through so any error label at the top is the
            // only thing the user sees (and the previous image isn't shown).
            let selected_path = self
                .selected_index
                .and_then(|i| self.files.get(i))
                .map(|f| f.path.as_path());
            let stale =
                selected_path.is_some() && selected_path != self.current_image_path.as_deref();
            let waiting = selected_path
                .map(|p| self.pending_image_requests.contains_key(p))
                .unwrap_or(false);
            if stale || self.current_image_size.is_none() {
                ui.centered_and_justified(|ui| {
                    ui.vertical_centered(|ui| {
                        if waiting {
                            ui.add(egui::Spinner::new().size(48.0));
                            ui.label("Loading...");
                        } else if self.last_load_error.is_none() {
                            ui.label("(no image)");
                        }
                    });
                });
                // Keep the egui frame ticking so the spinner animates while
                // the decode worker runs. Without this we'd repaint only on
                // user input and the spinner would freeze.
                if waiting {
                    ui.ctx().request_repaint();
                }
                return;
            }

            let Some([img_w, img_h]) = self.current_image_size else {
                return;
            };
            let preview_active = self
                .current_image
                .as_ref()
                .is_some_and(|img| img.is_preview);
            let img_w = img_w as f32;
            let img_h = img_h as f32;

            let rect = ui.max_rect();
            let response = ui.interact(rect, ui.id(), egui::Sense::drag());
            let screen = rect.size();

            match self.view {
                ViewState::FitOnNextFrame => {
                    let zoom = (screen.x / img_w).min(screen.y / img_h);
                    self.view = ViewState::Manual {
                        zoom,
                        pan: egui::Vec2::ZERO,
                    };
                }
                ViewState::FillOnNextFrame => {
                    let zoom = (screen.x / img_w).max(screen.y / img_h);
                    self.view = ViewState::Manual {
                        zoom,
                        pan: egui::Vec2::ZERO,
                    };
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

            // A/B compare divider drag-on-image: if the press began within
            // 6 screen-px of the divider line, capture this whole drag for
            // divider movement instead of pan / annotation. Re-evaluated
            // only on drag_started; held in `compare_divider_drag` until
            // drag_stopped so the user doesn't lose the grab while moving
            // away from the line.
            if response.drag_started()
                && self.compare_image.is_some()
                && !self.flip_h
                && let Some(press) = ui.input(|i| i.pointer.press_origin())
                && let Some([img_w, _]) = self.current_image_size
            {
                let scale_x = (img_w as f32 * zoom) / screen.x.max(1.0);
                // Image-UV.x = u maps to clip.x = scale*(2u-1) + pan.x,
                // then screen.x = center.x + clip.x * (width / 2).
                let div_clip = scale_x * (2.0 * self.compare_divider - 1.0) + pan.x;
                let div_screen_x = rect.center().x + div_clip * rect.width() * 0.5;
                if (press.x - div_screen_x).abs() <= 6.0 {
                    self.compare_divider_drag = true;
                }
            }
            if response.drag_stopped() {
                self.compare_divider_drag = false;
            }

            if response.dragged() {
                if self.compare_divider_drag
                    && let Some([img_w, _]) = self.current_image_size
                    && let Some(mp) = response.interact_pointer_pos()
                {
                    // Invert the screen->divider mapping above to derive the
                    // new compare_divider from the cursor x.
                    let scale_x = (img_w as f32 * zoom) / screen.x.max(1.0);
                    let clip_x = (mp.x - rect.center().x) / (rect.width() * 0.5);
                    let u = ((clip_x - pan.x) / scale_x.max(1e-3) + 1.0) * 0.5;
                    self.compare_divider = u.clamp(0.0, 1.0);
                } else if !self.annotating {
                    pan += ui.input(|i| i.pointer.delta()) / (screen * 0.5);
                }
            }

            self.view = ViewState::Manual { zoom, pan };

            let grid_active = !self.grid_paths.is_empty();
            // In grid mode the geometry quad fills the viewport (the shader
            // partitions UV into per-tile regions and aspect-fits each
            // image inside its tile). In single-image mode we shape the
            // quad to the source image's aspect, so the texture isn't
            // stretched. Manual rotation 90/270 swaps the displayed aspect
            // (the shader remaps texture UV; the quad's display W/H here
            // must match what the user sees).
            let rotated_quarter = self.manual_rotation == 1 || self.manual_rotation == 3;
            let (display_w, display_h) = if rotated_quarter {
                (img_h, img_w)
            } else {
                (img_w, img_h)
            };
            let scale_unflipped = if grid_active {
                egui::vec2(zoom, zoom)
            } else {
                egui::vec2((display_w * zoom) / screen.x, (display_h * zoom) / screen.y)
            };
            // Negative scale.x mirrors the displayed quad horizontally. Pass
            // this scale to screen_to_uv too so cursor → image-UV inverts
            // correctly under flip (loupe + eyedropper stay coherent).
            let scale = egui::vec2(
                if self.flip_h {
                    -scale_unflipped.x
                } else {
                    scale_unflipped.x
                },
                scale_unflipped.y,
            );
            let mouse_pos = ui.input(|i| i.pointer.hover_pos());

            // Annotation painting: while annotation mode is on, the same
            // viewport drag that would otherwise pan instead deposits brush
            // stamps along the path of the cursor in image-pixel space.
            if self.annotating && !grid_active && !preview_active {
                if response.drag_started() {
                    self.last_paint_pos = None;
                }
                if response.dragged()
                    && let Some(mp) = response.interact_pointer_pos()
                    && let Some(uv) = screen_to_uv(mp, rect, scale, pan)
                {
                    let uv = source_uv_from_display_uv(uv, self.manual_rotation);
                    let img_x = uv.x * img_w;
                    let img_y = uv.y * img_h;
                    if self.annotation.is_none() {
                        self.annotation = Some(AnnotationLayer::new(img_w as u32, img_h as u32));
                        self.annotation_path = self
                            .selected_index
                            .and_then(|i| self.files.get(i))
                            .map(|f| f.path.clone());
                        self.annotation_needs_full_upload = true;
                    }
                    let layer = self.annotation.as_mut().expect("just initialised");
                    let color = [
                        (self.brush_color[0] * 255.0).round().clamp(0.0, 255.0) as u8,
                        (self.brush_color[1] * 255.0).round().clamp(0.0, 255.0) as u8,
                        (self.brush_color[2] * 255.0).round().clamp(0.0, 255.0) as u8,
                        255,
                    ];
                    if let Some(prev) = self.last_paint_pos {
                        layer.stroke(prev, (img_x, img_y), self.brush_radius, color, self.erasing);
                    } else {
                        layer.stamp(img_x, img_y, self.brush_radius, color, self.erasing);
                    }
                    self.last_paint_pos = Some((img_x, img_y));
                }
                if response.drag_stopped() {
                    self.last_paint_pos = None;
                    self.save_current_annotation();
                }
            }

            // Eyedropper: sample the pixel under the cursor. Disabled in
            // grid mode because UV no longer maps 1:1 to a single image.
            self.cursor_pixel = if response.hovered() && !grid_active {
                mouse_pos.and_then(|mouse| {
                    sample_pixel_at(
                        mouse,
                        rect,
                        scale,
                        pan,
                        self.manual_rotation,
                        self.current_image.as_deref()?,
                    )
                })
            } else {
                None
            };

            // Loupe: active when Alt is held and the cursor is over the image.
            // Disabled in grid mode for the same reason as the eyedropper.
            let alt_held = ui.input(|i| i.modifiers.alt);
            let (loupe_active, loupe_center_screen, loupe_center_uv) = if alt_held
                && response.hovered()
                && !grid_active
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
                split_tone_active: u32::from(self.split_tone_amount > 0.0),
                split_tone_amount: self.split_tone_amount,
                clipping_warning: u32::from(self.clipping_warning),
                posterize_active: u32::from(self.value_study),
                posterize_levels: self.value_levels,
                // Annotation overlay sits in image-UV space; in grid mode
                // that space is shared across all tiles, so the strokes
                // would smear. Hide them while comparing.
                annotation_active: u32::from(self.annotation.is_some() && !grid_active),
                grid_active: if grid_active { 1 } else { 0 },
                grid_count: if grid_active {
                    (self.grid_paths.len() + 1).min(4) as u32
                } else {
                    0
                },
                screen_aspect: screen.x / screen.y.max(1.0),
                manual_rotation: self.manual_rotation,
                tile_image_aspects: {
                    // Tile 0 = current image; 1..=3 = grid extras. Falls back
                    // to 1.0 (square) for slots without a loaded image; this
                    // is just a placeholder while the decode is in flight.
                    let mut a = [1.0f32; 4];
                    if let Some([w, h]) = self.current_image_size {
                        a[0] = (w as f32) / (h.max(1) as f32);
                    }
                    for (i, slot) in self.grid_images.iter().enumerate() {
                        if let Some(img) = slot
                            && i < 3
                        {
                            a[i + 1] = (img.width as f32) / (img.height.max(1) as f32);
                        }
                    }
                    a
                },
                shadow_tint: [
                    self.shadow_tint[0],
                    self.shadow_tint[1],
                    self.shadow_tint[2],
                    0.0,
                ],
                highlight_tint: [
                    self.highlight_tint[0],
                    self.highlight_tint[1],
                    self.highlight_tint[2],
                    0.0,
                ],
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

            // Annotation upload: a full re-upload is required when entering a
            // new layer (size mismatch with prior GPU texture). After that,
            // only the dirty rect from this frame's strokes goes up the wire.
            let annotation_upload = if self.annotation_needs_full_upload {
                self.annotation_needs_full_upload = false;
                match &mut self.annotation {
                    Some(layer) => {
                        layer.take_dirty(); // clear so we don't double-upload
                        AnnotationUpload::SetFull {
                            width: layer.width,
                            height: layer.height,
                            rgba: Arc::new(layer.rgba.as_raw().clone()),
                        }
                    }
                    None => AnnotationUpload::Clear,
                }
            } else if let Some(layer) = self.annotation.as_mut() {
                match layer.take_dirty() {
                    Some((rect, data)) => AnnotationUpload::Patch {
                        x: rect.min_x,
                        y: rect.min_y,
                        width: rect.max_x - rect.min_x,
                        height: rect.max_y - rect.min_y,
                        rgba: data,
                    },
                    None => AnnotationUpload::NoChange,
                }
            } else {
                AnnotationUpload::NoChange
            };

            // Grid uploads: one entry per slot 1..=3. Mark dirty -> Set/Clear
            // based on whether we have an image; otherwise NoChange so the
            // GPU keeps the previously-bound texture (or its main fallback).
            let grid_uploads: [GridUpload; 3] = std::array::from_fn(|i| {
                if !self.grid_dirty[i] {
                    return GridUpload::NoChange;
                }
                self.grid_dirty[i] = false;
                match self.grid_images.get(i).and_then(|s| s.clone()) {
                    Some(img) => GridUpload::Set(img),
                    None => GridUpload::Clear,
                }
            });

            let callback = egui_wgpu::Callback::new_paint_callback(
                rect,
                TessellatorCallback {
                    image: upload,
                    compare: compare_upload,
                    annotation: annotation_upload,
                    grid: grid_uploads,
                    settings,
                    format: self.target_format,
                },
            );
            ui.painter().with_clip_rect(rect).add(callback);

            // Crop preview: paint dimming + outline over the displayed image.
            if let Some(aspect) = self.crop_ratio.aspect() {
                let half = egui::vec2(
                    scale_unflipped.x * rect.width() * 0.5,
                    scale_unflipped.y * rect.height() * 0.5,
                );
                let center = egui::pos2(
                    rect.center().x + pan.x * rect.width() * 0.5,
                    rect.center().y + pan.y * rect.height() * 0.5,
                );
                let image_rect =
                    egui::Rect::from_center_size(center, egui::vec2(half.x * 2.0, half.y * 2.0));
                draw_crop_overlay(ui, rect, image_rect, aspect);
            }

            // Histogram overlay sits on top of the viewport in the corner.
            if self.show_histogram
                && let Some(image) = self.current_image.as_ref()
            {
                draw_histogram_overlay(ui, &image.histogram, rect);
            }
        });
    }

    /// Free-floating, draggable EXIF panel. Backed by `egui::Window` so the
    /// user can move it out of the way of the image. Shown when the toolbar
    /// "EXIF" toggle is on and the current image's metadata is available.
    fn show_exif_window(&mut self, ctx: &egui::Context) {
        if !self.show_exif {
            return;
        }
        let Some(image) = self.current_image.clone() else {
            return;
        };
        let exif = &image.exif;

        let mut open = self.show_exif;
        egui::Window::new("EXIF")
            .open(&mut open)
            .default_width(320.0)
            .default_pos(egui::pos2(40.0, 80.0))
            .resizable(true)
            .collapsible(true)
            .show(ctx, |ui| {
                let row = |ui: &mut egui::Ui, label: &str, value: &str| {
                    ui.horizontal(|ui| {
                        ui.add_sized(
                            egui::vec2(70.0, 0.0),
                            egui::Label::new(egui::RichText::new(label).color(egui::Color32::GRAY)),
                        );
                        ui.add(egui::Label::new(value).truncate());
                    });
                };

                row(ui, "Size", &format!("{} x {}", image.width, image.height));
                if let Some(s) = &exif.camera {
                    row(ui, "Camera", s);
                }
                if let Some(s) = &exif.lens {
                    row(ui, "Lens", s);
                }
                if let Some(s) = &exif.focal_length {
                    row(ui, "Focal", s);
                }
                if let Some(s) = &exif.focal_length_35mm {
                    row(ui, "", s);
                }
                if let Some(s) = &exif.aperture {
                    row(ui, "Aperture", s);
                }
                if let Some(s) = &exif.shutter {
                    row(ui, "Shutter", s);
                }
                if let Some(s) = &exif.iso {
                    row(ui, "ISO", s);
                }
                if let Some(s) = &exif.date_taken {
                    row(ui, "Taken", s);
                }
                if exif.is_empty() {
                    ui.weak("(no EXIF data)");
                }
            });
        // Window's close button updates `open`; mirror it back so the toolbar
        // toggle stays in sync.
        self.show_exif = open;
    }
}

fn settings_row(ui: &mut egui::Ui, label: &str, controls: impl FnOnce(&mut egui::Ui)) {
    ui.label(egui::RichText::new(label).color(ui.visuals().hyperlink_color));
    ui.horizontal_wrapped(controls);
    ui.end_row();
}

/// Paint a crop-ratio preview: dim the area outside the centered crop rect
/// (clipped to the displayed image), then stroke a thin outline + thirds.
fn draw_crop_overlay(ui: &egui::Ui, viewport: egui::Rect, image_rect: egui::Rect, aspect: f32) {
    // Fit the largest aspect-correct rect inside image_rect.
    let img_aspect = image_rect.width() / image_rect.height().max(1e-6);
    let (cw, ch) = if aspect >= img_aspect {
        (image_rect.width(), image_rect.width() / aspect)
    } else {
        (image_rect.height() * aspect, image_rect.height())
    };
    let crop = egui::Rect::from_center_size(image_rect.center(), egui::vec2(cw, ch));
    let painter = ui.painter().with_clip_rect(viewport);
    let dim = egui::Color32::from_black_alpha(140);

    // Four bands of the image_rect outside the crop. Skip degenerate rects.
    let bands = [
        egui::Rect::from_min_max(image_rect.min, egui::pos2(image_rect.max.x, crop.min.y)),
        egui::Rect::from_min_max(egui::pos2(image_rect.min.x, crop.max.y), image_rect.max),
        egui::Rect::from_min_max(
            egui::pos2(image_rect.min.x, crop.min.y),
            egui::pos2(crop.min.x, crop.max.y),
        ),
        egui::Rect::from_min_max(
            egui::pos2(crop.max.x, crop.min.y),
            egui::pos2(image_rect.max.x, crop.max.y),
        ),
    ];
    for b in bands {
        if b.width() > 0.5 && b.height() > 0.5 {
            painter.rect_filled(b, 0.0, dim);
        }
    }

    let stroke = egui::Stroke::new(1.5, egui::Color32::from_white_alpha(220));
    painter.rect_stroke(crop, 0.0, stroke, egui::StrokeKind::Inside);
    // Rule-of-thirds inside the crop for composition reference.
    let thin = egui::Stroke::new(1.0, egui::Color32::from_white_alpha(110));
    let third_x1 = crop.left() + crop.width() / 3.0;
    let third_x2 = crop.left() + crop.width() * 2.0 / 3.0;
    let third_y1 = crop.top() + crop.height() / 3.0;
    let third_y2 = crop.top() + crop.height() * 2.0 / 3.0;
    painter.line_segment(
        [
            egui::pos2(third_x1, crop.top()),
            egui::pos2(third_x1, crop.bottom()),
        ],
        thin,
    );
    painter.line_segment(
        [
            egui::pos2(third_x2, crop.top()),
            egui::pos2(third_x2, crop.bottom()),
        ],
        thin,
    );
    painter.line_segment(
        [
            egui::pos2(crop.left(), third_y1),
            egui::pos2(crop.right(), third_y1),
        ],
        thin,
    );
    painter.line_segment(
        [
            egui::pos2(crop.left(), third_y2),
            egui::pos2(crop.right(), third_y2),
        ],
        thin,
    );
}

/// Draw a 256-bin RGB + luma histogram in the top-right corner of `viewport`.
/// Each channel is plotted as an additive translucent fill so overlapping bins
/// blend visibly. Luma drawn last in dim white as a reference curve.
fn draw_histogram_overlay(ui: &egui::Ui, hist: &crate::io::Histogram, viewport: egui::Rect) {
    const PAD: f32 = 12.0;
    const W: f32 = 256.0;
    const H: f32 = 96.0;

    let max = hist.max.max(1) as f32;
    let top_right = viewport.right_top();
    let panel_min = egui::pos2(top_right.x - W - PAD, top_right.y + PAD);
    let panel = egui::Rect::from_min_size(panel_min, egui::vec2(W, H));

    let painter = ui.painter().with_clip_rect(viewport);
    painter.rect_filled(panel.expand(4.0), 4.0, egui::Color32::from_black_alpha(180));

    let plot = |color: egui::Color32, bins: &[u32; 256]| {
        let mut pts: Vec<egui::Pos2> = Vec::with_capacity(256);
        for (i, &v) in bins.iter().enumerate() {
            let x = panel.left() + i as f32;
            let h = (v as f32 / max) * H;
            pts.push(egui::pos2(x, panel.bottom() - h));
        }
        // Filled polygon: down to baseline at both ends, then trace the curve.
        let mut filled: Vec<egui::Pos2> = Vec::with_capacity(258);
        filled.push(egui::pos2(panel.left(), panel.bottom()));
        filled.extend(pts.iter().copied());
        filled.push(egui::pos2(panel.right(), panel.bottom()));
        painter.add(egui::Shape::convex_polygon(
            filled,
            color,
            egui::Stroke::NONE,
        ));
    };

    plot(
        egui::Color32::from_rgba_unmultiplied(255, 60, 60, 130),
        &hist.r,
    );
    plot(
        egui::Color32::from_rgba_unmultiplied(60, 220, 60, 130),
        &hist.g,
    );
    plot(
        egui::Color32::from_rgba_unmultiplied(80, 120, 255, 130),
        &hist.b,
    );
    // Luma overlay as a thin stroke so it sits readably on top of the channel fills.
    let luma_pts: Vec<egui::Pos2> = (0..256)
        .map(|i| {
            let x = panel.left() + i as f32;
            let h = (hist.luma[i] as f32 / max) * H;
            egui::pos2(x, panel.bottom() - h)
        })
        .collect();
    painter.add(egui::Shape::line(
        luma_pts,
        egui::Stroke::new(1.0, egui::Color32::from_white_alpha(200)),
    ));
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
    toggle_flip_h: bool,
    toggle_value_study: bool,
    toggle_annotate: bool,
    toggle_reference_mode: bool,
    toggle_compact: bool,
    toggle_star: bool,
    toggle_starred_filter: bool,
    toggle_pinboard: bool,
    pin_selected: bool,
    delete_pinned: bool,
    rotate_cw: bool,
    rotate_ccw: bool,
    delete_image: bool,
    copy_path: bool,
}

fn read_pressed_keys(ctx: &egui::Context) -> Pressed {
    use egui::{Key, Modifiers};
    // Don't steal Cmd+C if any widget has keyboard focus (text fields, value
    // editors, etc.) - they want it for their own copy behavior.
    let any_focused = ctx.memory(|m| m.focused().is_some());
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
            toggle_flip_h: i.consume_key(Modifiers::NONE, Key::H),
            toggle_value_study: i.consume_key(Modifiers::NONE, Key::V),
            toggle_annotate: i.consume_key(Modifiers::NONE, Key::A),
            toggle_reference_mode: i.consume_key(Modifiers::NONE, Key::T),
            toggle_compact: i.consume_key(Modifiers::NONE, Key::Backslash),
            toggle_star: i.consume_key(Modifiers::NONE, Key::S),
            toggle_starred_filter: i.consume_key(Modifiers::SHIFT, Key::S),
            toggle_pinboard: i.consume_key(Modifiers::NONE, Key::P),
            pin_selected: i.consume_key(Modifiers::NONE, Key::B),
            delete_pinned: i.consume_key(Modifiers::NONE, Key::Delete)
                || i.consume_key(Modifiers::NONE, Key::Backspace),
            rotate_cw: i.consume_key(Modifiers::NONE, Key::R),
            rotate_ccw: i.consume_key(Modifiers::SHIFT, Key::R),
            // Cmd-Delete (or Cmd-Backspace on Mac) sends the current
            // image to the OS trash. Plain Delete is reserved for the
            // pinboard remove-pinned action and would be too easy to
            // mis-trigger here.
            delete_image: i.consume_key(Modifiers::COMMAND, Key::Delete)
                || i.consume_key(Modifiers::COMMAND, Key::Backspace),
            copy_path: !any_focused && i.consume_key(Modifiers::COMMAND, Key::C),
        }
    })
}

/// Pure transition function for the direction-streak state machine. Extracted
/// from `update_direction_streak` so it can be tested without an `App` instance.
fn next_streak(prev_streak: i32, signed_step: i32) -> i32 {
    if signed_step == 0 {
        0
    } else if signed_step.signum() == prev_streak.signum() {
        prev_streak.saturating_add(signed_step)
    } else {
        // Direction change (or fresh start from 0): begin a new streak.
        signed_step
    }
}

fn compute_preload_window(current: usize, len: usize) -> Vec<usize> {
    let mut window = Vec::with_capacity(8);
    if current >= len {
        return window;
    }
    for distance in 1..=4 {
        let next = current + distance;
        if next < len {
            window.push(next);
        }
        if let Some(prev) = current.checked_sub(distance) {
            window.push(prev);
        }
    }
    window
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

fn source_uv_from_display_uv(uv: egui::Vec2, rotation: u32) -> egui::Vec2 {
    match rotation & 3 {
        1 => egui::vec2(uv.y, 1.0 - uv.x),
        2 => egui::vec2(1.0 - uv.x, 1.0 - uv.y),
        3 => egui::vec2(1.0 - uv.y, uv.x),
        _ => uv,
    }
}

fn sample_pixel_at(
    mouse: egui::Pos2,
    rect: egui::Rect,
    scale: egui::Vec2,
    pan: egui::Vec2,
    rotation: u32,
    image: &DecodedImage,
) -> Option<CursorSample> {
    let uv = source_uv_from_display_uv(screen_to_uv(mouse, rect, scale, pan)?, rotation);

    let mx = ((uv.x * image.width as f32) as u32).min(image.width.saturating_sub(1));
    let my = ((uv.y * image.height as f32) as u32).min(image.height.saturating_sub(1));
    let idx = (my as usize * image.width as usize + mx as usize) * 4;
    let rgba = [
        *image.rgba.get(idx)?,
        *image.rgba.get(idx + 1)?,
        *image.rgba.get(idx + 2)?,
        *image.rgba.get(idx + 3)?,
    ];
    Some(CursorSample { x: mx, y: my, rgba })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(w: f32, h: f32) -> egui::Rect {
        egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w, h))
    }

    #[test]
    fn screen_to_uv_center_at_identity() {
        // scale=1, pan=0: center of screen → center of image.
        let uv = screen_to_uv(
            egui::pos2(100.0, 100.0),
            rect(200.0, 200.0),
            egui::vec2(1.0, 1.0),
            egui::Vec2::ZERO,
        )
        .unwrap();
        assert!((uv.x - 0.5).abs() < 1e-5);
        assert!((uv.y - 0.5).abs() < 1e-5);
    }

    #[test]
    fn screen_to_uv_outside_returns_none() {
        // scale=0.5: image fills only half the screen. The screen's top-left
        // corner is well outside the image bounds.
        let uv = screen_to_uv(
            egui::pos2(0.0, 0.0),
            rect(200.0, 200.0),
            egui::vec2(0.5, 0.5),
            egui::Vec2::ZERO,
        );
        assert!(uv.is_none());
    }

    #[test]
    fn screen_to_uv_pan_shifts_origin() {
        // pan.x = -0.5 shifts the image left in NDC. Center of screen now
        // sees a UV further toward the right of the image.
        // vx = (0 - (-0.5)) / 1 = 0.5  →  u = (0.5 + 1) / 2 = 0.75
        let uv = screen_to_uv(
            egui::pos2(100.0, 100.0),
            rect(200.0, 200.0),
            egui::vec2(1.0, 1.0),
            egui::vec2(-0.5, 0.0),
        )
        .unwrap();
        assert!((uv.x - 0.75).abs() < 1e-5, "got {}", uv.x);
        assert!((uv.y - 0.5).abs() < 1e-5);
    }

    #[test]
    fn screen_to_uv_zoom_keeps_center_at_center() {
        // 2x zoom expands the image off-screen but the center pixel still maps
        // to UV (0.5, 0.5).
        let uv = screen_to_uv(
            egui::pos2(100.0, 100.0),
            rect(200.0, 200.0),
            egui::vec2(2.0, 2.0),
            egui::Vec2::ZERO,
        )
        .unwrap();
        assert!((uv.x - 0.5).abs() < 1e-5);
        assert!((uv.y - 0.5).abs() < 1e-5);
    }

    #[test]
    fn source_uv_rotation_matches_shader_mapping() {
        let uv = egui::vec2(0.25, 0.75);
        assert_eq!(source_uv_from_display_uv(uv, 0), egui::vec2(0.25, 0.75));
        assert_eq!(source_uv_from_display_uv(uv, 1), egui::vec2(0.75, 0.75));
        assert_eq!(source_uv_from_display_uv(uv, 2), egui::vec2(0.75, 0.25));
        assert_eq!(source_uv_from_display_uv(uv, 3), egui::vec2(0.25, 0.25));
    }

    #[test]
    fn streak_starts_from_zero() {
        assert_eq!(next_streak(0, 1), 1);
        assert_eq!(next_streak(0, -1), -1);
        assert_eq!(next_streak(0, 0), 0);
    }

    #[test]
    fn streak_extends_in_same_direction() {
        assert_eq!(next_streak(1, 1), 2);
        assert_eq!(next_streak(5, 1), 6);
        assert_eq!(next_streak(-3, -1), -4);
    }

    #[test]
    fn streak_resets_on_reverse() {
        assert_eq!(next_streak(5, -1), -1);
        assert_eq!(next_streak(-2, 1), 1);
    }

    #[test]
    fn streak_collapses_on_pause() {
        assert_eq!(next_streak(10, 0), 0);
        assert_eq!(next_streak(-7, 0), 0);
    }

    #[test]
    fn streak_saturates_at_i32_bounds() {
        assert_eq!(next_streak(i32::MAX, 1), i32::MAX);
        assert_eq!(next_streak(i32::MIN, -1), i32::MIN);
    }

    #[test]
    fn preload_window_loads_four_before_and_after() {
        assert_eq!(compute_preload_window(5, 12), vec![6, 4, 7, 3, 8, 2, 9, 1]);
    }

    #[test]
    fn preload_window_clamps_at_edges() {
        assert_eq!(compute_preload_window(0, 5), vec![1, 2, 3, 4]);
        assert_eq!(compute_preload_window(4, 5), vec![3, 2, 1, 0]);
    }

    #[test]
    fn preload_window_ignores_out_of_range_current() {
        assert!(compute_preload_window(5, 5).is_empty());
    }
}
