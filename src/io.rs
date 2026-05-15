// ABOUTME: Background I/O for folder scanning and image decoding.
// ABOUTME: Workers send Message variants back to the UI thread via crossbeam.

use crossbeam_channel::Sender;
use eframe::egui;
use image::{DynamicImage, ImageDecoder};
use memmap2::Mmap;
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Cooperative cancellation handle shared between the app (which decides to
/// cancel) and the rayon worker (which polls at known checkpoints). Cheap to
/// clone; under the hood is an `Arc<AtomicBool>`.
#[derive(Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// Files smaller than this don't benefit from mmap (page-table overhead and
/// minimum mapping size dominate). Read into a Vec instead.
const MMAP_MIN_BYTES: u64 = 256 * 1024;

/// File contents held either in a heap buffer or as a memory map.
enum FileBytes {
    Heap(Vec<u8>),
    Mapped(Mmap),
}

impl FileBytes {
    fn as_slice(&self) -> &[u8] {
        match self {
            FileBytes::Heap(v) => v.as_slice(),
            FileBytes::Mapped(m) => &m[..],
        }
    }
}

/// Load file bytes, preferring memory-mapped I/O. Falls back to a regular read
/// for small files, network-mounted paths (where SIGBUS on disconnect is not
/// catchable in safe Rust), or any mmap error.
fn read_file_bytes(path: &Path) -> std::io::Result<FileBytes> {
    let mut file = File::open(path)?;
    let size = file.metadata()?.len();

    if size < MMAP_MIN_BYTES || is_likely_network_path(path) {
        return read_to_vec(&mut file, size).map(FileBytes::Heap);
    }

    // SAFETY: Unsafe because the kernel can't guarantee another process
    // won't truncate the file mid-read (SIGBUS). For an image viewer reading
    // local files that the user just selected, this risk is acceptable. The
    // network-share check above handles the most common SIGBUS scenario.
    match unsafe { Mmap::map(&file) } {
        Ok(mmap) => Ok(FileBytes::Mapped(mmap)),
        Err(e) => {
            log::warn!(
                "mmap failed for {:?}, falling back to buffered read: {}",
                path,
                e
            );
            read_to_vec(&mut file, size).map(FileBytes::Heap)
        }
    }
}

fn read_to_vec(file: &mut File, size_hint: u64) -> std::io::Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(size_hint as usize);
    file.read_to_end(&mut buf)?;
    Ok(buf)
}

/// Returns the total physical memory of the system in bytes.
/// Falls back to a sensible default (8GB) if detection fails.
pub fn get_total_memory() -> usize {
    #[cfg(unix)]
    {
        let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if pages > 0 && page_size > 0 {
            return (pages as usize).saturating_mul(page_size as usize);
        }
    }

    #[cfg(windows)]
    {
        use std::mem;
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

        let mut status: MEMORYSTATUSEX = unsafe { mem::zeroed() };
        status.dwLength = mem::size_of::<MEMORYSTATUSEX>() as u32;
        if unsafe { GlobalMemoryStatusEx(&mut status) } != 0 {
            return status.ullTotalPhys as usize;
        }
    }

    // Default to 8GB if detection fails.
    8 * 1024 * 1024 * 1024
}

/// Best-effort detection of network-mounted paths. False negatives mean we
/// take the SIGBUS risk; false positives mean a slightly slower buffered read.
#[cfg(unix)]
fn is_likely_network_path(path: &Path) -> bool {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let Ok(cpath) = CString::new(path.as_os_str().as_bytes()) else {
        return false;
    };

    #[cfg(target_os = "macos")]
    {
        let mut buf: libc::statfs = unsafe { std::mem::zeroed() };
        if unsafe { libc::statfs(cpath.as_ptr(), &mut buf) } != 0 {
            return false;
        }
        let raw: &[u8] = unsafe {
            std::slice::from_raw_parts(
                buf.f_fstypename.as_ptr() as *const u8,
                buf.f_fstypename.len(),
            )
        };
        let len = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
        let name = std::str::from_utf8(&raw[..len]).unwrap_or("");
        matches!(name, "nfs" | "smbfs" | "afpfs" | "webdav" | "ftp")
    }

    #[cfg(target_os = "linux")]
    {
        let mut buf: libc::statfs = unsafe { std::mem::zeroed() };
        if unsafe { libc::statfs(cpath.as_ptr(), &mut buf) } != 0 {
            return false;
        }
        // Linux statfs uses a magic number for the filesystem type.
        // Constants from <linux/magic.h>
        match buf.f_type as u32 {
            0x6969 => true,     // NFS_SUPER_MAGIC
            0x517B => true,     // SMB_SUPER_MAGIC
            0x564C => true,     // NCP_SUPER_MAGIC (NetWare)
            0xFE534D42 => true, // SMB2_MAGIC_NUMBER
            0x9753 => true,     // CIFS_MAGIC_NUMBER
            0x1173 => true,     // CODA_SUPER_MAGIC
            _ => false,
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        false
    }
}

#[cfg(windows)]
fn is_likely_network_path(path: &Path) -> bool {
    use std::os::windows::ffi::OsStrExt;
    use std::ptr::null_mut;

    let path_str = path.to_string_lossy();
    // UNC paths (\\server\share) are definitely network paths.
    if path_str.starts_with("\\\\") {
        return true;
    }

    // Check drive type for mapped network drives (Z:\...).
    let mut wide: Vec<u16> = path.as_os_str().encode_wide().collect();
    wide.push(0);

    // GetDriveTypeW needs a root path like "C:\".
    // We try to find the root part of the path.
    if let Some(root) = path.components().next().and_then(|c| match c {
        std::path::Component::Prefix(p) => Some(p.as_os_str()),
        _ => None,
    }) {
        let mut root_wide: Vec<u16> = root.encode_wide().collect();
        // Ensure it ends with a backslash.
        if !root_wide.ends_with(&[b'\\' as u16]) {
            root_wide.push(b'\\' as u16);
        }
        root_wide.push(0);

        unsafe {
            let drive_type =
                windows_sys::Win32::Storage::FileSystem::GetDriveTypeW(root_wide.as_ptr());
            drive_type == windows_sys::Win32::Storage::FileSystem::DRIVE_REMOTE
        }
    } else {
        false
    }
}

/// Convert a DynamicImage to an RgbaImage, consuming the buffer when the
/// source is already RGBA8. For other variants we still pay for the copy
/// (alpha needs to be added, sample width converted, etc.).
fn to_rgba8_consuming(img: DynamicImage) -> image::RgbaImage {
    match img {
        DynamicImage::ImageRgba8(buf) => buf,
        other => other.to_rgba8(),
    }
}

fn is_jpeg(bytes: &[u8]) -> bool {
    bytes.starts_with(&[0xFF, 0xD8, 0xFF])
}

/// Fast-path JPEG decode that uses libjpeg-style DCT scaling. For a 24MP
/// source and a 128 px thumbnail target this is roughly 64x faster than
/// decoding the full raster. Returns `None` for non-RGB/L8 pixel formats
/// (CMYK, 16-bit grayscale) so the caller can fall back. Returns the
/// scaled image and the *original* (pre-scale) image dimensions so callers
/// can report the source resolution to the user.
fn try_decode_jpeg_scaled(bytes: &[u8], target_min_dim: u16) -> Option<(DynamicImage, (u32, u32))> {
    use jpeg_decoder::{Decoder as JpegDecoder, PixelFormat};

    let mut decoder = JpegDecoder::new(Cursor::new(bytes));
    decoder.read_info().ok()?;
    // Capture the source dimensions before scaling - decoder.info() reflects
    // the requested scale after `scale()` is called.
    let pre_info = decoder.info()?;
    let original_dims = (pre_info.width as u32, pre_info.height as u32);

    // Snap the request to the closest 1/N (N in {1, 2, 4, 8}) DCT scale.
    decoder.scale(target_min_dim, target_min_dim).ok()?;

    let pixels = decoder.decode().ok()?;
    let info = decoder.info()?;
    let w = info.width as u32;
    let h = info.height as u32;

    let scaled = match info.pixel_format {
        PixelFormat::RGB24 => {
            // Promote to RGBA so downstream paths (thumbnail egui::ColorImage,
            // viewport upload) stay on a single pixel format.
            let mut rgba = Vec::with_capacity(pixels.len() / 3 * 4);
            for px in pixels.chunks_exact(3) {
                rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
            }
            image::RgbaImage::from_raw(w, h, rgba).map(DynamicImage::ImageRgba8)?
        }
        PixelFormat::L8 => {
            image::GrayImage::from_raw(w, h, pixels).map(DynamicImage::ImageLuma8)?
        }
        // CMYK32, L16: rare. Fall back to the full image-crate path.
        _ => return None,
    };
    Some((scaled, original_dims))
}

/// Read EXIF orientation directly from JPEG bytes without a full decode.
/// Returns `NoTransforms` if the file has no EXIF or the field is missing.
fn read_exif_orientation(bytes: &[u8]) -> image::metadata::Orientation {
    use exif::{In, Reader, Tag};

    let reader = Reader::new();
    let Ok(data) = reader.read_from_container(&mut Cursor::new(bytes)) else {
        return image::metadata::Orientation::NoTransforms;
    };
    let value = data
        .get_field(Tag::Orientation, In::PRIMARY)
        .and_then(|f| f.value.get_uint(0))
        .unwrap_or(1);
    image::metadata::Orientation::from_exif(value as u8)
        .unwrap_or(image::metadata::Orientation::NoTransforms)
}

/// Human-readable EXIF metadata fields surfaced in the UI panel. All
/// strings are pre-formatted on the worker so the render thread just
/// shows them. `None` for any field means the file didn't carry that tag.
#[derive(Default, Clone, Debug)]
pub struct ExifMetadata {
    pub camera: Option<String>,
    pub lens: Option<String>,
    pub iso: Option<String>,
    pub aperture: Option<String>,
    pub shutter: Option<String>,
    pub focal_length: Option<String>,
    pub focal_length_35mm: Option<String>,
    pub date_taken: Option<String>,
}

impl ExifMetadata {
    pub fn is_empty(&self) -> bool {
        self.camera.is_none()
            && self.lens.is_none()
            && self.iso.is_none()
            && self.aperture.is_none()
            && self.shutter.is_none()
            && self.focal_length.is_none()
            && self.focal_length_35mm.is_none()
            && self.date_taken.is_none()
    }
}

/// Parse the subset of EXIF tags we display. Returns an empty struct for
/// containers without EXIF (PNG without chunks, etc).
fn read_exif_metadata(bytes: &[u8]) -> ExifMetadata {
    use exif::{In, Reader, Tag, Value};

    let reader = Reader::new();
    let Ok(data) = reader.read_from_container(&mut Cursor::new(bytes)) else {
        return ExifMetadata::default();
    };

    let display = |tag: Tag| -> Option<String> {
        let f = data.get_field(tag, In::PRIMARY)?;
        let s = f.display_value().with_unit(&data).to_string();
        let s = s.trim().trim_matches('"').to_string();
        if s.is_empty() { None } else { Some(s) }
    };

    let rational_as_string = |tag: Tag| -> Option<String> {
        let f = data.get_field(tag, In::PRIMARY)?;
        match &f.value {
            Value::Rational(v) if !v.is_empty() => {
                let r = v[0];
                if r.denom == 0 {
                    return None;
                }
                Some(format!("{}", r.to_f64()))
            }
            _ => None,
        }
    };

    let shutter = data
        .get_field(Tag::ExposureTime, In::PRIMARY)
        .and_then(|f| {
            if let Value::Rational(v) = &f.value
                && let Some(r) = v.first()
                && r.denom != 0
            {
                return Some(if r.num == 0 {
                    "0 s".to_string()
                } else if r.num >= r.denom {
                    format!("{:.1} s", r.to_f64())
                } else {
                    format!("1/{} s", (r.denom as f64 / r.num as f64).round() as u64)
                });
            }
            None
        });

    let aperture = rational_as_string(Tag::FNumber).map(|s| format!("f/{}", s));
    let focal_length = rational_as_string(Tag::FocalLength).map(|s| format!("{} mm", s));
    let focal_length_35mm =
        display(Tag::FocalLengthIn35mmFilm).map(|s| format!("{} mm (35mm eq)", s));

    let date_taken = display(Tag::DateTimeOriginal).or_else(|| display(Tag::DateTime));

    let make = display(Tag::Make);
    let model = display(Tag::Model);
    let camera = match (make, model) {
        (Some(make), Some(model)) => {
            if model.starts_with(&make) {
                Some(model)
            } else {
                Some(format!("{} {}", make, model))
            }
        }
        (Some(s), None) | (None, Some(s)) => Some(s),
        (None, None) => None,
    };

    ExifMetadata {
        camera,
        lens: display(Tag::LensModel),
        iso: display(Tag::PhotographicSensitivity),
        aperture,
        shutter,
        focal_length,
        focal_length_35mm,
        date_taken,
    }
}

/// A fully prepared image: dimensions plus the base RGBA buffer. All
/// heavy CPU work (decode, EXIF rotate, RGBA convert, histogram,
/// palette extraction) is done on the worker thread that produces
/// this so the render thread only needs to memcpy bytes to the GPU.
/// Mipmaps are generated on the GPU.
pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    pub histogram: Histogram,
    pub palette: Vec<[u8; 3]>,
    pub exif: ExifMetadata,
    pub is_preview: bool,
}

impl DecodedImage {
    /// Approximate footprint of the decoded image in bytes.
    pub fn byte_size(&self) -> usize {
        self.rgba.len()
    }
}

/// 256-bucket histograms for each channel + luminance, plus the max bin count
/// across all channels (used to scale the on-screen overlay).
#[derive(Clone)]
pub struct Histogram {
    pub r: [u32; 256],
    pub g: [u32; 256],
    pub b: [u32; 256],
    pub luma: [u32; 256],
    pub max: u32,
}

impl Default for Histogram {
    fn default() -> Self {
        Self {
            r: [0; 256],
            g: [0; 256],
            b: [0; 256],
            luma: [0; 256],
            max: 0,
        }
    }
}

/// Build per-channel + luma histograms from raw RGBA bytes using strided
/// sampling for speed on large images.
fn compute_histogram(rgba: &[u8], width: u32, height: u32) -> Histogram {
    let mut h = Histogram::default();
    // Aim for ~250k samples.
    let total_pixels = (width as u64) * (height as u64);
    let stride = (total_pixels / 250_000).max(1) as usize;

    for px in rgba.chunks_exact(4).step_by(stride) {
        h.r[px[0] as usize] += 1;
        h.g[px[1] as usize] += 1;
        h.b[px[2] as usize] += 1;
        // Luma via the same coefficients used by the grayscale shader path.
        let l = (px[0] as u32 * 299 + px[1] as u32 * 587 + px[2] as u32 * 114) / 1000;
        h.luma[l.min(255) as usize] += 1;
    }
    h.max =
        h.r.iter()
            .chain(h.g.iter())
            .chain(h.b.iter())
            .chain(h.luma.iter())
            .copied()
            .max()
            .unwrap_or(0);
    h
}

/// Median-cut palette extraction. Splits the color box repeatedly along the
/// widest channel until `target_count` boxes exist, then averages each box to
/// produce a palette color. Pixel-frequency-weighted (no dedupe).
/// Uses strided sampling for speed on large images.
fn extract_palette(rgba: &[u8], width: u32, height: u32, target_count: usize) -> Vec<[u8; 3]> {
    if rgba.len() < 4 || target_count == 0 {
        return Vec::new();
    }
    // Aim for ~16k samples for palette extraction.
    let total_pixels = (width as u64) * (height as u64);
    let stride = (total_pixels / 16_000).max(1) as usize;

    let mut pixels: Vec<[u8; 3]> = rgba
        .chunks_exact(4)
        .step_by(stride)
        .map(|p| [p[0], p[1], p[2]])
        .collect();
    if pixels.is_empty() {
        return Vec::new();
    }

    // Index ranges into `pixels` so we can mutably sort each slice without
    // juggling overlapping borrows.
    let mut boxes: Vec<(usize, usize)> = vec![(0, pixels.len())];

    while boxes.len() < target_count {
        // Find the splittable box with the widest channel range.
        let pick = boxes
            .iter()
            .enumerate()
            .filter(|(_, (s, e))| e - s > 1)
            .map(|(idx, &(s, e))| {
                let slice = &pixels[s..e];
                let (channel, range) = widest_channel(slice);
                (idx, channel, range)
            })
            .max_by_key(|&(_, _, r)| r);
        let Some((idx, channel, _)) = pick else { break };

        let (s, e) = boxes.swap_remove(idx);
        let slice = &mut pixels[s..e];
        slice.sort_by_key(|p| p[channel]);
        let mid = s + slice.len() / 2;
        boxes.push((s, mid));
        boxes.push((mid, e));
    }

    let mut palette: Vec<[u8; 3]> = boxes
        .iter()
        .map(|&(s, e)| average_color(&pixels[s..e]))
        .collect();
    // Sort by luma so the displayed swatches feel ordered.
    palette.sort_by_key(|p| (p[0] as u32 * 299 + p[1] as u32 * 587 + p[2] as u32 * 114) as i32);
    palette
}

/// Returns (channel index 0..3, range) for the channel with the widest spread.
fn widest_channel(pixels: &[[u8; 3]]) -> (usize, u8) {
    let mut mn = [u8::MAX; 3];
    let mut mx = [0u8; 3];
    for p in pixels {
        for c in 0..3 {
            if p[c] < mn[c] {
                mn[c] = p[c];
            }
            if p[c] > mx[c] {
                mx[c] = p[c];
            }
        }
    }
    let mut best = (0, 0u8);
    for c in 0..3 {
        let range = mx[c].saturating_sub(mn[c]);
        if range > best.1 {
            best = (c, range);
        }
    }
    best
}

fn average_color(pixels: &[[u8; 3]]) -> [u8; 3] {
    if pixels.is_empty() {
        return [0, 0, 0];
    }
    let mut sum = [0u64; 3];
    for p in pixels {
        sum[0] += p[0] as u64;
        sum[1] += p[1] as u64;
        sum[2] += p[2] as u64;
    }
    let n = pixels.len() as u64;
    [(sum[0] / n) as u8, (sum[1] / n) as u8, (sum[2] / n) as u8]
}

/// Decode an image, honoring EXIF orientation when the format provides it
/// (notably JPEG). Without this, phone photos display rotated.
///
/// File contents are loaded via memory-mapped I/O when possible, avoiding the
/// page-cache → user-space copy that the standard `BufReader` path triggers.
fn decode_image_from_bytes(slice: &[u8]) -> image::ImageResult<DynamicImage> {
    let cursor = Cursor::new(slice);
    let reader = image::ImageReader::new(cursor)
        .with_guessed_format()
        .map_err(image::ImageError::IoError)?;
    let mut decoder = reader.into_decoder()?;
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut img = DynamicImage::from_decoder(decoder)?;
    img.apply_orientation(orientation);
    Ok(img)
}

/// Decode + EXIF rotate + RGBA convert + full mip chain. All heavy CPU work is
/// concentrated here so the render thread sees only `write_texture` calls.
/// Errors a decode worker can produce. `Cancelled` is informational only -
/// the app removes its pending entry at cancel time, so cancelled workers
/// silently return without sending any message.
enum DecodeError {
    Image(image::ImageError),
    Cancelled,
}

impl From<image::ImageError> for DecodeError {
    fn from(e: image::ImageError) -> Self {
        DecodeError::Image(e)
    }
}

fn decode_image_prepared(path: &Path, cancel: &CancelToken) -> Result<DecodedImage, DecodeError> {
    let bytes =
        read_file_bytes(path).map_err(|e| DecodeError::Image(image::ImageError::IoError(e)))?;
    let exif = read_exif_metadata(bytes.as_slice());
    if cancel.is_cancelled() {
        return Err(DecodeError::Cancelled);
    }
    let img = decode_image_from_bytes(bytes.as_slice())?;
    drop(bytes);
    if cancel.is_cancelled() {
        return Err(DecodeError::Cancelled);
    }
    let rgba = to_rgba8_consuming(img);
    if cancel.is_cancelled() {
        return Err(DecodeError::Cancelled);
    }
    let (width, height) = rgba.dimensions();

    // Color analysis: histogram from strided samples, palette from even
    // coarser samples.
    if cancel.is_cancelled() {
        return Err(DecodeError::Cancelled);
    }
    let histogram = compute_histogram(rgba.as_raw(), width, height);
    if cancel.is_cancelled() {
        return Err(DecodeError::Cancelled);
    }
    let palette = extract_palette(rgba.as_raw(), width, height, 8);

    Ok(DecodedImage {
        width,
        height,
        rgba: rgba.into_raw(),
        histogram,
        palette,
        exif,
        is_preview: false,
    })
}

const LARGE_IMAGE_PREVIEW_MIN_FILE_BYTES: u64 = 12 * 1024 * 1024;
const LARGE_IMAGE_PREVIEW_TARGET: u16 = 2048;

fn decode_large_jpeg_preview(path: &Path, cancel: &CancelToken) -> Option<DecodedImage> {
    let bytes = read_file_bytes(path).ok()?;
    let slice = bytes.as_slice();
    if !is_jpeg(slice) {
        return None;
    }
    if cancel.is_cancelled() {
        return None;
    }
    let exif = read_exif_metadata(slice);
    let (mut img, original_dims) = try_decode_jpeg_scaled(slice, LARGE_IMAGE_PREVIEW_TARGET)?;
    let orientation = read_exif_orientation(slice);
    img.apply_orientation(orientation);
    let original_dims = orient_dims(original_dims, orientation);
    let rgba = to_rgba8_consuming(img);
    if cancel.is_cancelled() {
        return None;
    }
    let (width, height) = rgba.dimensions();
    if (width, height) == original_dims {
        return None;
    }
    let histogram = compute_histogram(rgba.as_raw(), width, height);
    let palette = extract_palette(rgba.as_raw(), width, height, 8);
    Some(DecodedImage {
        width,
        height,
        rgba: rgba.into_raw(),
        histogram,
        palette,
        exif,
        is_preview: true,
    })
}

/// Why an image was decoded. `Display` results drive the viewport (subject to
/// generation matching); `Preload` results only populate the cache; `Compare`
/// results populate the right-side compare slot.
#[derive(Debug, Clone, Copy)]
pub enum ImagePurpose {
    Display {
        generation: u64,
    },
    PreviewDisplay {
        generation: u64,
    },
    Preload,
    Compare {
        generation: u64,
    },
    /// Multi-image grid mode tile. `slot` is 1..=3 (slot 0 is the main image).
    Grid {
        slot: u32,
        generation: u64,
    },
}

pub enum Message {
    FilesFound {
        generation: u64,
        files: Vec<ScannedFile>,
    },
    ThumbnailLoaded {
        path: PathBuf,
        image: egui::ColorImage,
        /// Original image dimensions (post-EXIF-orientation) so the sidebar
        /// can show e.g. "4032 x 3024" without a separate header read.
        source_dims: (u32, u32),
    },
    ThumbnailFailed,
    ImageDecoded {
        path: PathBuf,
        image: Arc<DecodedImage>,
        purpose: ImagePurpose,
    },
    ImageFailed {
        path: PathBuf,
        error: String,
        purpose: ImagePurpose,
    },
    /// A change was detected in the watched folder. Debounced and re-scanned
    /// by the app, not directly handled here.
    FolderChanged,
    ExportFinished {
        path: PathBuf,
        result: Result<(), String>,
    },
}

const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "webp", "bmp", "tiff"];

pub fn is_image_file(path: &Path) -> bool {
    path.extension()
        .and_then(|s| s.to_str())
        .map(|s| IMAGE_EXTENSIONS.contains(&s.to_lowercase().as_str()))
        .unwrap_or(false)
}

pub struct ScannedFile {
    pub path: PathBuf,
    pub size_bytes: u64,
}

pub fn scan_folder(
    path: PathBuf,
    max_depth: usize,
    generation: u64,
    cancel: CancelToken,
    sender: Sender<Message>,
    ctx: egui::Context,
) {
    // I/O bound, run on a one-off thread to avoid tying up Rayon's CPU pool.
    std::thread::spawn(move || {
        let mut found = Vec::new();
        for entry in walkdir::WalkDir::new(&path)
            .max_depth(max_depth)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            if cancel.is_cancelled() {
                log::debug!("Folder scan cancelled: {:?} (gen {})", path, generation);
                return;
            }
            if is_image_file(entry.path()) {
                let size_bytes = entry.metadata().map(|m| m.len()).unwrap_or(0);
                found.push(ScannedFile {
                    path: entry.path().to_path_buf(),
                    size_bytes,
                });
            }
        }
        // Case-insensitive sort by filename. `sort_by_cached_key` allocates
        // each lowercase key exactly once, vs. once per comparison.
        found.sort_by_cached_key(|f| {
            f.path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_lowercase()
        });
        log::info!(
            "Folder scan complete: {} images under {:?} (depth {})",
            found.len(),
            path,
            max_depth
        );
        if cancel.is_cancelled() {
            log::debug!(
                "Folder scan cancelled after walk: {:?} (gen {})",
                path,
                generation
            );
            return;
        }
        let _ = sender.send(Message::FilesFound {
            generation,
            files: found,
        });
        ctx.request_repaint();
    });
}

pub fn request_thumbnail(path: PathBuf, sender: Sender<Message>, ctx: egui::Context) {
    rayon::spawn(move || {
        // Thumbnails don't need mips - they're sampled near 1:1 by egui.
        match decode_thumbnail(&path) {
            Some((thumbnail, source_dims)) => {
                let size = [thumbnail.width() as usize, thumbnail.height() as usize];
                let pixels = to_rgba8_consuming(thumbnail);
                let color_image = egui::ColorImage::from_rgba_unmultiplied(
                    size,
                    pixels.as_flat_samples().as_slice(),
                );
                let _ = sender.send(Message::ThumbnailLoaded {
                    path,
                    image: color_image,
                    source_dims,
                });
            }
            None => {
                log::warn!("Thumbnail decode failed for {:?}", path);
                let _ = sender.send(Message::ThumbnailFailed);
            }
        }
        ctx.request_repaint();
    });
}

/// Produce a thumbnail-sized DynamicImage along with the source image's
/// orientation-corrected dimensions. Tries the JPEG DCT fast-path first
/// and falls back to the full image-crate decode for non-JPEG or unsupported
/// JPEG variants. EXIF orientation is applied so the reported dims match
/// what the user will see displayed.
fn decode_thumbnail(path: &Path) -> Option<(DynamicImage, (u32, u32))> {
    let bytes = read_file_bytes(path).ok()?;
    let slice = bytes.as_slice();

    // Fast path: JPEG via DCT scale. Request 256 so we have a 2x supersample
    // for the final thumbnail() downsample, which gives smoother edges.
    if is_jpeg(slice)
        && let Some((mut img, original_dims)) = try_decode_jpeg_scaled(slice, 256)
    {
        let orientation = read_exif_orientation(slice);
        img.apply_orientation(orientation);
        let dims = orient_dims(original_dims, orientation);
        return Some((img.thumbnail(128, 128), dims));
    }

    // Fallback: full decode. We've already read the bytes once via mmap, so
    // hand them straight to ImageReader instead of re-opening the file.
    let cursor = Cursor::new(slice);
    let reader = image::ImageReader::new(cursor).with_guessed_format().ok()?;
    let mut decoder = reader.into_decoder().ok()?;
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut img = DynamicImage::from_decoder(decoder).ok()?;
    drop(bytes);
    img.apply_orientation(orientation);
    // After apply_orientation, `img` already reflects the swapped axes, so
    // its own dimensions are the user-facing source resolution.
    let dims = (img.width(), img.height());
    Some((img.thumbnail(128, 128), dims))
}

/// Apply an EXIF Orientation to a (width, height) pair so the result
/// matches what `DynamicImage::apply_orientation` would produce.
fn orient_dims((w, h): (u32, u32), orientation: image::metadata::Orientation) -> (u32, u32) {
    use image::metadata::Orientation::*;
    match orientation {
        Rotate90 | Rotate270 | Rotate90FlipH | Rotate270FlipH => (h, w),
        _ => (w, h),
    }
}

pub fn request_image(
    path: PathBuf,
    purpose: ImagePurpose,
    cancel: CancelToken,
    sender: Sender<Message>,
    ctx: egui::Context,
) {
    rayon::spawn(move || {
        // Cancelled while queued (rare but possible under heavy traversal).
        if cancel.is_cancelled() {
            log::debug!(
                "Decode pre-cancelled ({:?}): {:?}",
                purpose,
                path.file_name().unwrap_or_default()
            );
            return;
        }
        log::debug!(
            "Image decode start ({:?}): {:?}",
            purpose,
            path.file_name().unwrap_or_default()
        );
        if let ImagePurpose::Display { generation } = purpose {
            let large = std::fs::metadata(&path)
                .map(|m| m.len() >= LARGE_IMAGE_PREVIEW_MIN_FILE_BYTES)
                .unwrap_or(false);
            if large && let Some(img) = decode_large_jpeg_preview(&path, &cancel) {
                let _ = sender.send(Message::ImageDecoded {
                    path: path.clone(),
                    image: Arc::new(img),
                    purpose: ImagePurpose::PreviewDisplay { generation },
                });
                ctx.request_repaint();
            }
        }
        match decode_image_prepared(&path, &cancel) {
            Ok(img) => {
                let _ = sender.send(Message::ImageDecoded {
                    path,
                    image: Arc::new(img),
                    purpose,
                });
                ctx.request_repaint();
            }
            Err(DecodeError::Image(e)) => {
                log::error!("Image decode failed for {:?}: {}", path, e);
                let _ = sender.send(Message::ImageFailed {
                    path,
                    error: e.to_string(),
                    purpose,
                });
                ctx.request_repaint();
            }
            Err(DecodeError::Cancelled) => {
                log::debug!(
                    "Decode cancelled mid-flight ({:?}): {:?}",
                    purpose,
                    path.file_name().unwrap_or_default()
                );
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_jpeg_magic_bytes() {
        assert!(is_jpeg(&[0xFF, 0xD8, 0xFF, 0xE0]));
        assert!(is_jpeg(&[0xFF, 0xD8, 0xFF, 0xDB]));
        assert!(!is_jpeg(&[0x89, 0x50, 0x4E, 0x47]), "PNG should not match");
        assert!(!is_jpeg(&[]), "empty");
        assert!(!is_jpeg(&[0xFF, 0xD8]), "too short");
    }

    #[test]
    fn cancel_token_observes_cancellation() {
        let t = CancelToken::new();
        assert!(!t.is_cancelled());
        let t2 = t.clone();
        t2.cancel();
        assert!(
            t.is_cancelled(),
            "cancellation must be visible across clones"
        );
    }
}
