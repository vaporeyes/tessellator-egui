// ABOUTME: Background I/O for folder scanning and image decoding.
// ABOUTME: Workers send Message variants back to the UI thread via crossbeam.

use crossbeam_channel::Sender;
use eframe::egui;
use image::{DynamicImage, ImageDecoder};
use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use wide::u32x8;

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
            log::warn!("mmap failed for {:?}, falling back to buffered read: {}", path, e);
            read_to_vec(&mut file, size).map(FileBytes::Heap)
        }
    }
}

fn read_to_vec(file: &mut File, size_hint: u64) -> std::io::Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(size_hint as usize);
    file.read_to_end(&mut buf)?;
    Ok(buf)
}

/// Best-effort detection of network-mounted paths. False negatives mean we
/// take the SIGBUS risk; false positives mean a slightly slower buffered read.
#[cfg(target_os = "macos")]
fn is_likely_network_path(path: &Path) -> bool {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let Ok(cpath) = CString::new(path.as_os_str().as_bytes()) else {
        return false;
    };
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

#[cfg(not(target_os = "macos"))]
fn is_likely_network_path(_path: &Path) -> bool {
    false
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
/// (CMYK, 16-bit grayscale) so the caller can fall back.
fn try_decode_jpeg_scaled(bytes: &[u8], target_min_dim: u16) -> Option<DynamicImage> {
    use jpeg_decoder::{Decoder as JpegDecoder, PixelFormat};

    let mut decoder = JpegDecoder::new(Cursor::new(bytes));
    decoder.read_info().ok()?;

    // Snap the request to the closest 1/N (N in {1, 2, 4, 8}) DCT scale.
    decoder.scale(target_min_dim, target_min_dim).ok()?;

    let pixels = decoder.decode().ok()?;
    let info = decoder.info()?;
    let w = info.width as u32;
    let h = info.height as u32;

    match info.pixel_format {
        PixelFormat::RGB24 => {
            // Promote to RGBA so downstream paths (thumbnail egui::ColorImage,
            // viewport upload) stay on a single pixel format.
            let mut rgba = Vec::with_capacity(pixels.len() / 3 * 4);
            for px in pixels.chunks_exact(3) {
                rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
            }
            image::RgbaImage::from_raw(w, h, rgba).map(DynamicImage::ImageRgba8)
        }
        PixelFormat::L8 => {
            image::GrayImage::from_raw(w, h, pixels).map(DynamicImage::ImageLuma8)
        }
        // CMYK32, L16: rare. Fall back to the full image-crate path.
        _ => None,
    }
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

/// One level of an RGBA mip chain, ready for direct GPU upload.
pub struct MipLevel {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// A fully prepared image: dimensions plus a complete RGBA mip chain. All
/// CPU-side work (decode, EXIF rotate, RGBA convert, mip generation) is done
/// on the worker thread that produces this so the render thread only needs to
/// memcpy bytes to the GPU.
pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    pub mips: Vec<MipLevel>,
}

impl DecodedImage {
    /// Approximate footprint of the decoded mip chain in bytes.
    pub fn byte_size(&self) -> usize {
        self.mips.iter().map(|m| m.rgba.len()).sum()
    }
}

/// Decode an image, honoring EXIF orientation when the format provides it
/// (notably JPEG). Without this, phone photos display rotated.
///
/// File contents are loaded via memory-mapped I/O when possible, avoiding the
/// page-cache → user-space copy that the standard `BufReader` path triggers.
fn decode_image(path: &Path) -> image::ImageResult<DynamicImage> {
    let bytes = read_file_bytes(path).map_err(image::ImageError::IoError)?;
    let cursor = Cursor::new(bytes.as_slice());
    let reader = image::ImageReader::new(cursor)
        .with_guessed_format()
        .map_err(image::ImageError::IoError)?;
    let mut decoder = reader.into_decoder()?;
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut img = DynamicImage::from_decoder(decoder)?;
    // `bytes` must outlive the decoder; explicit drop here documents that.
    drop(bytes);
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

fn decode_image_with_mips(
    path: &Path,
    cancel: &CancelToken,
) -> Result<DecodedImage, DecodeError> {
    let img = decode_image(path)?;
    if cancel.is_cancelled() {
        return Err(DecodeError::Cancelled);
    }
    let rgba = to_rgba8_consuming(img);
    if cancel.is_cancelled() {
        return Err(DecodeError::Cancelled);
    }
    let (width, height) = rgba.dimensions();
    let max_dim = width.max(height).max(1);
    let level_count = max_dim.ilog2() + 1;

    let mut mips = Vec::with_capacity(level_count as usize);
    mips.push(MipLevel { width, height, rgba: rgba.into_raw() });
    for _ in 1..level_count {
        if cancel.is_cancelled() {
            return Err(DecodeError::Cancelled);
        }
        let prev = mips.last().unwrap();
        mips.push(downsample_box(&prev.rgba, prev.width, prev.height));
    }
    Ok(DecodedImage { width, height, mips })
}

/// 2x2 box-filter downsample of an RGBA8 buffer. Odd dimensions clamp to the
/// last row/column.
///
/// Uses SWAR-on-u32 averaging (the `0x00FF00FF` mask trick splits each pixel
/// into two pairs of channels that sum independently), accelerated to 8
/// pixels at a time via `wide::u32x8`. For mips at or above 256x256 output,
/// rows are processed in parallel via rayon. Below that, the rayon dispatch
/// overhead exceeds the SIMD work, so we run sequentially.
fn downsample_box(src: &[u8], src_w: u32, src_h: u32) -> MipLevel {
    let dst_w = (src_w / 2).max(1);
    let dst_h = (src_h / 2).max(1);
    let src_stride = (src_w as usize) * 4;
    let dst_stride = (dst_w as usize) * 4;
    let mut dst = vec![0u8; (dst_w * dst_h * 4) as usize];

    // Below this output size, rayon dispatch overhead exceeds the work.
    const PARALLEL_THRESHOLD: u32 = 256 * 256;
    let parallel = dst_w * dst_h >= PARALLEL_THRESHOLD;

    let row_op = |dst_y: usize, dst_row: &mut [u8]| {
        let sy0 = dst_y * 2;
        let sy1 = (sy0 + 1).min((src_h - 1) as usize);
        // Both source rows are 4-byte aligned because `src` came from a Vec<u8>
        // (≥ 16-byte alignment) and `src_stride` is a multiple of 4.
        let top = bytemuck::cast_slice::<u8, u32>(
            &src[sy0 * src_stride..sy0 * src_stride + src_stride],
        );
        let bot = bytemuck::cast_slice::<u8, u32>(
            &src[sy1 * src_stride..sy1 * src_stride + src_stride],
        );
        let dst_row = bytemuck::cast_slice_mut::<u8, u32>(dst_row);
        downsample_row(top, bot, dst_row, src_w as usize, dst_w as usize);
    };

    if parallel {
        dst.par_chunks_mut(dst_stride)
            .enumerate()
            .for_each(|(y, row)| row_op(y, row));
    } else {
        for (y, row) in dst.chunks_mut(dst_stride).enumerate() {
            row_op(y, row);
        }
    }

    MipLevel { width: dst_w, height: dst_h, rgba: dst }
}

#[inline]
fn downsample_row(top: &[u32], bot: &[u32], dst: &mut [u32], src_w: usize, dst_w: usize) {
    let simd_chunks = dst_w / 8;
    for chunk in 0..simd_chunks {
        let x = chunk * 16;
        // Manual deinterleave: even-indexed input pixels into one vector,
        // odd-indexed into the other. wide doesn't expose shuffles, so we
        // build the lanes explicitly. The compiler typically unrolls this
        // into a few NEON/AVX shuffle ops.
        let p00 = u32x8::from([
            top[x],     top[x + 2], top[x + 4], top[x + 6],
            top[x + 8], top[x + 10], top[x + 12], top[x + 14],
        ]);
        let p01 = u32x8::from([
            top[x + 1], top[x + 3], top[x + 5], top[x + 7],
            top[x + 9], top[x + 11], top[x + 13], top[x + 15],
        ]);
        let p10 = u32x8::from([
            bot[x],     bot[x + 2], bot[x + 4], bot[x + 6],
            bot[x + 8], bot[x + 10], bot[x + 12], bot[x + 14],
        ]);
        let p11 = u32x8::from([
            bot[x + 1], bot[x + 3], bot[x + 5], bot[x + 7],
            bot[x + 9], bot[x + 11], bot[x + 13], bot[x + 15],
        ]);
        let avg = avg_4_u32x8(p00, p01, p10, p11);
        dst[chunk * 8..chunk * 8 + 8].copy_from_slice(&avg.to_array());
    }
    // Scalar tail for the remaining 0..7 output pixels.
    let tail_start = simd_chunks * 8;
    for (i, slot) in dst[tail_start..dst_w].iter_mut().enumerate() {
        let x = tail_start + i;
        let sx0 = x * 2;
        let sx1 = (sx0 + 1).min(src_w - 1);
        *slot = avg_4_u32(top[sx0], top[sx1], bot[sx0], bot[sx1]);
    }
}

/// Average 8 sets of 4 RGBA pixels in parallel using SWAR + SIMD.
///
/// Splitting each u32 with `0x00FF00FF` puts R and B into separate u16 lanes
/// inside one u32; shifting by 8 then masking does the same for G and A.
/// 4 channel values sum to ≤ 1020, which fits in 10 bits — no carry into
/// the adjacent lane.
#[inline]
fn avg_4_u32x8(p00: u32x8, p01: u32x8, p10: u32x8, p11: u32x8) -> u32x8 {
    let mask = u32x8::splat(0x00FF_00FF);
    let lo = (p00 & mask) + (p01 & mask) + (p10 & mask) + (p11 & mask);
    let hi = ((p00 >> 8) & mask)
        + ((p01 >> 8) & mask)
        + ((p10 >> 8) & mask)
        + ((p11 >> 8) & mask);
    ((lo >> 2) & mask) | (((hi >> 2) & mask) << 8)
}

#[inline]
fn avg_4_u32(p00: u32, p01: u32, p10: u32, p11: u32) -> u32 {
    let mask = 0x00FF_00FF_u32;
    let lo = (p00 & mask) + (p01 & mask) + (p10 & mask) + (p11 & mask);
    let hi = ((p00 >> 8) & mask)
        + ((p01 >> 8) & mask)
        + ((p10 >> 8) & mask)
        + ((p11 >> 8) & mask);
    ((lo >> 2) & mask) | (((hi >> 2) & mask) << 8)
}

/// Why an image was decoded. `Display` results drive the viewport (subject to
/// generation matching); `Preload` results only populate the cache; `Compare`
/// results populate the right-side compare slot.
#[derive(Debug, Clone, Copy)]
pub enum ImagePurpose {
    Display { generation: u64 },
    Preload,
    Compare { generation: u64 },
}

pub enum Message {
    FilesFound(Vec<ScannedFile>),
    ThumbnailLoaded { path: PathBuf, image: egui::ColorImage },
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
        let _ = sender.send(Message::FilesFound(found));
        ctx.request_repaint();
    });
}

pub fn request_thumbnail(path: PathBuf, sender: Sender<Message>, ctx: egui::Context) {
    rayon::spawn(move || {
        // Thumbnails don't need mips - they're sampled near 1:1 by egui.
        let result = decode_thumbnail(&path);
        match result {
            Some(thumbnail) => {
                let size = [thumbnail.width() as usize, thumbnail.height() as usize];
                let pixels = to_rgba8_consuming(thumbnail);
                let color_image = egui::ColorImage::from_rgba_unmultiplied(
                    size,
                    pixels.as_flat_samples().as_slice(),
                );
                let _ = sender.send(Message::ThumbnailLoaded { path, image: color_image });
            }
            None => {
                log::warn!("Thumbnail decode failed for {:?}", path);
                let _ = sender.send(Message::ThumbnailFailed);
            }
        }
        ctx.request_repaint();
    });
}

/// Produce a thumbnail-sized DynamicImage. Tries the JPEG DCT fast-path first
/// and falls back to the full image-crate decode for non-JPEG or unsupported
/// JPEG variants. Always applies EXIF orientation so phone photos display
/// upright.
fn decode_thumbnail(path: &Path) -> Option<DynamicImage> {
    let bytes = read_file_bytes(path).ok()?;
    let slice = bytes.as_slice();

    // Fast path: JPEG via DCT scale. Request 256 so we have a 2x supersample
    // for the final thumbnail() downsample, which gives smoother edges.
    if is_jpeg(slice)
        && let Some(mut img) = try_decode_jpeg_scaled(slice, 256)
    {
        let orientation = read_exif_orientation(slice);
        img.apply_orientation(orientation);
        return Some(img.thumbnail(128, 128));
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
    Some(img.thumbnail(128, 128))
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
        match decode_image_with_mips(&path, &cancel) {
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
