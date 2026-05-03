// ABOUTME: Background I/O for folder scanning and image decoding.
// ABOUTME: Workers send Message variants back to the UI thread via crossbeam.

use crossbeam_channel::Sender;
use eframe::egui;
use image::{DynamicImage, ImageDecoder};
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
fn decode_image(path: &Path) -> image::ImageResult<DynamicImage> {
    let reader = image::ImageReader::open(path)?.with_guessed_format()?;
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
fn decode_image_with_mips(path: &Path) -> image::ImageResult<DecodedImage> {
    let img = decode_image(path)?;
    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();
    let max_dim = width.max(height).max(1);
    let level_count = max_dim.ilog2() + 1;

    let mut mips = Vec::with_capacity(level_count as usize);
    mips.push(MipLevel { width, height, rgba: rgba.into_raw() });
    for _ in 1..level_count {
        let prev = mips.last().unwrap();
        mips.push(downsample_box(&prev.rgba, prev.width, prev.height));
    }
    Ok(DecodedImage { width, height, mips })
}

/// 2x2 box-filter downsample of an RGBA8 buffer. Odd dimensions clamp to the
/// last row/column. Box filtering is the mathematically correct choice when
/// chained from the previous mip level, and runs ~4x faster than Triangle.
fn downsample_box(src: &[u8], src_w: u32, src_h: u32) -> MipLevel {
    let dst_w = (src_w / 2).max(1);
    let dst_h = (src_h / 2).max(1);
    let stride = (src_w * 4) as usize;
    let mut dst = vec![0u8; (dst_w * dst_h * 4) as usize];

    for y in 0..dst_h as usize {
        let sy0 = y * 2;
        let sy1 = (sy0 + 1).min((src_h - 1) as usize);
        for x in 0..dst_w as usize {
            let sx0 = x * 2;
            let sx1 = (sx0 + 1).min((src_w - 1) as usize);
            let p00 = sy0 * stride + sx0 * 4;
            let p01 = sy0 * stride + sx1 * 4;
            let p10 = sy1 * stride + sx0 * 4;
            let p11 = sy1 * stride + sx1 * 4;
            let di = (y * dst_w as usize + x) * 4;
            for c in 0..4 {
                let sum = src[p00 + c] as u16
                    + src[p01 + c] as u16
                    + src[p10 + c] as u16
                    + src[p11 + c] as u16;
                dst[di + c] = (sum / 4) as u8;
            }
        }
    }
    MipLevel { width: dst_w, height: dst_h, rgba: dst }
}

/// Why an image was decoded. `Display` results drive the viewport (subject to
/// generation matching); `Preload` results only populate the cache.
#[derive(Debug, Clone, Copy)]
pub enum ImagePurpose {
    Display { generation: u64 },
    Preload,
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
        match decode_image(&path) {
            Ok(img) => {
                let thumbnail = img.thumbnail(128, 128);
                let size = [thumbnail.width() as usize, thumbnail.height() as usize];
                let pixels = thumbnail.to_rgba8();
                let color_image = egui::ColorImage::from_rgba_unmultiplied(
                    size,
                    pixels.as_flat_samples().as_slice(),
                );
                let _ = sender.send(Message::ThumbnailLoaded { path, image: color_image });
            }
            Err(e) => {
                log::warn!("Thumbnail decode failed for {:?}: {}", path, e);
                let _ = sender.send(Message::ThumbnailFailed);
            }
        }
        ctx.request_repaint();
    });
}

pub fn request_image(
    path: PathBuf,
    purpose: ImagePurpose,
    sender: Sender<Message>,
    ctx: egui::Context,
) {
    rayon::spawn(move || {
        log::debug!(
            "Image decode start ({:?}): {:?}",
            purpose,
            path.file_name().unwrap_or_default()
        );
        match decode_image_with_mips(&path) {
            Ok(img) => {
                let _ = sender.send(Message::ImageDecoded {
                    path,
                    image: Arc::new(img),
                    purpose,
                });
            }
            Err(e) => {
                log::error!("Image decode failed for {:?}: {}", path, e);
                let _ = sender.send(Message::ImageFailed {
                    path,
                    error: e.to_string(),
                    purpose,
                });
            }
        }
        ctx.request_repaint();
    });
}
