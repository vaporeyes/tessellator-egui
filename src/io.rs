// ABOUTME: Background I/O for folder scanning and image decoding.
// ABOUTME: Workers send Message variants back to the UI thread via crossbeam.

use crossbeam_channel::Sender;
use eframe::egui;
use image::DynamicImage;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub enum Message {
    FilesFound(Vec<PathBuf>),
    ThumbnailLoaded { path: PathBuf, image: egui::ColorImage },
    ThumbnailFailed,
    HighResLoaded { generation: u64, image: Arc<DynamicImage> },
    HighResFailed { generation: u64, path: PathBuf, error: String },
}

const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "webp", "bmp", "tiff"];

pub fn is_image_file(path: &Path) -> bool {
    path.extension()
        .and_then(|s| s.to_str())
        .map(|s| IMAGE_EXTENSIONS.contains(&s.to_lowercase().as_str()))
        .unwrap_or(false)
}

pub fn scan_folder(path: PathBuf, sender: Sender<Message>, ctx: egui::Context) {
    // I/O bound, run on a one-off thread to avoid tying up Rayon's CPU pool.
    std::thread::spawn(move || {
        let mut found = Vec::new();
        for entry in walkdir::WalkDir::new(&path)
            .max_depth(2)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            if is_image_file(entry.path()) {
                found.push(entry.path().to_path_buf());
            }
        }
        log::info!("Folder scan complete: {} images under {:?}", found.len(), path);
        let _ = sender.send(Message::FilesFound(found));
        ctx.request_repaint();
    });
}

pub fn request_thumbnail(path: PathBuf, sender: Sender<Message>, ctx: egui::Context) {
    rayon::spawn(move || {
        match image::open(&path) {
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

pub fn request_high_res(
    path: PathBuf,
    generation: u64,
    sender: Sender<Message>,
    ctx: egui::Context,
) {
    rayon::spawn(move || {
        log::debug!(
            "High-res decode start (gen {}): {:?}",
            generation,
            path.file_name().unwrap_or_default()
        );
        match image::open(&path) {
            Ok(img) => {
                let _ = sender.send(Message::HighResLoaded {
                    generation,
                    image: Arc::new(img),
                });
            }
            Err(e) => {
                log::error!("High-res decode failed for {:?}: {}", path, e);
                let _ = sender.send(Message::HighResFailed {
                    generation,
                    path,
                    error: e.to_string(),
                });
            }
        }
        ctx.request_repaint();
    });
}
