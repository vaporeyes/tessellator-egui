// ABOUTME: Per-image annotation layer for paint-over sketches.
// ABOUTME: Persists to a sidecar PNG next to the source image (e.g. photo.jpg.tess.png).

use image::RgbaImage;
use std::path::{Path, PathBuf};

/// Sub-region of the annotation buffer modified since the last GPU upload.
/// `max_x` / `max_y` are exclusive.
#[derive(Debug, Clone, Copy)]
pub struct DirtyRect {
    pub min_x: u32,
    pub min_y: u32,
    pub max_x: u32,
    pub max_y: u32,
}

pub struct AnnotationLayer {
    pub width: u32,
    pub height: u32,
    pub rgba: RgbaImage,
    /// Pixels modified since the last `take_dirty` call, if any.
    dirty: Option<DirtyRect>,
    /// True when in-memory state diverges from the sidecar on disk.
    pub unsaved: bool,
    /// True if the layer has had any pixel painted since creation/load.
    /// Used to decide whether to write a sidecar at all (don't litter folders
    /// with empty PNGs for images the user only looked at).
    pub touched: bool,
}

impl AnnotationLayer {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            rgba: RgbaImage::new(width, height),
            // Fresh transparent buffer needs a full upload to clear any prior
            // texture content the GPU side may be holding.
            dirty: Some(DirtyRect { min_x: 0, min_y: 0, max_x: width, max_y: height }),
            unsaved: false,
            touched: false,
        }
    }

    /// Sidecar path for `image_path`: append ".tess.png".
    pub fn sidecar_path(image_path: &Path) -> PathBuf {
        let mut s = image_path.as_os_str().to_owned();
        s.push(".tess.png");
        PathBuf::from(s)
    }

    /// Try to load a sidecar PNG. Returns `None` if absent, unreadable, or
    /// dimensions mismatch the source image.
    pub fn load_sidecar(image_path: &Path, width: u32, height: u32) -> Option<Self> {
        let sidecar = Self::sidecar_path(image_path);
        if !sidecar.exists() {
            return None;
        }
        let img = match image::open(&sidecar) {
            Ok(i) => i.to_rgba8(),
            Err(e) => {
                log::warn!("Failed to read sidecar {}: {}", sidecar.display(), e);
                return None;
            }
        };
        if img.width() != width || img.height() != height {
            log::warn!(
                "Sidecar {} is {}x{} but image is {}x{}; ignoring",
                sidecar.display(),
                img.width(),
                img.height(),
                width,
                height
            );
            return None;
        }
        Some(Self {
            width,
            height,
            rgba: img,
            dirty: Some(DirtyRect { min_x: 0, min_y: 0, max_x: width, max_y: height }),
            unsaved: false,
            touched: true,
        })
    }

    /// Stamp a soft-edged disc of radius `r` (image pixels) at (cx, cy).
    /// `color` is straight RGBA (0-255). When `erase` is true, multiplies
    /// existing alpha down toward zero rather than blending in `color`.
    pub fn stamp(&mut self, cx: f32, cy: f32, radius: f32, color: [u8; 4], erase: bool) {
        let r = radius.max(0.5);
        let r_sq = r * r;
        let x0 = ((cx - r).floor() as i32).max(0);
        let y0 = ((cy - r).floor() as i32).max(0);
        let x1 = ((cx + r).ceil() as i32).min(self.width as i32);
        let y1 = ((cy + r).ceil() as i32).min(self.height as i32);
        if x0 >= x1 || y0 >= y1 {
            return;
        }
        for y in y0..y1 {
            for x in x0..x1 {
                let dx = x as f32 + 0.5 - cx;
                let dy = y as f32 + 0.5 - cy;
                let d_sq = dx * dx + dy * dy;
                if d_sq > r_sq {
                    continue;
                }
                // 1px linear AA at the edge.
                let d = d_sq.sqrt();
                let edge = (r - d).clamp(0.0, 1.0);
                let pixel = self.rgba.get_pixel_mut(x as u32, y as u32);
                if erase {
                    let cur = pixel[3] as f32 / 255.0;
                    let new = cur * (1.0 - edge);
                    pixel[3] = (new * 255.0).round() as u8;
                } else {
                    // Straight-alpha source-over compositing of `color` onto
                    // the existing pixel, with `edge` modulating source alpha.
                    let src_a = (color[3] as f32 / 255.0) * edge;
                    let dst_a = pixel[3] as f32 / 255.0;
                    let out_a = src_a + dst_a * (1.0 - src_a);
                    if out_a < 1e-3 {
                        continue;
                    }
                    let blend = |sc: u8, dc: u8| -> u8 {
                        let s = sc as f32 / 255.0;
                        let d = dc as f32 / 255.0;
                        ((s * src_a + d * dst_a * (1.0 - src_a)) / out_a * 255.0).round() as u8
                    };
                    pixel[0] = blend(color[0], pixel[0]);
                    pixel[1] = blend(color[1], pixel[1]);
                    pixel[2] = blend(color[2], pixel[2]);
                    pixel[3] = (out_a * 255.0).round() as u8;
                }
                extend_dirty(&mut self.dirty, x as u32, y as u32);
            }
        }
        self.unsaved = true;
        self.touched = true;
    }

    /// Stamp a chain of overlapping discs from `a` to `b` so fast drags don't
    /// leave dotted lines. Spacing is `radius / 2` (capped at 1px minimum).
    pub fn stroke(
        &mut self,
        a: (f32, f32),
        b: (f32, f32),
        radius: f32,
        color: [u8; 4],
        erase: bool,
    ) {
        let dx = b.0 - a.0;
        let dy = b.1 - a.1;
        let dist = (dx * dx + dy * dy).sqrt();
        let step = (radius * 0.5).max(1.0);
        let n = ((dist / step).ceil() as usize).max(1);
        for i in 0..=n {
            let t = i as f32 / n as f32;
            self.stamp(a.0 + dx * t, a.1 + dy * t, radius, color, erase);
        }
    }

    /// True if every pixel is fully transparent. Used by the save path so an
    /// erased-to-empty layer deletes the sidecar instead of writing a blank PNG.
    pub fn is_empty(&self) -> bool {
        self.rgba.as_raw().chunks_exact(4).all(|p| p[3] == 0)
    }

    /// Take the current dirty rect and the row-tight RGBA slice for it.
    /// Returns `None` if nothing has changed since the last call.
    pub fn take_dirty(&mut self) -> Option<(DirtyRect, Vec<u8>)> {
        let r = self.dirty.take()?;
        let w = (r.max_x - r.min_x) as usize;
        let h = (r.max_y - r.min_y) as usize;
        let stride_src = self.width as usize * 4;
        let mut out = Vec::with_capacity(w * h * 4);
        for y in r.min_y..r.max_y {
            let row_start = y as usize * stride_src + r.min_x as usize * 4;
            out.extend_from_slice(&self.rgba.as_raw()[row_start..row_start + w * 4]);
        }
        Some((r, out))
    }
}

fn extend_dirty(d: &mut Option<DirtyRect>, x: u32, y: u32) {
    match d {
        None => {
            *d = Some(DirtyRect {
                min_x: x,
                min_y: y,
                max_x: x + 1,
                max_y: y + 1,
            });
        }
        Some(r) => {
            r.min_x = r.min_x.min(x);
            r.min_y = r.min_y.min(y);
            r.max_x = r.max_x.max(x + 1);
            r.max_y = r.max_y.max(y + 1);
        }
    }
}
