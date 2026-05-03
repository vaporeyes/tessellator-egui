// ABOUTME: Viewport transform state and matrix helpers for the photo viewer.
// ABOUTME: Encodes zoom/pan and produces the padded 3x3 matrix the shader expects.

use eframe::egui;

#[derive(Debug, Clone, Copy)]
pub enum ViewState {
    /// Resolve to a fit-to-screen Manual transform on the next frame, once the
    /// viewport size is known.
    FitOnNextFrame,
    Manual { zoom: f32, pan: egui::Vec2 },
}

impl ViewState {
    pub fn identity() -> Self {
        Self::Manual { zoom: 1.0, pan: egui::Vec2::ZERO }
    }
}

/// Build the column-major 3x3 view matrix laid out as `[f32; 12]` to match
/// WGSL's `mat3x3<f32>` (each column padded to 16 bytes). Indices map as:
/// `[c0.x, c0.y, c0.z, _, c1.x, c1.y, c1.z, _, c2.x, c2.y, c2.z, _]`.
/// NDC Y is flipped here so callers can pass screen-space pan directly.
pub fn view_matrix(scale: egui::Vec2, pan: egui::Vec2) -> [f32; 12] {
    [
        scale.x, 0.0,     0.0, 0.0,
        0.0,     scale.y, 0.0, 0.0,
        pan.x,   -pan.y,  1.0, 0.0,
    ]
}
