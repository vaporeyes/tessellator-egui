// ABOUTME: Embedded brand assets (app icon, logo, launch art) and helpers to
// ABOUTME: turn them into an eframe window icon and lazily-cached egui textures.

use eframe::egui;
use std::sync::Arc;

/// 1024x1024 app icon, used for the window/Dock/taskbar icon at runtime, the
/// About window, and as the source for the bundled macOS .icns.
const APP_ICON_PNG: &[u8] = include_bytes!("../assets/app_icon.png");
/// Launch/hero art shown in the viewport before a folder or image is selected.
const LAUNCH_SCREEN_PNG: &[u8] = include_bytes!("../assets/launch_screen.png");

/// Toolbar glyphs sliced from the icon sheet, recolored to white-on-transparent
/// so egui can tint them to match the theme / active state. Index order is the
/// sheet's row-major order (#0..#9).
const TOOLBAR_GLYPHS: [&[u8]; 10] = [
    include_bytes!("../assets/toolbar/glyph_0.png"),
    include_bytes!("../assets/toolbar/glyph_1.png"),
    include_bytes!("../assets/toolbar/glyph_2.png"),
    include_bytes!("../assets/toolbar/glyph_3.png"),
    include_bytes!("../assets/toolbar/glyph_4.png"),
    include_bytes!("../assets/toolbar/glyph_5.png"),
    include_bytes!("../assets/toolbar/glyph_6.png"),
    include_bytes!("../assets/toolbar/glyph_7.png"),
    include_bytes!("../assets/toolbar/glyph_8.png"),
    include_bytes!("../assets/toolbar/glyph_9.png"),
];

/// One-line description shown in the About window.
pub const TAGLINE: &str = "A high-definition photo viewer for artists";
/// Copyright line shown in the About window (mirror of Info.plist's
/// NSHumanReadableCopyright). Edit both together.
pub const COPYRIGHT: &str = "Copyright © 2026 jsh";

/// Decode an embedded PNG to premultiplied-free RGBA8 plus its dimensions.
/// Panics only on a corrupt embedded asset, which is a build-time mistake.
fn decode_rgba(bytes: &[u8]) -> (usize, usize, Vec<u8>) {
    let img = image::load_from_memory(bytes)
        .expect("embedded brand PNG failed to decode")
        .to_rgba8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    (w, h, img.into_raw())
}

/// Build the eframe window icon from the embedded app icon. Called once from
/// `main` before the event loop starts.
pub fn window_icon() -> Arc<egui::IconData> {
    let (width, height, rgba) = decode_rgba(APP_ICON_PNG);
    Arc::new(egui::IconData {
        rgba,
        width: width as u32,
        height: height as u32,
    })
}

fn color_image(bytes: &[u8]) -> egui::ColorImage {
    let (w, h, rgba) = decode_rgba(bytes);
    egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba)
}

/// Lazily-created textures for the brand art. Textures can't be built until an
/// egui context exists, so they're created on first use and cached.
#[derive(Default)]
pub struct Branding {
    icon: Option<egui::TextureHandle>,
    launch: Option<egui::TextureHandle>,
    toolbar: [Option<egui::TextureHandle>; 10],
    /// Whether the About window is currently shown.
    pub about_open: bool,
}

impl Branding {
    fn icon_texture(&mut self, ctx: &egui::Context) -> egui::TextureHandle {
        self.icon
            .get_or_insert_with(|| {
                ctx.load_texture(
                    "brand_icon",
                    color_image(APP_ICON_PNG),
                    egui::TextureOptions::LINEAR,
                )
            })
            .clone()
    }

    fn launch_texture(&mut self, ctx: &egui::Context) -> egui::TextureHandle {
        self.launch
            .get_or_insert_with(|| {
                ctx.load_texture(
                    "brand_launch",
                    color_image(LAUNCH_SCREEN_PNG),
                    egui::TextureOptions::LINEAR,
                )
            })
            .clone()
    }

    /// Texture for toolbar glyph `i` (0..10), created on first use.
    pub fn toolbar_texture(
        &mut self,
        ctx: &egui::Context,
        i: usize,
    ) -> egui::TextureHandle {
        self.toolbar[i]
            .get_or_insert_with(|| {
                ctx.load_texture(
                    format!("toolbar_glyph_{i}"),
                    color_image(TOOLBAR_GLYPHS[i]),
                    egui::TextureOptions::LINEAR,
                )
            })
            .clone()
    }

    /// Centered launch/hero art for the empty viewport state, sized to fit the
    /// available rect without upscaling past the source resolution.
    pub fn show_launch_hero(&mut self, ui: &mut egui::Ui) {
        let tex = self.launch_texture(ui.ctx());
        let src = tex.size_vec2();
        let avail = ui.available_size();
        // Fit within 70% of the viewport, never upscale past native size.
        let max_w = (avail.x * 0.7).min(src.x);
        let scale = (max_w / src.x).min(1.0);
        ui.add(egui::Image::new(&tex).fit_to_exact_size(src * scale));
    }

    /// Modal-style About window. Toggled via `about_open`.
    pub fn show_about_window(&mut self, ctx: &egui::Context) {
        if !self.about_open {
            return;
        }
        let icon = self.icon_texture(ctx);
        let mut open = self.about_open;
        let mut close_clicked = false;
        egui::Window::new("About Tessellator")
            .id(egui::Id::new("about_window"))
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(8.0);
                    ui.add(
                        egui::Image::new(&icon)
                            .fit_to_exact_size(egui::vec2(160.0, 160.0)),
                    );
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new("Tessellator")
                            .heading()
                            .size(28.0)
                            .strong(),
                    );
                    ui.label(
                        egui::RichText::new(format!(
                            "Version {}",
                            env!("CARGO_PKG_VERSION")
                        ))
                        .weak(),
                    );
                    ui.add_space(6.0);
                    ui.label(TAGLINE);
                    ui.add_space(2.0);
                    ui.label(egui::RichText::new(COPYRIGHT).weak().small());
                    ui.add_space(12.0);
                    if ui.button("Close").clicked() {
                        close_clicked = true;
                    }
                    ui.add_space(8.0);
                });
            });
        self.about_open = open && !close_clicked;
    }
}
