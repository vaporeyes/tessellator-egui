// ABOUTME: Entry point for Tessellator - configures eframe and launches the app.
// ABOUTME: All application logic lives in the app module.

mod app;
mod cache;
mod gpu;
mod io;
mod view;

use app::TessellatorApp;
use eframe::egui;

fn main() -> eframe::Result {
    env_logger::init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 720.0])
            .with_drag_and_drop(true),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    eframe::run_native(
        "Tessellator",
        options,
        Box::new(|cc| Ok(Box::new(TessellatorApp::new(cc)))),
    )
}
