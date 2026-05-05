// ABOUTME: Entry point for Tessellator - configures eframe and launches the app.
// ABOUTME: All application logic lives in the app module.

mod annotation;
mod app;
mod cache;
mod gpu;
mod io;
mod pinboard;
mod view;
mod watcher;

use app::TessellatorApp;
use eframe::egui;
use eframe::wgpu;
use std::sync::Arc;

fn main() -> eframe::Result {
    env_logger::init();

    // Request the largest 2D texture dimension the adapter advertises (M1+
    // Macs report 16384). Without this, wgpu's default 8192 cap rejects any
    // image whose long axis exceeds 8192 - common for 16 MB+ JPEGs.
    let mut wgpu_options = eframe::egui_wgpu::WgpuConfiguration::default();
    if let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &mut wgpu_options.wgpu_setup {
        setup.device_descriptor = Arc::new(|adapter: &wgpu::Adapter| {
            let adapter_max = adapter.limits().max_texture_dimension_2d;
            log::info!(
                "Adapter max_texture_dimension_2d = {} (default cap is 8192)",
                adapter_max
            );
            wgpu::DeviceDescriptor {
                label: Some("tessellator.device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_texture_dimension_2d: adapter_max,
                    ..wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::default(),
            }
        });
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 720.0])
            .with_drag_and_drop(true),
        renderer: eframe::Renderer::Wgpu,
        wgpu_options,
        ..Default::default()
    };

    eframe::run_native(
        "Tessellator",
        options,
        Box::new(|cc| Ok(Box::new(TessellatorApp::new(cc)))),
    )
}
