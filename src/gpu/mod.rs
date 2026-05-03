// ABOUTME: WGPU rendering subsystem - pipeline, textures, and egui paint callback.
// ABOUTME: Re-exports the public types used by the app shell.

mod callback;
mod resources;

pub use callback::TessellatorCallback;
pub use resources::ShaderSettings;
