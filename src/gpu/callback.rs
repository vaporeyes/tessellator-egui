// ABOUTME: egui_wgpu paint callback that lazily creates GPU resources and uploads
// ABOUTME: any pending high-res or compare image before drawing the textured viewport quad.

use eframe::wgpu;
use std::sync::Arc;

use super::resources::{ShaderSettings, TessellatorResources};
use crate::io::DecodedImage;

/// What to do with the compare-slot texture this frame.
pub enum CompareUpload {
    NoChange,
    Set(Arc<DecodedImage>),
    Clear,
}

/// What to do with the annotation-slot texture this frame.
pub enum AnnotationUpload {
    NoChange,
    /// Recreate a fresh annotation texture of the given size and upload these
    /// pixels (RGBA8, row-tight).
    SetFull { width: u32, height: u32, rgba: Arc<Vec<u8>> },
    /// Patch a sub-region of the existing annotation texture.
    Patch {
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        rgba: Vec<u8>,
    },
    Clear,
}

pub struct TessellatorCallback {
    pub image: Option<Arc<DecodedImage>>,
    pub compare: CompareUpload,
    pub annotation: AnnotationUpload,
    pub settings: ShaderSettings,
    pub format: wgpu::TextureFormat,
}

impl egui_wgpu::CallbackTrait for TessellatorCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if !resources.contains::<TessellatorResources>() {
            resources.insert(TessellatorResources::new(device, queue, self.format));
        }
        let tess = resources
            .get_mut::<TessellatorResources>()
            .expect("TessellatorResources was just inserted");

        if let Some(image) = &self.image {
            tess.set_main_texture(device, queue, image);
        }
        match &self.compare {
            CompareUpload::Set(image) => tess.set_compare_texture(device, queue, Some(image)),
            CompareUpload::Clear => tess.set_compare_texture(device, queue, None),
            CompareUpload::NoChange => {}
        }
        match &self.annotation {
            AnnotationUpload::SetFull { width, height, rgba } => {
                tess.set_annotation_texture(device, queue, *width, *height, rgba);
            }
            AnnotationUpload::Patch { x, y, width, height, rgba } => {
                tess.patch_annotation_region(queue, *x, *y, *width, *height, rgba);
            }
            AnnotationUpload::Clear => tess.clear_annotation_texture(device),
            AnnotationUpload::NoChange => {}
        }
        tess.update_settings(queue, self.settings);

        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let tess = resources
            .get::<TessellatorResources>()
            .expect("TessellatorResources should be inserted before paint");
        if let Some(bind_group) = tess.current_bind_group() {
            render_pass.set_pipeline(tess.pipeline());
            render_pass.set_bind_group(0, bind_group, &[]);
            render_pass.set_vertex_buffer(0, tess.vertex_buffer().slice(..));
            render_pass.draw(0..6, 0..1);
        }
    }
}
