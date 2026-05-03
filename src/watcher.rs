// ABOUTME: Filesystem watcher that fires Message::FolderChanged when image files
// ABOUTME: appear, disappear, or are modified inside the loaded folder.

use crate::io::{is_image_file, Message};
use crossbeam_channel::Sender;
use eframe::egui;
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use std::path::Path;

/// Owns a notify watcher for the lifetime of a loaded folder. Dropping it
/// stops the watch.
pub struct FolderWatcher {
    _watcher: RecommendedWatcher,
}

impl FolderWatcher {
    pub fn new(
        path: &Path,
        recursive: bool,
        sender: Sender<Message>,
        ctx: egui::Context,
    ) -> notify::Result<Self> {
        let mut watcher = notify::recommended_watcher(
            move |res: notify::Result<notify::Event>| {
                let Ok(event) = res else { return };
                // Filter out events that don't touch image files (avoids
                // refreshing on .DS_Store, sidecar XMP, etc.).
                if event.paths.iter().any(|p| is_image_file(p)) {
                    let _ = sender.send(Message::FolderChanged);
                    ctx.request_repaint();
                }
            },
        )?;
        let mode = if recursive {
            RecursiveMode::Recursive
        } else {
            RecursiveMode::NonRecursive
        };
        watcher.watch(path, mode)?;
        Ok(Self { _watcher: watcher })
    }
}
