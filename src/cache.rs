// ABOUTME: LRU cache of decoded images keyed by path, capped by total bytes.
// ABOUTME: Used to make keyboard navigation between neighboring images instant.

use crate::io::DecodedImage;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;

struct Entry {
    path: PathBuf,
    image: Arc<DecodedImage>,
    bytes: usize,
}

pub struct ImageCache {
    /// Front = most recently used. Linear scans are fine at the sizes this
    /// cache holds (tens of entries, not thousands).
    entries: VecDeque<Entry>,
    bytes: usize,
    cap_bytes: usize,
}

impl ImageCache {
    pub fn new(cap_bytes: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            bytes: 0,
            cap_bytes,
        }
    }

    pub fn contains(&self, path: &Path) -> bool {
        self.entries.iter().any(|e| e.path == path)
    }

    /// Look up an image. On hit, promotes the entry to most-recently-used.
    pub fn get(&mut self, path: &Path) -> Option<Arc<DecodedImage>> {
        let pos = self.entries.iter().position(|e| e.path == path)?;
        let entry = self.entries.remove(pos)?;
        let image = entry.image.clone();
        self.entries.push_front(entry);
        Some(image)
    }

    pub fn insert(&mut self, path: PathBuf, image: Arc<DecodedImage>) {
        if self.contains(&path) {
            // Already cached; just promote.
            let _ = self.get(&path);
            return;
        }
        let bytes = image.byte_size();
        self.entries.push_front(Entry { path, image, bytes });
        self.bytes += bytes;
        self.evict_to_cap();
    }

    fn evict_to_cap(&mut self) {
        // Always keep the most-recently-used entry even if it exceeds the cap
        // on its own (a single huge image still has to be displayable).
        while self.bytes > self.cap_bytes && self.entries.len() > 1 {
            if let Some(victim) = self.entries.pop_back() {
                self.bytes = self.bytes.saturating_sub(victim.bytes);
            }
        }
    }
}
