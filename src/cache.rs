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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{DecodedImage, ExifMetadata, Histogram};
    use proptest::prelude::*;

    fn img(bytes: usize) -> Arc<DecodedImage> {
        Arc::new(DecodedImage {
            width: 1,
            height: bytes.max(1) as u32,
            rgba: vec![0u8; bytes],
            histogram: Histogram::default(),
            palette: Vec::new(),
            exif: ExifMetadata::default(),
            is_preview: false,
        })
    }

    fn p(s: &str) -> PathBuf {
        PathBuf::from(s)
    }

    fn bytes_match_entries(c: &ImageCache) -> bool {
        let sum: usize = c.entries.iter().map(|e| e.bytes).sum();
        sum == c.bytes
    }

    #[test]
    fn insert_then_get_promotes_to_front() {
        let mut c = ImageCache::new(1024);
        c.insert(p("/a"), img(10));
        c.insert(p("/b"), img(10));
        // /a is now at the back. get should promote it.
        assert!(c.get(&p("/a")).is_some());
        assert_eq!(c.entries.front().unwrap().path, p("/a"));
    }

    #[test]
    fn evicts_least_recently_used_first() {
        let mut c = ImageCache::new(25);
        c.insert(p("/a"), img(10));
        c.insert(p("/b"), img(10));
        // Touching /a makes /b the LRU.
        let _ = c.get(&p("/a"));
        c.insert(p("/c"), img(10));
        assert!(c.contains(&p("/a")));
        assert!(!c.contains(&p("/b")), "LRU should have been evicted");
        assert!(c.contains(&p("/c")));
    }

    #[test]
    fn mru_kept_even_when_oversized() {
        let mut c = ImageCache::new(10);
        c.insert(p("/huge"), img(1_000));
        // Single oversized entry must still be retrievable.
        assert!(c.contains(&p("/huge")));
        assert!(c.get(&p("/huge")).is_some());
    }

    #[test]
    fn reinsert_same_path_does_not_duplicate() {
        let mut c = ImageCache::new(1024);
        c.insert(p("/a"), img(10));
        c.insert(p("/a"), img(999));
        assert_eq!(c.entries.len(), 1);
        // The second insert is a no-op aside from promotion; bytes must
        // still match the actually-stored image.
        assert!(bytes_match_entries(&c));
    }

    // Model-based proptest: drive a random sequence of ops against the cache
    // and assert invariants after each step.
    #[derive(Debug, Clone)]
    enum Op {
        Insert(u8, u16), // path-id, size
        Get(u8),
    }

    fn op_strategy() -> impl Strategy<Value = Op> {
        prop_oneof![
            (0u8..8, 1u16..200).prop_map(|(k, s)| Op::Insert(k, s)),
            (0u8..8).prop_map(Op::Get),
        ]
    }

    proptest! {
        #[test]
        fn invariants_hold_under_random_ops(
            ops in proptest::collection::vec(op_strategy(), 0..100),
            cap in 0usize..500,
        ) {
            let mut c = ImageCache::new(cap);
            for op in ops {
                match op {
                    Op::Insert(k, s) => c.insert(p(&format!("/{k}")), img(s as usize)),
                    Op::Get(k) => { let _ = c.get(&p(&format!("/{k}"))); }
                }
                prop_assert!(bytes_match_entries(&c), "bytes accounting drift");
                let over_cap = c.bytes > c.cap_bytes;
                prop_assert!(!over_cap || c.entries.len() == 1,
                    "over-cap only allowed when a single MRU entry is oversized");
                let mut seen = std::collections::HashSet::new();
                for e in &c.entries {
                    prop_assert!(seen.insert(e.path.clone()), "duplicate path in cache");
                }
            }
        }
    }
}
