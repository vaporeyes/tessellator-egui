# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Tracked debt

See [TODO.md](TODO.md) for known bugs that haven't fired yet, deferred features
(HDR, manual rotation, delete/rename), architecture risks, and quick UX wins
that have been identified but not done. Update it when you defer something
non-trivial.

## Commands

- Build: `cargo build` (release: `cargo build --release`)
- Run: `cargo run` (use `RUST_LOG=debug cargo run` to see decode/cache/watcher logs)
- Check / lint: `cargo check`, `cargo clippy --no-deps`
- No tests yet; `cargo test` is a no-op.

Rust edition is `2024`, so a recent stable toolchain is required.

**Performance note**: image decode + mipmap generation is much slower in debug builds. For realistic feel use `cargo run --release`.

## Architecture

Tessellator is a single-binary photo viewer for artists, built on `eframe`/`egui` with a WGPU backend. Source layout:

```
src/
  main.rs       Entry point; eframe NativeOptions; module declarations.
  app.rs        TessellatorApp + update loop. The bulk of UI logic.
  io.rs         Background I/O: folder scan, image decode, mip generation.
  cache.rs      LRU cache of DecodedImage keyed by path, capped by bytes.
  view.rs       ViewState enum + view_matrix helper.
  watcher.rs    notify wrapper that fires Message::FolderChanged.
  gpu/
    mod.rs      Re-exports.
    resources.rs  Pipeline, bind group layout, ShaderSettings, texture upload.
    callback.rs   egui_wgpu::CallbackTrait impl (prepare/paint).
  shader.wgsl   Fragment + vertex shader. include_str!'d into resources.rs.
```

### Threading model

- **UI thread** drives `App::update()` and reads from a `crossbeam_channel::Receiver<Message>`.
- **Rayon pool** does all CPU-heavy image work: file read, decode, EXIF rotate, RGBA convert, mip chain generation. Each result is sent back as a `Message`. Mip generation itself uses rayon's row-level parallelism for the larger mip levels.
- **One-off `std::thread`** for folder scanning (I/O-bound, doesn't deserve a Rayon worker).
- **notify watcher thread** owned by `FolderWatcher`. Sends `Message::FolderChanged` and is dropped when the folder changes.
- **Render thread** (egui_wgpu's) runs `TessellatorCallback::prepare` and `paint`. The decision to do all CPU prep on the worker (not in `prepare`) is load-bearing — see "High-res upload path" below.

### File I/O (mmap-first)

`io::read_file_bytes(path)` returns a `FileBytes` that's either an `Mmap` or a `Vec<u8>`. Decisions:

- File < 256 KB: heap-read (mmap setup overhead exceeds benefit).
- macOS network mount (`nfs`/`smbfs`/`afpfs`/`webdav`/`ftp` via `libc::statfs`): heap-read (SIGBUS on disconnect isn't catchable in safe Rust).
- Otherwise: try `Mmap::map`, fall back to heap-read on error.

The decoder reads from `Cursor<&[u8]>` over the underlying slice — no kernel-to-user copy on the mmap path. **The `unsafe` in `Mmap::map` is the truncation caveat** — see the `// SAFETY:` comment in `read_file_bytes`.

### Decode pipeline

`io::decode_image(path)`: full-quality decode for the viewport. Reads via mmap, decodes via `image::ImageReader`, applies EXIF orientation.

`io::decode_thumbnail(path)`: optimized for 128 px thumbnails.
- **JPEG fast-path** via `jpeg-decoder`: requests `decoder.scale(256, 256)` which snaps to the closest 1/N (N ∈ {1, 2, 4, 8}) DCT factor. Decodes at the scaled resolution — for a 24MP source this is ~64x less raster work than a full decode.
- EXIF orientation read separately via `kamadak-exif` (the fast path bypasses the image crate's `ImageDecoder::orientation()`).
- Falls back to the full image-crate decode for non-JPEG, CMYK JPEGs, and L16 grayscale JPEGs.
- Either way, then `.thumbnail(128, 128)` for final downsample.

**JPEG 2000** (`.jp2`/`.j2k`/`.jpf`/`.jpx`/`.j2c`/`.jpc`) is decoded by the `jpeg2k` crate (pure-Rust `openjp2` backend, no C/cmake). The `image` crate handles neither container nor codestream. `io::is_jp2` sniffs magic bytes (JP2 signature box or the `FF 4F FF 51` SOC marker) so the byte-only decode functions can route without an extension. Both `decode_image_from_bytes` and the `decode_thumbnail` fallback branch on it; `jpeg2k::Image::from_bytes` then `DynamicImage::try_from` (interop works because jpeg2k pins `image` 0.25, unified with ours). JP2 has no EXIF orientation and no DCT-scale fast path, so thumbnails decode at full resolution.

### Comic archives (CBZ/CBR)

`archive.rs` adds comic support by *extracting to a temp folder*, so the rest of the app needs no archive-awareness - a comic becomes an ordinary folder of pages. `is_archive_file` matches `.cbz`/`.cbr` (deliberately NOT in `io::IMAGE_EXTENSIONS`, so archives are entered on open, not listed inside folder scans). `extract_to_temp` writes image entries (filtered by `is_image_file`) as zero-padded indices (`00000.jpg`, ...) in archive-iteration order, so the folder's filename sort reproduces page order. CBZ uses the pure-Rust `zip` crate; CBR uses `unrar` (bundled C++ unrar via `unrar_sys`, strict front-to-back cursor: `read_header` then `extract_to`/`skip`).

Flow in `app.rs`: `open_path_list` routes archives to `open_archive`, which extracts on a one-off thread and sends `Message::ArchiveExtracted`. The handler calls `exit_archive_mode` (deletes the prior temp dir), records `current_archive`/`archive_temp`, pushes the *archive* (not the temp dir) to recents, sets `pending_select_first`, and `load_folder(temp)`. `open_folder` is the normal-folder wrapper (exit archive + recents + `load_folder`); `load_folder` is the shared core. Persistence saves `archive_path` instead of `folder_path` when in a comic (the temp dir won't exist next launch), reopening (re-extracting) it on startup. `Drop` removes the temp dir on exit; killed processes leak it to the OS temp cleaner.

### High-res upload path

CPU prep is concentrated in the decode worker. `io::decode_image_with_mips` produces a `DecodedImage { width, height, mips: Vec<MipLevel> }` where each `MipLevel` is raw RGBA bytes ready for `queue.write_texture`. The render thread does almost no CPU work — just iterating mips and uploading. **Do not move mip generation back into `TessellatorResources::set_main_texture`**; that's what made the viewer feel sluggish before this change.

### Mip generation (SWAR + SIMD + rayon)

`io::downsample_box` does a 2x2 box filter on RGBA8 with three layers of speedup:
1. **SWAR averaging** per `u32` pixel: the `0x00FF00FF` mask trick splits R/B from G/A into separate u16 lanes within one u32 so 4-channel averaging is one mask + add + shift, no per-byte loop. Max sum (4 × 255 = 1020) fits in 10 bits, no carry into adjacent lanes.
2. **SIMD via `wide::u32x8`**: 8 output pixels per batch. Manual deinterleave (no shuffle in `wide`) into even/odd vectors. Compiler typically lowers to a few NEON/AVX shuffles.
3. **Rayon row parallelism** for output mips ≥ 256×256. Below that, dispatch overhead exceeds the work — runs sequentially.

`bytemuck::cast_slice` reinterprets row bytes as `&[u32]` without copying. Alignment is guaranteed by Vec's allocator + 4-byte stride.

### Zero-copy RGBA

`io::to_rgba8_consuming` matches on `DynamicImage::ImageRgba8` and consumes the buffer in-place. For other variants (e.g. `ImageRgb8` from JPEG) the conversion is unavoidable. Used in both the high-res and thumbnail paths.

The Display path:
1. `select_image` checks the LRU cache; on hit, no decode is issued.
2. On miss, `io::request_image(path, ImagePurpose::Display { generation })` runs on Rayon.
3. `Message::ImageDecoded` arrives; if its generation is current, the image is shown via `show_image`.
4. `show_image` sets `current_image: Option<Arc<DecodedImage>>` and `needs_upload = true`.
5. The viewport's paint callback clones `current_image` into `TessellatorCallback::image` *only* when `needs_upload` is set. Slider tweaks therefore don't churn the GPU.

### Stale-result cancellation

`high_res_generation: u64` (and `compare_generation` for compare mode) increments on every user-initiated request. Decoded results carry the generation they were dispatched with; mismatches are silently dropped (but still cached). Without this, fast keyboard-flipping would let earlier slow decodes clobber the latest selection.

### LRU image cache

`ImageCache` (in `cache.rs`) holds `Arc<DecodedImage>` keyed by path. Capped at 512 MB by summing each image's mip-chain bytes. `get` is mark-LRU (promotes to front). The most-recently-used entry is never evicted, so a single huge image still displays.

Used for two things:
- **Neighbor preload**: `select_image` fires Preload requests for N-1 and N+1.
- **Compare hits**: `start_compare` checks the cache first.

### Shader / uniform layout (load-bearing)

`ShaderSettings` is `#[repr(C)]` and exactly 112 bytes:

```
view_matrix       [f32; 12]    offset 0    (3 columns of mat3x3<f32> padded to vec4 each)
grayscale         f32          offset 48
overlay_opacity   f32          offset 52
grid_size         f32          offset 56
overlay_mode      u32          offset 60
compare_divider   f32          offset 64
compare_active    u32          offset 68
loupe_active      u32          offset 72
loupe_zoom        f32          offset 76
loupe_center_uv   [f32; 2]     offset 80   (vec2 align = 8, 80 % 8 = 0 OK)
loupe_center_screen [f32; 2]   offset 88
loupe_radius      f32          offset 96
dither            u32          offset 100
_pad0, _pad1      f32, f32     offset 104, 108
```

A `const _: () = assert!(size_of::<ShaderSettings>() == 112)` in `gpu/resources.rs` catches misalignment at compile time. **WGSL `Settings` struct in `shader.wgsl` must match field-for-field**.

### View matrix convention

3x3 column-major stored as `[f32; 12]` (each column padded to vec4). Build via `view::view_matrix(scale, pan)`, which encodes:
- Indices 0/5: scale.x / scale.y
- Indices 8/9: pan.x / -pan.y (NDC Y is up; egui Y is down)

`ViewState` is `FitOnNextFrame | FillOnNextFrame | Manual { zoom, pan }`. Fit/Fill resolve to `Manual` in `show_viewport` once the rect is known. The eyedropper and loupe both use `screen_to_uv` to invert this transform.

### Compare mode

Bind group has 4 entries: `t_diffuse` (slot 0), sampler (1), uniform (2), `t_compare` (3). When compare is off the slot 3 view falls back to the main texture so the binding is always valid. `TessellatorResources` rebuilds the bind group whenever either texture changes; the `CompareUpload` enum (`NoChange | Set | Clear`) avoids unnecessary rebuilds.

### Loupe

Pure shader. Inside a screen-space disc centered on `loupe_center_screen` with radius `loupe_radius` (pixels), UV is remapped: `loupe_center_uv + (in.tex_coords - loupe_center_uv) / loupe_zoom`. Overlays (rule-of-thirds etc.) deliberately use the *original* UV so they stay anchored to the image rather than warping inside the loupe.

### Folder watcher (debounced)

`FolderWatcher` wraps `notify::RecommendedWatcher`. Events touching image files send `Message::FolderChanged`; the app debounces with a 500 ms inactivity window (`folder_refresh_at: Option<Instant>`) and uses `ctx.request_repaint_after` so the wake doesn't depend on user interaction. Selection is preserved across rescan via `restore_selection: Option<PathBuf>` resolved in the next `FilesFound` handler.

### Folder sort

`io::scan_folder` sorts results case-insensitively by filename via `sort_by_cached_key` (one lowercase allocation per file, not per comparison) before sending `FilesFound`. The UI never sees an unsorted list.

### Keyboard shortcuts

All shortcuts are read in one place: `read_pressed_keys` returns a `Pressed` struct of `bool`s, dispatched by `handle_keyboard_nav`. Adding a new shortcut means adding a field + a `consume_key` line + a dispatch branch — three edits in two functions. `consume_key` ensures focused widgets (sliders, text inputs) don't also receive the press.

`Modifiers::COMMAND` is egui's portable "Cmd on macOS, Ctrl elsewhere" — use it for app-level shortcuts like Open/Refresh.

Keyboard zoom shortcuts (`=`/`+`/`-`) write to `pending_zoom_step: Option<f32>`, which `show_viewport` consumes when it has the live zoom in scope. Same deferral pattern as `pending_scroll_to_index`.

### Persistence

Uses `eframe::App::save` with the `persistence` feature. `PersistentState` is a `serde` struct of `Option<T>` fields so adding new fields later doesn't break old saved state. Auto-opens the last folder on startup if it still exists. Sidebar width and window size are handled by eframe's context persistence (no extra code).

### Thumbnails

Issued only for currently-visible rows (driven by `ScrollArea::show_rows`'s `row_range`). Once requested, a path stays in `requested_thumbnails` for the session whether decoding succeeds or fails, so broken files aren't retried every frame.

## Conventions

- Files start with two `// ABOUTME:` lines (per global CLAUDE.md).
- No emojis or em-dashes in code, comments, or docs.
- Surgical edits only - don't refactor unrelated areas while touching this file.
- Use `log::*` macros, never `println!` for diagnostics.
- WGSL: avoid reserved words (`target`, `binding`, etc.) for identifiers - naga rejects them and the failure only surfaces at runtime.
- Adding a keyboard shortcut: add a field to `Pressed`, a `consume_key` line in `read_pressed_keys`, a dispatch branch in `handle_keyboard_nav`. Update README's shortcut tables.
- Persistence (`PersistentState`): all fields are `Option<T>` so adding new fields doesn't break old saved state.

## Known limitations

- **HDR / wide-gamut display** is not implemented. eframe 0.31 doesn't expose surface format selection. Revisit when upgrading eframe. The dither pass is a partial substitute that eliminates banding within the LDR pipeline.
- **EXIF orientation** is honored only for formats whose `image::ImageDecoder` reports orientation (JPEG works; PNG/WebP technically can carry EXIF chunks but the crate may not parse them). The thumbnail JPEG fast-path uses `kamadak-exif` directly so it gets orientation regardless.
- **Folder watcher** uses `notify::RecursiveMode::Recursive` for any depth > 1 - notify doesn't accept a depth limit, so we may receive (and ignore) events outside the visible depth.
- **Filename sort is lexicographic, not natural.** `Photo (10).jpg` sorts before `Photo (2).jpg`. Camera/phone filenames (zero-padded `IMG_0001`) sort correctly. Add `natord` if natural sort is needed.
- **Network share detection is macOS-only.** Linux/Windows always pick mmap, taking the SIGBUS risk on remote mounts. Cross-platform statfs is straightforward to add (~10 lines per platform) when needed.
