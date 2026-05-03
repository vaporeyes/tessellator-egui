# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Build: `cargo build` (release: `cargo build --release`)
- Run: `cargo run` (use `RUST_LOG=info cargo run` to see `env_logger` output)
- Check / lint: `cargo check`, `cargo clippy`
- No tests exist yet; `cargo test` is a no-op.

Rust edition is `2024`, so a recent stable toolchain is required.

## Architecture

Tessellator is a single-binary photo viewer built on `eframe`/`egui` with a WGPU backend. Code is split into four modules:

- `src/main.rs` — entry point and `eframe` configuration only.
- `src/app.rs` — `TessellatorApp` (UI state, message routing, sidebar/tools/viewport panels). Drains messages at the top of `update()`, then renders the three panels.
- `src/io.rs` — `Message` enum and free functions for folder scanning (`std::thread`, I/O bound) and image decoding (`rayon::spawn`, CPU bound). Workers send messages back through a `crossbeam-channel` and call `ctx.request_repaint()` to wake the UI.
- `src/view.rs` — `ViewState` enum (`FitOnNextFrame` or `Manual { zoom, pan }`) and the `view_matrix` helper.
- `src/gpu/` — WGPU pipeline (`resources.rs`) and the `egui_wgpu::CallbackTrait` impl (`callback.rs`). The WGSL shader lives in `src/shader.wgsl` and is included via `include_str!`.

### High-res upload path

There is a single upload path: `Message::HighResLoaded` stashes the decoded `Arc<DynamicImage>` in `pending_high_res`, which `TessellatorCallback::prepare` consumes on the next frame and uploads via `TessellatorResources::update_texture`. The first paint also lazily creates `TessellatorResources` and inserts it into `egui_wgpu::CallbackResources` (keyed by type).

### View matrix convention

`ShaderSettings::view_matrix` is a 3x3 stored as `[f32; 12]` with column-major padding to 4-float alignment to match WGSL `mat3x3<f32>`. Construct it via `view::view_matrix(scale, pan)`, which handles the NDC Y flip. A `const _: () = assert!(size_of::<ShaderSettings>() == 64)` in `gpu/resources.rs` catches alignment regressions at compile time.

### Stale-result cancellation

Each high-res request is tagged with a monotonically-increasing `high_res_generation`. When the user clicks a different image, the counter advances and any in-flight decode that arrives with an older generation is discarded — preventing a slow decode from clobbering a newer selection.

### Thumbnails

Thumbnail decodes are issued only for currently-visible rows (driven by `ScrollArea::show_rows`'s `row_range`). Once requested, a path stays in `requested_thumbnails` for the session whether decoding succeeds or fails, so broken files aren't retried every frame.

## Conventions

- Files start with two `// ABOUTME:` lines (per global CLAUDE.md).
- No emojis or em-dashes in code, comments, or docs.
- Surgical edits only — don't refactor unrelated areas while touching this file.
