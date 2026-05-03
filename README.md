# Tessellator

A high-performance photo viewer for artists, built in Rust on `eframe` + WGPU.

Designed for the "flip rapidly through a folder of references" workflow, with extras for visual analysis: composition overlays, A/B compare, magnifier loupe, eyedropper.

## Features

- **Fast browsing.** Folder list with thumbnails, keyboard navigation, neighbor preload, and an LRU image cache so flipping back and forth is instant.
- **GPU-accelerated viewport.** WGPU with mipmaps and trilinear filtering. Smooth zoom-to-cursor and drag-to-pan.
- **Fit / Fill / 100% / arbitrary zoom.** One-key access via `F`, `Shift+F`, `1`.
- **Composition overlays.** Rule of thirds, golden ratio, diagonal cross, custom grid - all with adjustable opacity.
- **A/B compare.** Pick a second image; drag the divider to reveal one over the other at the same zoom and pan.
- **Loupe / magnifier.** Hold `Alt` to get a circular magnifier centered on the cursor.
- **Eyedropper.** Hovering shows the exact RGB / hex of the pixel under the cursor in the status bar.
- **Drag-and-drop.** Drop a folder (or any image) onto the window to open.
- **Live folder watching.** Files added or removed externally appear in the list within ~500 ms.
- **Grayscale slider.** Useful for value studies.
- **EXIF orientation.** Phone photos display upright.
- **Persistence.** Last folder, panel sizes, and tool settings restore on launch.

## Build & run

Requires a recent stable Rust toolchain (edition 2024).

```sh
cargo run --release
```

Debug builds work but image decoding is much slower; use release for realistic feel.

To enable verbose logging:

```sh
RUST_LOG=debug cargo run --release
```

## Keyboard shortcuts

### Navigation

| Key | Action |
|---|---|
| `Left` / `Right` | Previous / next image |
| `Space` | Next image (alias) |
| `Home` / `End` | First / last image |
| `PageUp` / `PageDown` | Jump 10 images |

### View

| Key | Action |
|---|---|
| `F` or `0` | Fit image to viewport |
| `Shift+F` | Fill viewport (may crop) |
| `1` | 100% (native pixel size) |
| `=` or `+` | Zoom in 10% |
| `-` | Zoom out 10% |
| `Alt` (held) | Show magnifier loupe under cursor |

### Tools

| Key | Action |
|---|---|
| `G` | Toggle grayscale (full color ↔ full B&W) |
| `Esc` | Clear compare mode |

### App

| Key | Action |
|---|---|
| `Cmd+O` / `Ctrl+O` | Open folder picker |
| `Cmd+R` / `Ctrl+R` | Re-scan current folder (preserves selection) |

## Mouse

| Action | Effect |
|---|---|
| Scroll | Zoom in/out around cursor |
| Drag | Pan |
| Hover | Eyedropper (status bar) |

## Supported formats

JPEG, PNG, WebP, BMP, TIFF. (Driven by the `image` crate.)

## Configuration

Recursion depth for folder scanning is exposed in the sidebar (default `2`, max `16`). Other settings live in the tools panel and persist between sessions.

## License

MIT (or whatever you want; this project is unlicensed by default).
