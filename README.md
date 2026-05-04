# Tessellator

A high-performance photo viewer for artists, built in Rust on `eframe` + WGPU.

Designed for the "flip rapidly through a folder of references" workflow, with extras for visual analysis: composition overlays, A/B compare, magnifier loupe, eyedropper.

## Features

- **Fast browsing.** Folder list with thumbnails, keyboard navigation, neighbor preload, and an LRU image cache so flipping back and forth is instant.
- **GPU-accelerated viewport.** WGPU with mipmaps and trilinear filtering. Smooth zoom-to-cursor and drag-to-pan.
- **Fit / Fill / 100% / arbitrary zoom.** One-key access via `F`, `Shift+F`, `1`.
- **Composition overlays.** Rule of thirds, golden ratio, diagonal cross, custom grid - all with adjustable opacity.
- **A/B compare.** Pick a second image; drag the divider to reveal one over the other at the same zoom and pan.
- **Grid compare (A/B/C/D).** Pick 2-4 images and view them side-by-side (1xN strip or 2x2 grid), each aspect-fitted in its tile, sharing a single zoom + pan.
- **Loupe / magnifier.** Hold `Alt` to get a circular magnifier centered on the cursor.
- **Eyedropper.** Hovering shows the exact RGB / hex of the pixel under the cursor in the status bar.
- **Drag-and-drop.** Drop a folder (or any image) onto the window to open.
- **Live folder watching.** Files added or removed externally appear in the list within ~500 ms.
- **Grayscale slider.** Useful for value studies.
- **Value study mode.** Posterize to 2-8 luma bands to check light/dark structure.
- **Flip horizontal.** One-key mirror to spot composition issues.
- **Crop preview.** Overlay the framing for square, 4:5, 16:9, or golden-rectangle ratios.
- **Annotation layer.** Toggle on, then drag to paint over the image (color picker + brush size + eraser). Strokes auto-save to a sidecar PNG (`photo.jpg.tess.png`) next to the original and reload when you revisit the image.
- **Stars / favourites.** Mark images as favourites (sidecar JSON `photo.jpg.tess.json`); filter the sidebar to "starred only" with one click.
- **Reference mode.** Always-on-top + borderless window so the viewer floats above your painting app. Pair with Compact mode to hide all panels for a chrome-free reference image.
- **Histogram overlay.** RGB + luminance, computed once per image, drawn on the viewport.
- **Color palette extraction.** 8 dominant colors per image via median cut, click a swatch to copy its hex code.
- **Split-toning preview.** Cinematic teal/orange by default; pick custom shadow + highlight tints.
- **Clipping warning.** Magenta over blown highlights, cyan over crushed shadows.
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
| `H` | Flip horizontal (mirror displayed image) |
| `V` | Toggle value study (posterized grayscale) |
| `A` | Toggle annotation mode (drag paints) |
| `S` | Toggle star on the current image |
| `Shift+S` | Filter sidebar to starred images only |
| `T` | Always-on-top + borderless (reference mode) |
| `\` | Compact (hide all panels - just the image) |
| `Esc` | Clear compare mode |

### App

| Key | Action |
|---|---|
| `Cmd+O` / `Ctrl+O` | Open folder picker |
| `Cmd+R` / `Ctrl+R` | Re-scan current folder (preserves selection) |
| `Cmd+C` / `Ctrl+C` | Copy current image's path to clipboard |

The sidebar's "Recent" button shows the last few folders you've opened.

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
