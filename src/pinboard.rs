// ABOUTME: Pinboard / moodboard mode - pin multiple images on a 2D canvas.
// ABOUTME: Holds canvas state and drag/resize transitions; rendering lives in app.rs.

use eframe::egui;
use std::path::PathBuf;

/// One image pinned to the board. Position and size are in canvas-world
/// units; the renderer maps them to screen via `view_center` + `view_scale`.
#[derive(Clone)]
pub struct PinnedItem {
    pub path: PathBuf,
    /// Top-left corner in canvas coordinates.
    pub pos: egui::Vec2,
    /// Width / height in canvas units. Aspect is preserved when the user
    /// resizes via a corner handle (uniform scale).
    pub size: egui::Vec2,
}

/// Which corner of an item is being dragged for a resize.
#[derive(Clone, Copy, Debug)]
pub enum Corner {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

/// What the pointer is currently doing on the canvas, if anything. Captured
/// on `drag_started` and persisted across frames until `drag_stopped`.
#[derive(Clone)]
pub enum DragState {
    None,
    /// Moving an item; `anchor` is `mouse_world - item.pos` at drag start.
    Move { idx: usize, anchor: egui::Vec2 },
    /// Resizing from a corner. We snapshot the item's pre-drag pos+size so
    /// the math doesn't accumulate floating-point drift.
    Resize {
        idx: usize,
        corner: Corner,
        start_pos: egui::Vec2,
        start_size: egui::Vec2,
        aspect: f32,
    },
    /// Empty-area drag pans the canvas.
    Pan,
}

pub struct Pinboard {
    pub items: Vec<PinnedItem>,
    /// Canvas-world coordinate that maps to the centre of the viewport rect.
    pub view_center: egui::Vec2,
    /// Pixels-per-canvas-unit. 1.0 means a unit-sized item is one pixel tall.
    pub view_scale: f32,
    pub selected: Option<usize>,
    pub drag: DragState,
}

impl Default for Pinboard {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            view_center: egui::Vec2::ZERO,
            view_scale: 1.0,
            selected: None,
            drag: DragState::None,
        }
    }
}

impl Pinboard {
    pub fn screen_to_world(&self, p: egui::Pos2, rect: egui::Rect) -> egui::Vec2 {
        let c = rect.center();
        egui::vec2(
            (p.x - c.x) / self.view_scale + self.view_center.x,
            (p.y - c.y) / self.view_scale + self.view_center.y,
        )
    }

    pub fn world_to_screen_rect(
        &self,
        world: egui::Rect,
        rect: egui::Rect,
    ) -> egui::Rect {
        let c = rect.center();
        let to_screen = |v: egui::Vec2| -> egui::Pos2 {
            egui::pos2(
                c.x + (v.x - self.view_center.x) * self.view_scale,
                c.y + (v.y - self.view_center.y) * self.view_scale,
            )
        };
        egui::Rect::from_min_max(
            to_screen(world.min.to_vec2()),
            to_screen(world.max.to_vec2()),
        )
    }

    /// World-space rect of `idx`. None if out of range.
    pub fn item_world_rect(&self, idx: usize) -> Option<egui::Rect> {
        let it = self.items.get(idx)?;
        Some(egui::Rect::from_min_size(
            egui::pos2(it.pos.x, it.pos.y),
            egui::vec2(it.size.x, it.size.y),
        ))
    }

    /// Hit-test a world-space point. Returns (item index, optional corner).
    /// Items at the end of `items` are on top, so we iterate in reverse.
    /// `handle_radius_world` is the corner-handle hit radius in world units.
    pub fn hit_test(
        &self,
        world: egui::Vec2,
        handle_radius_world: f32,
    ) -> Option<(usize, Option<Corner>)> {
        for idx in (0..self.items.len()).rev() {
            let r = self.item_world_rect(idx)?;
            // Corner first - selecting from a corner takes priority over
            // body when they overlap (corners are tiny).
            if self.selected == Some(idx) {
                let h = handle_radius_world;
                let corners = [
                    (Corner::TopLeft, r.min),
                    (Corner::TopRight, egui::pos2(r.max.x, r.min.y)),
                    (Corner::BottomLeft, egui::pos2(r.min.x, r.max.y)),
                    (Corner::BottomRight, r.max),
                ];
                for (c, p) in corners {
                    if (egui::vec2(world.x - p.x, world.y - p.y)).length() <= h {
                        return Some((idx, Some(c)));
                    }
                }
            }
            if r.contains(egui::pos2(world.x, world.y)) {
                return Some((idx, None));
            }
        }
        None
    }

    /// Insert an item at the top of the z-stack and select it. The caller
    /// supplies the canvas-space position and size; placement near
    /// `view_center` with a slight per-item offset is the typical pattern.
    pub fn add(&mut self, item: PinnedItem) {
        self.items.push(item);
        self.selected = Some(self.items.len() - 1);
    }

    pub fn remove_selected(&mut self) {
        if let Some(i) = self.selected.take()
            && i < self.items.len()
        {
            self.items.remove(i);
        }
    }

    /// Raise `idx` to the top of the z-stack and select it. Returns the new
    /// index of the raised item.
    pub fn raise(&mut self, idx: usize) -> usize {
        if idx >= self.items.len() {
            return idx;
        }
        let item = self.items.remove(idx);
        self.items.push(item);
        let new_idx = self.items.len() - 1;
        self.selected = Some(new_idx);
        new_idx
    }
}

/// Apply a corner-resize delta to a starting rect, preserving `aspect`.
/// `mouse_world` is the current pointer position in world coords.
/// Returns the updated (pos, size).
pub fn resize_uniform(
    corner: Corner,
    start_pos: egui::Vec2,
    start_size: egui::Vec2,
    aspect: f32,
    mouse_world: egui::Vec2,
) -> (egui::Vec2, egui::Vec2) {
    // The "anchor" is the corner diagonally opposite the one being dragged.
    let anchor = match corner {
        Corner::BottomRight => start_pos,
        Corner::BottomLeft => egui::vec2(start_pos.x + start_size.x, start_pos.y),
        Corner::TopRight => egui::vec2(start_pos.x, start_pos.y + start_size.y),
        Corner::TopLeft => start_pos + start_size,
    };
    // Raw signed deltas from anchor to mouse.
    let dx = mouse_world.x - anchor.x;
    let dy = mouse_world.y - anchor.y;
    // Pick whichever axis the mouse has moved further along (in aspect-
    // adjusted terms) and clamp the other to it. This keeps aspect locked
    // even if the user drags off-axis.
    let (w, h) = if dx.abs() / aspect.max(1e-3) >= dy.abs() {
        let w = dx.abs().max(8.0);
        (w, w / aspect)
    } else {
        let h = dy.abs().max(8.0);
        (h * aspect, h)
    };
    // Place new pos so the anchor corner stays put; flip relative to anchor
    // when the user has dragged past it.
    let sign_x = dx.signum();
    let sign_y = dy.signum();
    let new_pos = egui::vec2(
        if sign_x >= 0.0 { anchor.x } else { anchor.x - w },
        if sign_y >= 0.0 { anchor.y } else { anchor.y - h },
    );
    (new_pos, egui::vec2(w, h))
}
