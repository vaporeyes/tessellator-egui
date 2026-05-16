// ABOUTME: macOS menu-bar (system tray) presence using the Menu Bar Icon asset.
// ABOUTME: Created lazily on the first frame; menu clicks polled each update.

#[cfg(target_os = "macos")]
mod imp {
    use eframe::egui;
    use std::time::Duration;
    use tray_icon::menu::{Menu, MenuEvent, MenuId, MenuItem, PredefinedMenuItem};
    use tray_icon::{TrayIcon, TrayIconBuilder};

    const MENU_BAR_ICON_PNG: &[u8] = include_bytes!("../assets/menu_bar_icon.png");

    fn load_icon() -> Option<tray_icon::Icon> {
        let img = image::load_from_memory(MENU_BAR_ICON_PNG).ok()?.to_rgba8();
        let (w, h) = (img.width(), img.height());
        tray_icon::Icon::from_rgba(img.into_raw(), w, h).ok()
    }

    struct Tray {
        // Held to keep the tray alive; dropping it removes the menu-bar icon.
        _icon: TrayIcon,
        show_id: MenuId,
        hide_id: MenuId,
        quit_id: MenuId,
    }

    impl Tray {
        fn build() -> Option<Self> {
            let menu = Menu::new();
            let show = MenuItem::new("Show Tessellator", true, None);
            let hide = MenuItem::new("Hide Tessellator", true, None);
            let quit = MenuItem::new("Quit Tessellator", true, None);
            menu.append(&show).ok()?;
            menu.append(&hide).ok()?;
            menu.append(&PredefinedMenuItem::separator()).ok()?;
            menu.append(&quit).ok()?;
            let show_id = show.id().clone();
            let hide_id = hide.id().clone();
            let quit_id = quit.id().clone();
            let icon = TrayIconBuilder::new()
                .with_tooltip("Tessellator")
                .with_menu(Box::new(menu))
                .with_icon(load_icon()?)
                .build()
                .ok()?;
            Some(Self {
                _icon: icon,
                show_id,
                hide_id,
                quit_id,
            })
        }

        fn poll(&self, ctx: &egui::Context) {
            while let Ok(ev) = MenuEvent::receiver().try_recv() {
                if ev.id == self.show_id {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Visible(true));
                    ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
                } else if ev.id == self.hide_id {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Visible(false));
                } else if ev.id == self.quit_id {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            }
        }
    }

    #[derive(Default)]
    pub struct TrayHost {
        tray: Option<Tray>,
        tried: bool,
    }

    impl TrayHost {
        pub fn update(&mut self, ctx: &egui::Context) {
            // Built on the first frame: by then eframe's event loop and the
            // macOS NSApplication exist, which tray-icon requires.
            if !self.tried {
                self.tried = true;
                self.tray = Tray::build();
                if self.tray.is_none() {
                    log::warn!("Tray icon could not be created");
                }
            }
            if let Some(tray) = &self.tray {
                tray.poll(ctx);
                // Keep the loop ticking while the window is hidden so a
                // "Show" click is handled promptly without user input.
                ctx.request_repaint_after(Duration::from_millis(200));
            }
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod imp {
    use eframe::egui;

    #[derive(Default)]
    pub struct TrayHost;

    impl TrayHost {
        pub fn update(&mut self, _ctx: &egui::Context) {}
    }
}

pub use imp::TrayHost;
