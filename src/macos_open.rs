// ABOUTME: Receives files opened via Finder "Open With" (and argv) and queues
// ABOUTME: them for the UI thread; macOS uses a Carbon Apple Event handler.

use eframe::egui;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// Paths waiting to be opened by the UI thread. Filled by the macOS Apple Event
/// handler and by argv parsing at startup; drained each frame in `update`.
/// A queue (not a direct call) decouples the AE handler, which fires on the
/// main thread before the egui app's `update`, from the UI loop.
static PENDING: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());

/// Set once the egui context exists, so the handler can wake an idle window
/// when a file is opened while the app is already running.
static REPAINT: OnceLock<egui::Context> = OnceLock::new();

/// Queue paths to open (from the Apple Event handler or argv). Ignores empty.
pub fn queue_paths(paths: Vec<PathBuf>) {
    if paths.is_empty() {
        return;
    }
    if let Ok(mut q) = PENDING.lock() {
        q.extend(paths);
    }
    // Wake the window if it's idle (already-running "Open With"). At launch the
    // context isn't registered yet, but the app renders its first frame anyway.
    if let Some(ctx) = REPAINT.get() {
        ctx.request_repaint();
    }
}

/// Remove and return all queued paths. Called once per frame by the UI.
pub fn take_pending() -> Vec<PathBuf> {
    PENDING
        .lock()
        .map(|mut q| std::mem::take(&mut *q))
        .unwrap_or_default()
}

/// Register the egui context used to wake the window when files arrive while
/// the app is idle. Call once from `App::new`.
pub fn register_repaint(ctx: egui::Context) {
    let _ = REPAINT.set(ctx);
}

#[cfg(target_os = "macos")]
mod platform {
    use super::queue_paths;
    use block2::RcBlock;
    use objc2::ffi::class_addMethod;
    use objc2::runtime::{AnyClass, AnyObject, Imp, Sel};
    use objc2::{msg_send, sel};
    use objc2_app_kit::{NSApplication, NSApplicationWillFinishLaunchingNotification};
    use objc2_foundation::{MainThreadMarker, NSArray, NSNotification, NSNotificationCenter, NSURL};
    use std::path::PathBuf;
    use std::ptr::NonNull;

    /// Implementation of `-[delegate application:openURLs:]`, added to winit's
    /// app delegate class at runtime. AppKit's normal open handling calls this
    /// when the delegate responds to the selector.
    unsafe extern "C-unwind" fn application_open_urls(
        _this: *mut AnyObject,
        _cmd: Sel,
        _app: *mut AnyObject,
        urls: *mut NSArray<NSURL>,
    ) {
        let mut paths = Vec::new();
        if !urls.is_null() {
            let urls: &NSArray<NSURL> = unsafe { &*urls };
            for i in 0..urls.count() {
                let url = urls.objectAtIndex(i);
                if let Some(p) = url.path() {
                    paths.push(PathBuf::from(p.to_string()));
                }
            }
        }
        queue_paths(paths);
    }

    /// Add `application:openURLs:` to winit's NSApplicationDelegate class so
    /// Finder "Open With" reaches us. Called from the willFinishLaunching
    /// observer, by which point winit has created and set its delegate (in
    /// EventLoop::new) but AppKit has not yet dispatched the launch document.
    fn add_open_urls_to_delegate() {
        let Some(mtm) = MainThreadMarker::new() else {
            return;
        };
        let app = NSApplication::sharedApplication(mtm);
        let delegate: *mut AnyObject = unsafe { msg_send![&*app, delegate] };
        if delegate.is_null() {
            log::warn!("open-files: no app delegate at willFinishLaunching");
            return;
        }
        let sel = sel!(application:openURLs:);
        let responds: bool = unsafe { msg_send![delegate, respondsToSelector: sel] };
        if responds {
            return; // Already present (defensive against double-fire).
        }
        let cls: &AnyClass = unsafe { (*delegate).class() };
        let imp: Imp = unsafe {
            std::mem::transmute::<
                unsafe extern "C-unwind" fn(*mut AnyObject, Sel, *mut AnyObject, *mut NSArray<NSURL>),
                Imp,
            >(application_open_urls)
        };
        // "v@:@@" = void return; self, _cmd, NSApplication*, NSArray*.
        let added = unsafe {
            class_addMethod(
                cls as *const AnyClass as *mut AnyClass,
                sel,
                imp,
                c"v@:@@".as_ptr(),
            )
        };
        if added.as_bool() {
            log::info!("open-files: added application:openURLs: to {:?}", cls.name());
        } else {
            log::warn!("open-files: class_addMethod failed for {:?}", cls.name());
        }
    }

    /// Register a willFinishLaunching observer. It fires during AppKit's launch,
    /// after winit has set its delegate but before the launch document is
    /// dispatched - the only window where we can make winit's delegate respond
    /// to openURLs. Call from `main` before `eframe::run_native`.
    pub fn install() {
        if MainThreadMarker::new().is_none() {
            return;
        }
        let center = NSNotificationCenter::defaultCenter();
        let block = RcBlock::new(|_note: NonNull<NSNotification>| {
            add_open_urls_to_delegate();
        });
        let token = unsafe {
            center.addObserverForName_object_queue_usingBlock(
                Some(NSApplicationWillFinishLaunchingNotification),
                None,
                None,
                &block,
            )
        };
        // The center keeps its own copy of the block and the observer token;
        // leak ours so the observation lasts the whole process.
        std::mem::forget(token);
        std::mem::forget(block);
        log::info!("open-files: registered willFinishLaunching observer");
    }
}

#[cfg(target_os = "macos")]
pub use platform::install;

/// No-op on non-macOS; file opening there arrives via argv (queue_paths).
#[cfg(not(target_os = "macos"))]
pub fn install() {}
