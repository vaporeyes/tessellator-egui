// ABOUTME: Comic archive (CBZ/CBR) support: extract image pages to a temp dir
// ABOUTME: so the rest of the app treats a comic as an ordinary folder of pages.

use crate::io::is_image_file;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// True for comic archive containers we can open (CBZ = zip, CBR = rar).
pub fn is_archive_file(path: &Path) -> bool {
    matches!(ext_lower(path).as_deref(), Some("cbz") | Some("cbr"))
}

fn ext_lower(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase())
}

/// Extract a comic archive's image pages into a fresh temp directory. Pages are
/// written as zero-padded indices in archive order (`00000.jpg`, `00001.png`,
/// ...) so the folder's filename sort reproduces the archive's page order.
/// On success returns the temp dir; the caller owns its cleanup.
pub fn extract_to_temp(archive: &Path) -> Result<PathBuf, String> {
    let dir = make_temp_dir(archive)?;
    let count = match ext_lower(archive).as_deref() {
        Some("cbz") => extract_zip(archive, &dir),
        Some("cbr") => extract_rar(archive, &dir),
        _ => Err("not a comic archive".to_string()),
    };
    match count {
        Ok(0) => {
            let _ = std::fs::remove_dir_all(&dir);
            Err("archive contains no images".to_string())
        }
        Ok(n) => {
            log::info!("Extracted {} pages from {:?}", n, archive);
            Ok(dir)
        }
        Err(e) => {
            let _ = std::fs::remove_dir_all(&dir);
            Err(e)
        }
    }
}

fn make_temp_dir(archive: &Path) -> Result<PathBuf, String> {
    let stem = archive
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("comic");
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!(
        "tessellator-comic-{}-{}-{}",
        sanitize(stem),
        std::process::id(),
        nanos
    ));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

/// Keep temp-dir names safe and short.
fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .take(40)
        .collect()
}

fn extract_zip(archive: &Path, dest: &Path) -> Result<usize, String> {
    let file = File::open(archive).map_err(|e| e.to_string())?;
    let mut zip = zip::ZipArchive::new(BufReader::new(file)).map_err(|e| e.to_string())?;
    let mut counter = 0usize;
    // by_index walks entries in stored (archive) order.
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).map_err(|e| e.to_string())?;
        if !entry.is_file() {
            continue;
        }
        let name = entry.name().to_string();
        if !is_image_name(&name) {
            continue;
        }
        let out = dest.join(page_name(counter, &name));
        let mut out_f = File::create(&out).map_err(|e| e.to_string())?;
        std::io::copy(&mut entry, &mut out_f).map_err(|e| e.to_string())?;
        counter += 1;
    }
    Ok(counter)
}

fn extract_rar(archive: &Path, dest: &Path) -> Result<usize, String> {
    let mut open = unrar::Archive::new(archive)
        .open_for_processing()
        .map_err(|e| e.to_string())?;
    let mut counter = 0usize;
    // The RAR API is a strict front-to-back cursor: read a header, then either
    // extract or skip, each returning the archive positioned at the next header.
    while let Some(header) = open.read_header().map_err(|e| e.to_string())? {
        let entry = header.entry();
        let name = entry.filename.to_string_lossy().to_string();
        if entry.is_file() && is_image_name(&name) {
            let out = dest.join(page_name(counter, &name));
            open = header.extract_to(&out).map_err(|e| e.to_string())?;
            counter += 1;
        } else {
            open = header.skip().map_err(|e| e.to_string())?;
        }
    }
    Ok(counter)
}

/// `00000.jpg` etc., preserving the page's original extension.
fn page_name(index: usize, original: &str) -> String {
    let ext = Path::new(original)
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_else(|| "jpg".to_string());
    format!("{index:05}.{ext}")
}

fn is_image_name(name: &str) -> bool {
    is_image_file(Path::new(name))
}
