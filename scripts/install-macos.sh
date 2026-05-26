#!/usr/bin/env bash
# ABOUTME: Builds Tessellator, assembles a macOS .app with image file
# ABOUTME: associations, installs it, and registers it with Launch Services.
set -euo pipefail

# Run from the repo root regardless of where the script is invoked from.
cd "$(dirname "$0")/.."

APP_NAME="Tessellator"
BIN_NAME="tessellator-egui"          # cargo's binary name (CFBundleExecutable)
IDENTIFIER="com.tessellator.viewer"
ICNS="assets/Tessellator.icns"
CREDITS="assets/Credits.html"
COPYRIGHT="Copyright © 2026 jsh"   # shown in the native About panel
VERSION="$(grep -m1 '^version' Cargo.toml | sed 's/.*"\(.*\)".*/\1/')"

# Install to /Applications when writable, else ~/Applications (no sudo needed).
DEST_DIR="/Applications"
if [ ! -w "$DEST_DIR" ]; then
  DEST_DIR="$HOME/Applications"
  mkdir -p "$DEST_DIR"
fi
APP_DIR="$DEST_DIR/$APP_NAME.app"

echo "==> Building release binary"
cargo build --release

if [ ! -f "$ICNS" ]; then
  echo "error: $ICNS not found (generate it from assets/app_icon.png first)" >&2
  exit 1
fi

echo "==> Assembling $APP_DIR (v$VERSION)"
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS" "$APP_DIR/Contents/Resources"
cp "target/release/$BIN_NAME" "$APP_DIR/Contents/MacOS/$BIN_NAME"
chmod +x "$APP_DIR/Contents/MacOS/$BIN_NAME"
cp "$ICNS" "$APP_DIR/Contents/Resources/$APP_NAME.icns"
# Credits.html shows under the version in the native "About Tessellator" panel.
[ -f "$CREDITS" ] && cp "$CREDITS" "$APP_DIR/Contents/Resources/Credits.html"

cat > "$APP_DIR/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>$APP_NAME</string>
  <key>CFBundleDisplayName</key><string>$APP_NAME</string>
  <key>CFBundleIdentifier</key><string>$IDENTIFIER</string>
  <key>CFBundleExecutable</key><string>$BIN_NAME</string>
  <key>CFBundleIconFile</key><string>$APP_NAME</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSHumanReadableCopyright</key><string>$COPYRIGHT</string>
  <key>CFBundleDocumentTypes</key>
  <array>
    <dict>
      <key>CFBundleTypeName</key><string>Image</string>
      <key>CFBundleTypeRole</key><string>Viewer</string>
      <!-- Default = offer Tessellator as the default opener (user still
           chooses in Finder). Change to Alternate to only appear in Open With. -->
      <key>LSHandlerRank</key><string>Default</string>
      <key>LSItemContentTypes</key>
      <array>
        <string>public.jpeg</string>
        <string>public.png</string>
        <string>public.tiff</string>
        <string>com.microsoft.bmp</string>
        <string>org.webmproject.webp</string>
        <string>public.jpeg-2000</string>
      </array>
    </dict>
    <dict>
      <key>CFBundleTypeName</key><string>Comic Book Archive</string>
      <key>CFBundleTypeRole</key><string>Viewer</string>
      <key>LSHandlerRank</key><string>Default</string>
      <!-- System-declared UTIs for .cbz/.cbr (public.cbz-archive /
           public.cbr-archive); no custom UTI declaration needed. -->
      <key>LSItemContentTypes</key>
      <array>
        <string>public.cbz-archive</string>
        <string>public.cbr-archive</string>
      </array>
    </dict>
  </array>
</dict>
</plist>
PLIST

# Ad-hoc code signature: makes locally-built apps behave better with Gatekeeper
# and keychain. Best-effort; not a distributable signature.
echo "==> Ad-hoc signing"
codesign --force --deep --sign - "$APP_DIR" || echo "   (codesign skipped/failed; app still runs locally)"

# Register with Launch Services so it shows up in right-click -> Open With.
echo "==> Registering with Launch Services"
LSREGISTER="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
"$LSREGISTER" -f "$APP_DIR"

echo "==> Done: $APP_DIR"
echo "    Right-click an image -> Open With -> $APP_NAME."
echo "    To make it the default: Get Info on an image -> Open with -> $APP_NAME -> Change All."
