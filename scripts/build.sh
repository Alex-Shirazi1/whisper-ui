#!/usr/bin/env bash
#
# Build build/WhisperMac.app from the SwiftPM package.
#
#   ./scripts/build.sh              # release build + bundle
#   ./scripts/build.sh --debug      # faster, unoptimised
#   ./scripts/build.sh --clean      # wipe .build first
#
set -euo pipefail
# shellcheck source=scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

CONFIGURATION="release"
CLEAN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug) CONFIGURATION="debug" ;;
    --release) CONFIGURATION="release" ;;
    --clean) CLEAN=1 ;;
    -h | --help)
      sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) die "Unknown option: $1" ;;
  esac
  shift
done

require_macos_arm
have swift || die "Swift is missing. Install the Xcode command line tools: xcode-select --install"

step "Compiling WhisperMac ($CONFIGURATION)"
if [[ "$CLEAN" == "1" ]]; then
  rm -rf "$APP_DIR/.build"
  info "Cleaned $APP_DIR/.build"
fi
(cd "$APP_DIR" && swift build -c "$CONFIGURATION")
BINARY="$APP_DIR/.build/$CONFIGURATION/WhisperMac"
[[ -x "$BINARY" ]] || die "Build finished but $BINARY is missing."
ok "Built $(basename "$BINARY")"

step "Assembling the app bundle"
CONTENTS="$APP_BUNDLE/Contents"
rm -rf "$APP_BUNDLE"
mkdir -p "$CONTENTS/MacOS" "$CONTENTS/Resources"
cp "$BINARY" "$CONTENTS/MacOS/WhisperMac"

# Stamp this checkout's path so the app can start its own backend.
sed "s|__WM_REPO_ROOT__|$REPO_ROOT|g" "$APP_DIR/WhisperMac/Info.plist" \
  >"$CONTENTS/Info.plist"
printf 'APPL????' >"$CONTENTS/PkgInfo"
ok "Bundle laid out at ${APP_BUNDLE#"$REPO_ROOT"/}"

step "Generating the app icon"
ICONSET="$BUILD_DIR/WhisperMac.iconset"
rm -rf "$ICONSET"
# Cosmetic: a failure here must not fail the build.
if swift "$REPO_ROOT/scripts/make_icon.swift" "$ICONSET" >/dev/null 2>&1 &&
  have iconutil &&
  iconutil --convert icns --output "$CONTENTS/Resources/AppIcon.icns" "$ICONSET" 2>/dev/null; then
  rm -rf "$ICONSET"
  ok "AppIcon.icns generated"
else
  warn "Could not generate an icon; the app will use the generic one."
fi

step "Signing"
# Ad-hoc signature: enough for Gatekeeper to run a locally built app, and it
# keeps the bundle stable across moves.
if codesign --force --deep --sign - --timestamp=none "$APP_BUNDLE" 2>/dev/null; then
  ok "Ad-hoc signed"
else
  warn "codesign failed; macOS may warn on first launch (right-click ▸ Open)."
fi

step "Done"
dim "App:    $APP_BUNDLE"
dim "Launch: ./run.sh   (or: open '$APP_BUNDLE')"
