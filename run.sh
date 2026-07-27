#!/usr/bin/env bash
#
# Start WhisperMac: backend first, then the app.
#
#   ./run.sh                # start everything (build the app if needed)
#   ./run.sh --server-only  # backend only, stay in the foreground
#   ./run.sh --rebuild      # rebuild the app before launching
#
# Ctrl-C — or quitting the app — shuts the backend down again.
#
set -euo pipefail
# shellcheck source=scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/scripts/lib.sh"

SERVER_ONLY=0
REBUILD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --server-only) SERVER_ONLY=1 ;;
    --rebuild) REBUILD=1 ;;
    -h | --help)
      sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) die "Unknown option: $1" ;;
  esac
  shift
done

require_macos_arm
venv_ready || die "Not set up yet. Run ./scripts/install.sh first."
have ffmpeg || warn "ffmpeg is missing — transcription will fail. Fix: brew install ffmpeg"

BACKEND_PID=""

cleanup() {
  if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    printf '\n'
    step "Stopping the backend"
    kill "$BACKEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
    ok "Stopped"
  fi
}
trap cleanup EXIT INT TERM

# ------------------------------------------------------------------- backend

step "Starting the transcription engine"
if backend_healthy; then
  ok "Already running at $(backend_url)"
else
  mkdir -p "$LOG_DIR"
  "$REPO_ROOT/scripts/run_backend.sh" >>"$LOG_DIR/backend.out" 2>&1 &
  BACKEND_PID=$!
  info "Waiting for it to come up…"
  if ! wait_for_backend 80; then
    fail "The backend did not start within 40 seconds."
    dim "Last lines of $LOG_DIR/backend.out:"
    tail -n 20 "$LOG_DIR/backend.out" 2>/dev/null || true
    exit 1
  fi
  ok "Listening on $(backend_url)"
fi

if [[ "$SERVER_ONLY" == "1" ]]; then
  step "Backend running — press Ctrl-C to stop"
  dim "API docs: $(backend_url)/docs"
  # Nothing else to supervise; just hold the terminal open.
  if [[ -n "$BACKEND_PID" ]]; then wait "$BACKEND_PID"; else sleep infinity; fi
  exit 0
fi

# ----------------------------------------------------------------------- app

if [[ "$REBUILD" == "1" || ! -d "$APP_BUNDLE" ]]; then
  step "Building the macOS app"
  "$REPO_ROOT/scripts/build.sh"
fi

step "Launching WhisperMac"
open "$APP_BUNDLE"
ok "Window opening — drag a video or audio file onto it"
dim "Transcripts: $REPO_ROOT/output"
dim "Quit the app (or press Ctrl-C here) to shut the engine down."

# Supervise: exit once the app goes away, so the trap can stop the backend.
sleep 2
while pgrep -f "$APP_BUNDLE/Contents/MacOS/WhisperMac" >/dev/null 2>&1; do
  if [[ -n "$BACKEND_PID" ]] && ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    fail "The backend exited unexpectedly. See $LOG_DIR/whisper-mac.log"
    exit 1
  fi
  sleep 1
done

step "WhisperMac closed"
