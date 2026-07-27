#!/usr/bin/env bash
#
# Start the transcription backend in the foreground.
#
# Safe to call when one is already running — it checks /health first and exits 0
# rather than colliding on the port. The macOS app launches this directly.
#
set -euo pipefail
# shellcheck source=scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

if backend_healthy; then
  info "Backend already running at $(backend_url)"
  exit 0
fi

venv_ready || die "No Python environment yet. Run ./scripts/install.sh first."

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

# Keep the model cache inside the repo so weights are downloaded exactly once.
export WHISPER_MAC_MODEL_DIR="${WHISPER_MAC_MODEL_DIR:-$REPO_ROOT/backend/models}"
export PYTHONUNBUFFERED=1

exec "$VENV_PY" -m backend.whisper_server
