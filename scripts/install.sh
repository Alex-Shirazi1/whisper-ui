#!/usr/bin/env bash
#
# One-shot setup for WhisperMac.
#
#   ./scripts/install.sh                 # interactive, does everything
#   ./scripts/install.sh --yes           # never ask (CI / re-runs)
#   ./scripts/install.sh --skip-model    # don't download ~5 GB of weights yet
#   ./scripts/install.sh --skip-app      # backend only
#   ./scripts/install.sh --slim          # drop the unused torch dependency
#   ./scripts/install.sh --model REPO    # use a different Parakeet or MLX Whisper model
#
# Every step is idempotent: re-running repairs whatever is missing and leaves
# the rest alone.
#
set -euo pipefail
# shellcheck source=scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

MODEL="${WHISPER_MAC_MODEL:-mlx-community/parakeet-tdt-0.6b-v3}"
# Whisper behind Parakeet: translation, other languages, language detection.
# Set WHISPER_MAC_FALLBACK_MODEL= (empty) to skip it.
FALLBACK_MODEL="${WHISPER_MAC_FALLBACK_MODEL-mlx-community/whisper-large-v3-mlx}"
export WHISPER_MAC_FALLBACK_MODEL="$FALLBACK_MODEL"
SKIP_MODEL=0
SKIP_APP=0
SLIM=0
export ASSUME_YES="${ASSUME_YES:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -y | --yes) ASSUME_YES=1 ;;
    --skip-model) SKIP_MODEL=1 ;;
    --skip-app) SKIP_APP=1 ;;
    --slim) SLIM=1 ;;
    --model)
      shift
      [[ $# -gt 0 ]] || die "--model needs a Hugging Face repo id"
      MODEL="$1"
      ;;
    -h | --help)
      sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) die "Unknown option: $1" ;;
  esac
  shift
done

printf '%s\n' "$C_BOLD"
cat <<'BANNER'
  ┌─────────────────────────────────────────────┐
  │  WhisperMac · local transcription installer  │
  └─────────────────────────────────────────────┘
BANNER
printf '%s' "$C_RESET"

# ---------------------------------------------------------------- 1. hardware

step "Checking this Mac"
require_macos_arm
ok "$(sw_vers -productName) $(sw_vers -productVersion) on $(uname -m)"

MEMORY_GB=$(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024))
info "Memory: ${MEMORY_GB} GB"
if [[ "$MODEL $FALLBACK_MODEL" == *whisper-large* && "$MEMORY_GB" -lt 12 ]]; then
  warn "whisper-large-v3 wants ~10 GB free. Consider WHISPER_MAC_FALLBACK_MODEL=mlx-community/whisper-medium-mlx."
fi

# ---------------------------------------------------------------- 2. homebrew

step "Checking Homebrew"
if have brew; then
  ok "brew $(brew --version | head -1 | awk '{print $2}')"
else
  fail "Homebrew is not installed."
  info "Install it, then re-run this script:"
  # Printed verbatim for the user to copy — expansion here would be wrong.
  # shellcheck disable=SC2016
  dim '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
  exit 1
fi

# ------------------------------------------------------------------ 3. ffmpeg

step "Checking ffmpeg"
if have ffmpeg && have ffprobe; then
  ok "$(ffmpeg -version | head -1 | cut -c1-60)"
else
  warn "ffmpeg is required to read audio out of media files."
  if confirm "Install it now with 'brew install ffmpeg'?"; then
    brew install ffmpeg
    have ffmpeg || die "ffmpeg still isn't on PATH after installing."
    ok "ffmpeg installed"
  else
    die "Install ffmpeg (brew install ffmpeg) and re-run this script."
  fi
fi

# ------------------------------------------------------------------ 4. python

step "Setting up Python"
SETUP_ARGS=(--dev)
[[ "$SLIM" == "1" ]] && SETUP_ARGS+=(--slim)
ASSUME_YES="$ASSUME_YES" "$REPO_ROOT/scripts/setup_python.sh" "${SETUP_ARGS[@]}"

# ------------------------------------------------------------------- 5. model

step "Checking the speech models"
info "Model: $MODEL"
[[ -n "$FALLBACK_MODEL" ]] && info "Fallback: $FALLBACK_MODEL"
MODEL_DIR="$REPO_ROOT/backend/models"
mkdir -p "$MODEL_DIR"

model_cached() {
  WHISPER_MAC_MODEL_DIR="$MODEL_DIR" "$VENV_PY" - "$MODEL" <<'PY' >/dev/null 2>&1
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backend.config import get_settings
from backend.transcriber import MLXEngine

settings = get_settings()
settings.apply_model_cache_env()
engine = MLXEngine(settings.fallback_model)
raise SystemExit(0 if engine.model_is_local(sys.argv[1]) else 1)
PY
}

cd "$REPO_ROOT"
if model_cached; then
  ok "Weights already cached in backend/models — nothing to download"
elif [[ "$SKIP_MODEL" == "1" ]]; then
  warn "Skipping the download. It will happen automatically on your first file."
else
  info "This is a one-time download of roughly 5 GB into backend/models."
  if confirm "Download it now?"; then
    WHISPER_MAC_MODEL_DIR="$MODEL_DIR" "$VENV_PY" - "$MODEL" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backend.config import get_settings
from backend.transcriber import MLXEngine

settings = get_settings()
settings.apply_model_cache_env()
MLXEngine(settings.fallback_model).ensure_model(sys.argv[1])
print("Models ready.")
PY
    ok "Weights downloaded"
  else
    warn "Skipped. The first transcription will fetch them (once)."
  fi
fi

# --------------------------------------------------------------- 6. smoke test

step "Running a real transcription end to end"
SMOKE_DIR="$(mktemp -d)"
trap 'rm -rf "$SMOKE_DIR"' EXIT

if have say && model_cached; then
  say -o "$SMOKE_DIR/sample.aiff" "WhisperMac is installed and running locally." 2>/dev/null || true
  if [[ -f "$SMOKE_DIR/sample.aiff" ]]; then
    ffmpeg -hide_banner -loglevel error -y \
      -i "$SMOKE_DIR/sample.aiff" -ac 1 -ar 16000 "$SMOKE_DIR/sample.wav"
    export SMOKE_WAV="$SMOKE_DIR/sample.wav"
    if WHISPER_MAC_MODEL_DIR="$MODEL_DIR" WHISPER_MAC_MODEL="$MODEL" \
      WHISPER_MAC_OUTPUT_DIR="$SMOKE_DIR/out" HF_HUB_DISABLE_PROGRESS_BARS=1 \
      "$VENV_PY" - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import os

from backend import media
from backend.config import get_settings
from backend.formats import build_transcript, write_outputs
from backend.transcriber import MLXEngine, TranscribeOptions

settings = get_settings()
settings.apply_model_cache_env()
samples = media.load_pcm16_wav(Path(os.environ["SMOKE_WAV"]))
result = MLXEngine(settings.fallback_model).transcribe(
    samples, TranscribeOptions.from_settings(settings, language="en")
)
transcript = build_transcript(
    source_name="sample.wav", raw=result.raw, model=result.model,
    duration=media.audio_duration(samples),
)
written = write_outputs(
    transcript, output_dir=settings.output_dir, formats=["txt"], stem="install-check"
)
text = transcript.text.strip()
print(f"heard: {text!r}")
raise SystemExit(0 if text else 1)
PY
    then
      ok "Transcription works"
    else
      warn "The smoke test did not produce text. Check logs/whisper-mac.log."
    fi
  fi
else
  info "Skipped (model not downloaded yet)."
fi

# --------------------------------------------------------------- 7. build app

if [[ "$SKIP_APP" == "1" ]]; then
  step "Skipping the macOS app build"
else
  step "Building the macOS app"
  if have swift; then
    "$REPO_ROOT/scripts/build.sh"
  else
    warn "Swift is missing; install the Xcode command line tools with: xcode-select --install"
  fi
fi

# ------------------------------------------------------------------- 8. finish

step "Installed"
cat <<EOF

    Start WhisperMac:

        ${C_BOLD}./run.sh${C_RESET}

    Then drag a video or audio file onto the window.

    Transcripts:  ${REPO_ROOT}/output
    Model cache:  ${REPO_ROOT}/backend/models
    Logs:         ${REPO_ROOT}/logs/whisper-mac.log

EOF
