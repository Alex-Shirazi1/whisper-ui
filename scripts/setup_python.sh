#!/usr/bin/env bash
#
# Create .venv and install the backend dependencies.
#
#   ./scripts/setup_python.sh              # runtime deps
#   ./scripts/setup_python.sh --dev        # + pytest/ruff
#   ./scripts/setup_python.sh --slim       # drop torch (~2.5 GB, unused here)
#   ./scripts/setup_python.sh --recreate   # start from a clean venv
#
set -euo pipefail
# shellcheck source=scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

WANT_DEV=0
WANT_SLIM=0
RECREATE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev) WANT_DEV=1 ;;
    --slim) WANT_SLIM=1 ;;
    --recreate) RECREATE=1 ;;
    -h | --help)
      sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) die "Unknown option: $1" ;;
  esac
  shift
done

require_macos_arm

step "Locating Python 3.11+"
PYTHON="$(find_python)" || die "No Python 3.11+ found. Install one with: brew install python@3.12"
ok "$PYTHON ($("$PYTHON" -V 2>&1))"

step "Preparing virtual environment"
if [[ "$RECREATE" == "1" && -d "$VENV_DIR" ]]; then
  info "Removing existing $VENV_DIR"
  rm -rf "$VENV_DIR"
fi
if venv_ready; then
  ok "Reusing $VENV_DIR"
else
  "$PYTHON" -m venv "$VENV_DIR"
  ok "Created $VENV_DIR"
fi

step "Installing dependencies"
"$VENV_PY" -m pip install --quiet --upgrade pip wheel
REQUIREMENTS="$REPO_ROOT/backend/requirements.txt"
[[ "$WANT_DEV" == "1" ]] && REQUIREMENTS="$REPO_ROOT/backend/requirements-dev.txt"
info "pip install -r ${REQUIREMENTS#"$REPO_ROOT"/}"
"$VENV_PY" -m pip install --requirement "$REQUIREMENTS"
ok "Dependencies installed"

if [[ "$WANT_SLIM" == "1" ]]; then
  step "Trimming unused PyTorch"
  # mlx-whisper declares torch, but only its weight-conversion module imports it.
  # Inference is pure MLX/Metal, so torch is dead weight on disk. Removed only if
  # the package still imports cleanly afterwards.
  if "$VENV_PY" -m pip show torch >/dev/null 2>&1; then
    "$VENV_PY" -m pip uninstall --yes --quiet torch || true
    if "$VENV_PY" -c "import mlx_whisper" >/dev/null 2>&1; then
      ok "Removed torch; mlx-whisper still imports cleanly"
    else
      warn "mlx-whisper needs torch after all — reinstalling"
      "$VENV_PY" -m pip install --quiet torch
    fi
  else
    info "torch is not installed; nothing to trim"
  fi
fi

step "Verifying the MLX stack"
if "$VENV_PY" - <<'PY'
import sys

try:
    import mlx.core as mx
    import mlx_whisper
    import parakeet_mlx  # noqa: F401
except Exception as exc:  # noqa: BLE001
    print(f"import failed: {type(exc).__name__}: {exc}")
    sys.exit(1)

device = mx.default_device()
# A real (tiny) Metal op proves the GPU path works, not just that imports resolve.
value = float((mx.ones((256, 256)) @ mx.ones((256, 256)))[0, 0])
assert value == 256.0, value
print(f"mlx {getattr(mx, '__version__', '?')} on {device}; mlx-whisper {getattr(mlx_whisper, '__version__', '?')}")
PY
then
  ok "Metal acceleration is working"
else
  die "MLX could not run on this Mac. See logs above; re-run with --recreate to rebuild the venv."
fi

step "Done"
dim "Interpreter: $VENV_PY"
