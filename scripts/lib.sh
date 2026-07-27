#!/usr/bin/env bash
# Shared helpers for the WhisperMac scripts. Source, don't execute.
#
# Several paths below are read only by the scripts that source this file, so the
# linter cannot see their use from here.
# shellcheck disable=SC2034

# Colour only when we're attached to a terminal.
if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'; C_DIM=$'\033[2m'; C_BOLD=$'\033[1m'
  C_RED=$'\033[31m'; C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_BLUE=$'\033[34m'
else
  C_RESET=''; C_DIM=''; C_BOLD=''; C_RED=''; C_GREEN=''; C_YELLOW=''; C_BLUE=''
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
VENV_PY="$VENV_DIR/bin/python"
APP_DIR="$REPO_ROOT/app"
BUILD_DIR="$REPO_ROOT/build"
APP_BUNDLE="$BUILD_DIR/WhisperMac.app"
LOG_DIR="$REPO_ROOT/logs"
SESSION_FILE="$HOME/.whisper-mac/session.json"
DEFAULT_PORT="${WHISPER_MAC_PORT:-8756}"

# Homebrew lives outside the PATH a GUI-launched process inherits.
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

step()  { printf '\n%s==>%s %s%s%s\n' "$C_BLUE" "$C_RESET" "$C_BOLD" "$1" "$C_RESET"; }
info()  { printf '    %s\n' "$1"; }
ok()    { printf '    %s✓%s %s\n' "$C_GREEN" "$C_RESET" "$1"; }
warn()  { printf '    %s!%s %s\n' "$C_YELLOW" "$C_RESET" "$1"; }
fail()  { printf '    %s✗%s %s\n' "$C_RED" "$C_RESET" "$1" >&2; }
dim()   { printf '    %s%s%s\n' "$C_DIM" "$1" "$C_RESET"; }

die() {
  fail "$1"
  exit "${2:-1}"
}

have() { command -v "$1" >/dev/null 2>&1; }

is_tty() { [[ -t 0 ]]; }

# confirm "question" -> 0 = yes. Honours ASSUME_YES for unattended runs.
confirm() {
  if [[ "${ASSUME_YES:-0}" == "1" ]]; then return 0; fi
  if ! is_tty; then return 1; fi
  local reply
  printf '    %s [y/N] ' "$1"
  read -r reply
  [[ "$reply" =~ ^([yY]|[yY][eE][sS])$ ]]
}

require_macos_arm() {
  [[ "$(uname -s)" == "Darwin" ]] || die "WhisperMac is macOS only (found $(uname -s))."
  if [[ "$(uname -m)" != "arm64" ]]; then
    die "WhisperMac needs Apple Silicon (M1 or newer). MLX has no Intel build."
  fi
}

# Newest suitable interpreter: 3.11+ is required by the backend.
find_python() {
  local candidate
  for candidate in python3.13 python3.12 python3.11 python3; do
    if have "$candidate" && "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
    then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

venv_ready() {
  [[ -x "$VENV_PY" ]]
}

backend_url() {
  if [[ -f "$SESSION_FILE" ]] && have python3; then
    python3 - "$SESSION_FILE" <<'PY' 2>/dev/null || true
import json, sys
try:
    print(json.load(open(sys.argv[1]))["url"])
except Exception:
    pass
PY
  fi
}

# Is a healthy backend answering? Prefers the session file's URL.
backend_healthy() {
  local url
  url="$(backend_url)"
  url="${url:-http://127.0.0.1:$DEFAULT_PORT}"
  curl --silent --show-error --fail --max-time 2 "$url/health" >/dev/null 2>&1
}

wait_for_backend() {
  local attempts="${1:-60}" i
  for ((i = 0; i < attempts; i++)); do
    if backend_healthy; then return 0; fi
    sleep 0.5
  done
  return 1
}
