"""Loopback session token.

The server binds to 127.0.0.1, but "local" is not the same as "private" — any
process or web page on the machine could otherwise POST a file path and read
back the transcript. A token written to a 0600 file in ``~/.whisper-mac`` keeps
the API usable by the app (which reads the file) and closed to everything else,
with zero setup for the user.
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Annotated

from fastapi import Header, HTTPException, Request, status

from .config import SESSION_FILE, STATE_DIR, Settings
from .logging_setup import get_logger

log = get_logger(__name__)

TOKEN_HEADER = "X-Whisper-Token"


def generate_token() -> str:
    return secrets.token_urlsafe(32)


def write_session_file(settings: Settings, *, version: str) -> Path:
    """Publish connection details for the macOS app to pick up."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.chmod(0o700)
    payload = {
        "url": settings.base_url,
        "token": settings.token,
        "pid": os.getpid(),
        "model": settings.model,
        "output_dir": str(settings.output_dir),
        "version": version,
    }
    # Write-then-rename so a reader never sees a half-written file.
    temp = SESSION_FILE.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.chmod(0o600)
    temp.replace(SESSION_FILE)
    log.info("Session details written to %s", SESSION_FILE)
    return SESSION_FILE


def clear_session_file() -> None:
    try:
        if SESSION_FILE.exists() and json.loads(SESSION_FILE.read_text()).get("pid") == os.getpid():
            SESSION_FILE.unlink()
    except (OSError, json.JSONDecodeError):  # pragma: no cover - best effort
        pass


def read_session_file() -> dict | None:
    try:
        return json.loads(SESSION_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


async def require_token(
    request: Request,
    x_whisper_token: Annotated[str | None, Header(alias=TOKEN_HEADER)] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    """FastAPI dependency enforcing the session token."""
    settings: Settings = request.app.state.settings
    if not settings.require_token:
        return

    presented = x_whisper_token
    if not presented and authorization and authorization.lower().startswith("bearer "):
        presented = authorization[7:].strip()

    if not presented or not secrets.compare_digest(presented, settings.token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Missing or invalid session token. The WhisperMac app reads it from "
                f"{SESSION_FILE}; for manual requests send the "
                f"{TOKEN_HEADER} header."
            ),
        )
