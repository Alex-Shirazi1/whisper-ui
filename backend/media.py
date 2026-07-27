"""ffmpeg-backed media inspection and audio extraction.

Whisper only ever sees 16 kHz mono float32 audio, so every input — a 4K screen
recording, a voice memo, a FLAC — is funnelled through the same normalisation
step here. Keeping ffmpeg behind this module means the transcription code never
has to know what a container is.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import wave
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import SAMPLE_RATE, VIDEO_EXTENSIONS
from .logging_setup import get_logger

log = get_logger(__name__)

ProgressCallback = Callable[[float], None]
CancelCheck = Callable[[], bool]

#: A GUI-launched app inherits a minimal PATH that usually lacks Homebrew, so
#: look in the standard install locations too before giving up.
_EXTRA_BIN_DIRS = ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/opt/local/bin")


class MediaError(RuntimeError):
    """Any failure while inspecting or decoding a media file.

    The message is written for a human reading it in the app UI, not for a log
    grep — keep it that way.
    """


class FFmpegNotFoundError(MediaError):
    pass


class OperationCancelled(MediaError):
    pass


# --------------------------------------------------------------------------
# Tool discovery
# --------------------------------------------------------------------------


def find_tool(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for directory in _EXTRA_BIN_DIRS:
        candidate = Path(directory) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def require_tool(name: str) -> str:
    path = find_tool(name)
    if path is None:
        raise FFmpegNotFoundError(
            f"{name} was not found on this Mac. Install it with:\n\n"
            f"    brew install ffmpeg\n\n"
            "then quit and reopen WhisperMac."
        )
    return path


def ffmpeg_version() -> str | None:
    """First line of ``ffmpeg -version``, or ``None`` when unavailable."""
    path = find_tool("ffmpeg")
    if path is None:
        return None
    try:
        proc = subprocess.run(
            [path, "-version"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    first = (proc.stdout or "").splitlines()
    return first[0].strip() if first else None


# --------------------------------------------------------------------------
# Probing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    duration: float | None
    has_video: bool
    has_audio: bool
    audio_codec: str | None
    video_codec: str | None
    size_bytes: int

    @property
    def is_video(self) -> bool:
        return self.has_video or self.path.suffix.lower() in VIDEO_EXTENSIONS


def probe(path: Path) -> MediaInfo:
    """Read stream metadata with ffprobe."""
    ffprobe = require_tool("ffprobe")
    command: Sequence[str] = (
        ffprobe,
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    )
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - pathological input
        raise MediaError(f"Timed out while inspecting {path.name}.") from exc

    if proc.returncode != 0:
        detail = (proc.stderr or "").strip().splitlines()
        hint = detail[-1] if detail else "unknown ffprobe error"
        raise MediaError(f"{path.name} could not be read as audio or video ({hint}).")

    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise MediaError(f"{path.name} produced unreadable metadata.") from exc

    streams: Iterable[dict] = payload.get("streams") or ()
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    video = [
        s
        for s in streams
        if s.get("codec_type") == "video"
        # Cover art in an mp3 shows up as a video stream; ignore it.
        and s.get("disposition", {}).get("attached_pic", 0) != 1
    ]

    return MediaInfo(
        path=path,
        duration=_first_duration(payload, audio),
        has_video=bool(video),
        has_audio=bool(audio),
        audio_codec=audio[0].get("codec_name") if audio else None,
        video_codec=video[0].get("codec_name") if video else None,
        size_bytes=path.stat().st_size if path.exists() else 0,
    )


def _first_duration(payload: dict, audio_streams: Sequence[dict]) -> float | None:
    candidates = [payload.get("format", {}).get("duration")]
    candidates += [s.get("duration") for s in audio_streams]
    for raw in candidates:
        try:
            value = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def extract_audio(
    source: Path,
    destination: Path,
    *,
    duration: float | None = None,
    on_progress: ProgressCallback | None = None,
    should_cancel: CancelCheck | None = None,
) -> Path:
    """Decode ``source`` into a 16 kHz mono 16-bit WAV at ``destination``.

    Progress is read from ffmpeg's ``-progress`` stream, so the reported value
    reflects real decode position rather than a guess.
    """
    ffmpeg = require_tool("ffmpeg")
    destination.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-progress", "pipe:1",
        "-y",
        "-i", str(source),
        "-vn", "-sn", "-dn",          # drop video, subtitles, data
        "-map", "0:a:0?",             # first audio stream only
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        "-f", "wav",
        str(destination),
    ]

    log.debug("Extracting audio: %s", " ".join(command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    try:
        _pump_progress(process, duration, on_progress, should_cancel)
    except OperationCancelled:
        _terminate(process)
        destination.unlink(missing_ok=True)
        raise

    stderr = (process.stderr.read() if process.stderr else "") or ""
    returncode = process.wait()

    if returncode != 0:
        destination.unlink(missing_ok=True)
        raise MediaError(_explain_ffmpeg_failure(source, stderr))

    if not destination.exists() or destination.stat().st_size <= 44:  # header only
        destination.unlink(missing_ok=True)
        raise MediaError(
            f"No audio could be extracted from {source.name}. "
            "The file may be silent, video-only, or corrupt."
        )

    if on_progress is not None:
        on_progress(1.0)
    return destination


def _pump_progress(
    process: subprocess.Popen,
    duration: float | None,
    on_progress: ProgressCallback | None,
    should_cancel: CancelCheck | None,
) -> None:
    if process.stdout is None:  # pragma: no cover - defensive
        return
    for line in process.stdout:
        if should_cancel is not None and should_cancel():
            raise OperationCancelled("Audio extraction cancelled.")
        if on_progress is None or not duration or duration <= 0:
            continue
        key, _, value = line.strip().partition("=")
        if key != "out_time_us":
            continue
        try:
            seconds = int(value) / 1_000_000
        except ValueError:
            continue
        on_progress(max(0.0, min(1.0, seconds / duration)))


def _terminate(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover - stubborn child
        process.kill()
        process.wait(timeout=5)


def _explain_ffmpeg_failure(source: Path, stderr: str) -> str:
    lowered = stderr.lower()
    if "does not contain any stream" in lowered or "stream map" in lowered:
        return f"{source.name} has no audio track to transcribe."
    if "permission denied" in lowered:
        return f"macOS blocked access to {source.name}. Move it out of a protected folder and retry."
    if "invalid data found" in lowered or "moov atom not found" in lowered:
        return f"{source.name} appears to be corrupt or still being written."
    tail = [line for line in stderr.strip().splitlines() if line.strip()]
    detail = tail[-1] if tail else "ffmpeg exited with an error"
    return f"Could not extract audio from {source.name}: {detail}"


# --------------------------------------------------------------------------
# WAV -> float32 samples
# --------------------------------------------------------------------------


def load_pcm16_wav(path: Path) -> np.ndarray:
    """Load a 16-bit PCM WAV as the float32 mono array Whisper expects.

    Reading the file we just wrote ourselves avoids a second ffmpeg decode pass
    inside mlx-whisper's ``load_audio``.
    """
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise MediaError(f"{path.name} is not 16-bit PCM audio.")
        channels = handle.getnchannels()
        frames = handle.readframes(handle.getnframes())

    samples = np.frombuffer(frames, dtype="<i2")
    if channels > 1:  # pragma: no cover - extraction forces mono
        samples = samples.reshape(-1, channels).mean(axis=1)
    return (samples.astype(np.float32) / 32768.0).copy()


def audio_duration(samples: np.ndarray) -> float:
    return float(len(samples)) / SAMPLE_RATE
