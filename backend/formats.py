"""Transcript data model and output writers (txt / srt / vtt / json).

The data model is deliberately richer than what any single format renders:
segments carry an optional ``speaker`` and word-level timings so that speaker
diarization and subtitle editing can be added later without touching the
transcription path or breaking existing files.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from .config import OUTPUT_FORMATS

#: Subtitle line budget. 42 characters over 2 lines is the Netflix-style
#: convention and reads comfortably at any window size.
CAPTION_LINE_CHARS = 42
CAPTION_MAX_LINES = 2


class UnsupportedFormatError(ValueError):
    pass


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str
    probability: float | None = None

    def to_dict(self) -> dict:
        payload = {"start": round(self.start, 3), "end": round(self.end, 3), "word": self.text}
        if self.probability is not None:
            payload["probability"] = round(self.probability, 4)
        return payload


@dataclass(frozen=True)
class Segment:
    index: int
    start: float
    end: float
    text: str
    #: Reserved for diarization. ``None`` today; ``"SPEAKER_01"`` tomorrow.
    speaker: str | None = None
    words: tuple[Word, ...] = ()
    no_speech_prob: float | None = None
    avg_logprob: float | None = None

    @property
    def display_text(self) -> str:
        body = self.text.strip()
        return f"{self.speaker}: {body}" if self.speaker else body

    def to_dict(self) -> dict:
        payload: dict = {
            "id": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text.strip(),
        }
        if self.speaker:
            payload["speaker"] = self.speaker
        if self.words:
            payload["words"] = [w.to_dict() for w in self.words]
        if self.no_speech_prob is not None:
            payload["no_speech_prob"] = round(self.no_speech_prob, 4)
        if self.avg_logprob is not None:
            payload["avg_logprob"] = round(self.avg_logprob, 4)
        return payload


@dataclass(frozen=True)
class Transcript:
    source_name: str
    segments: tuple[Segment, ...]
    language: str | None = None
    duration: float | None = None
    model: str = ""
    task: str = "transcribe"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    extra: Mapping[str, object] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip()).strip()

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def has_speakers(self) -> bool:
        return any(s.speaker for s in self.segments)

    def with_segments(self, segments: Iterable[Segment]) -> Transcript:
        """Return a copy carrying new segments (used by post-processors)."""
        return replace(self, segments=tuple(segments))


def build_transcript(
    *,
    source_name: str,
    raw: Mapping[str, object],
    model: str,
    task: str = "transcribe",
    duration: float | None = None,
) -> Transcript:
    """Convert an ``mlx_whisper.transcribe`` result dict into a Transcript."""
    segments: list[Segment] = []
    for item in raw.get("segments") or []:  # type: ignore[union-attr]
        if not isinstance(item, Mapping):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        segments.append(
            Segment(
                index=len(segments),
                start=_as_float(item.get("start"), 0.0),
                end=_as_float(item.get("end"), 0.0),
                text=text,
                words=_build_words(item.get("words")),
                no_speech_prob=_as_optional_float(item.get("no_speech_prob")),
                avg_logprob=_as_optional_float(item.get("avg_logprob")),
            )
        )

    return Transcript(
        source_name=source_name,
        segments=tuple(segments),
        language=str(raw.get("language")) if raw.get("language") else None,
        duration=duration,
        model=model,
        task=task,
    )


def _build_words(raw_words: object) -> tuple[Word, ...]:
    if not isinstance(raw_words, Sequence):
        return ()
    words: list[Word] = []
    for entry in raw_words:
        if not isinstance(entry, Mapping):
            continue
        text = str(entry.get("word", "")).strip()
        if not text:
            continue
        words.append(
            Word(
                start=_as_float(entry.get("start"), 0.0),
                end=_as_float(entry.get("end"), 0.0),
                text=text,
                probability=_as_optional_float(entry.get("probability")),
            )
        )
    return tuple(words)


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_optional_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Timestamps & wrapping
# --------------------------------------------------------------------------


def format_timestamp(seconds: float, *, decimal_separator: str = ".", always_hours: bool = True) -> str:
    seconds = max(0.0, seconds)
    milliseconds = int(round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1_000)
    stamp = f"{minutes:02d}:{secs:02d}{decimal_separator}{milliseconds:03d}"
    if always_hours or hours:
        stamp = f"{hours:02d}:{stamp}"
    return stamp


def wrap_caption(
    text: str, *, max_chars: int = CAPTION_LINE_CHARS, max_lines: int = CAPTION_MAX_LINES
) -> str:
    """Soft-wrap caption text without ever dropping words.

    Once ``max_lines`` is reached the remainder is appended to the final line —
    an overlong subtitle is annoying, a truncated one is a bug.
    """
    words = text.split()
    if not words:
        return ""

    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars and len(lines) < max_lines - 1:
            lines.append(current)
            current = word
        else:
            current = candidate
    lines.append(current)
    return "\n".join(line for line in lines if line)


# --------------------------------------------------------------------------
# Renderers
# --------------------------------------------------------------------------


def render_txt(transcript: Transcript) -> str:
    if transcript.has_speakers:
        body = "\n\n".join(s.display_text for s in transcript.segments if s.text.strip())
    else:
        body = "\n".join(s.text.strip() for s in transcript.segments if s.text.strip())
    return body.rstrip() + "\n" if body else ""


def render_srt(transcript: Transcript) -> str:
    blocks: list[str] = []
    for number, segment in enumerate(transcript.segments, start=1):
        if not segment.text.strip():
            continue
        start = format_timestamp(segment.start, decimal_separator=",")
        end = format_timestamp(max(segment.end, segment.start + 0.001), decimal_separator=",")
        blocks.append(f"{number}\n{start} --> {end}\n{wrap_caption(segment.display_text)}")
    return "\n\n".join(blocks) + "\n" if blocks else ""


def render_vtt(transcript: Transcript) -> str:
    header = "WEBVTT\n"
    if transcript.language:
        header += f"Language: {transcript.language}\n"
    cues: list[str] = []
    for segment in transcript.segments:
        if not segment.text.strip():
            continue
        start = format_timestamp(segment.start)
        end = format_timestamp(max(segment.end, segment.start + 0.001))
        cues.append(f"{start} --> {end}\n{wrap_caption(segment.display_text)}")
    return header + "\n" + "\n\n".join(cues) + "\n" if cues else header + "\n"


def render_json(transcript: Transcript) -> str:
    payload = {
        "source": transcript.source_name,
        "model": transcript.model,
        "task": transcript.task,
        "language": transcript.language,
        "duration": round(transcript.duration, 3) if transcript.duration else None,
        "created_at": transcript.created_at,
        "word_count": transcript.word_count,
        "text": transcript.text,
        "segments": [segment.to_dict() for segment in transcript.segments],
    }
    if transcript.extra:
        payload["extra"] = dict(transcript.extra)
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


_RENDERERS = {
    "txt": render_txt,
    "srt": render_srt,
    "vtt": render_vtt,
    "json": render_json,
}


def render(transcript: Transcript, output_format: str) -> str:
    key = output_format.lower().lstrip(".")
    try:
        return _RENDERERS[key](transcript)
    except KeyError as exc:
        raise UnsupportedFormatError(
            f"'{output_format}' is not a supported output format. "
            f"Choose from: {', '.join(OUTPUT_FORMATS)}."
        ) from exc


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------


def normalise_formats(formats: Iterable[str] | None) -> tuple[str, ...]:
    """Validate, de-duplicate and order requested formats."""
    requested = [f.lower().lstrip(".") for f in (formats or ()) if f and f.strip()]
    if not requested:
        return ("txt",)
    unknown = [f for f in requested if f not in _RENDERERS]
    if unknown:
        raise UnsupportedFormatError(
            f"Unsupported output format(s): {', '.join(sorted(set(unknown)))}. "
            f"Choose from: {', '.join(OUTPUT_FORMATS)}."
        )
    return tuple(f for f in OUTPUT_FORMATS if f in set(requested))


def unique_path(directory: Path, stem: str, extension: str) -> Path:
    """``meeting.txt`` -> ``meeting-1.txt`` when the name is already taken."""
    safe_stem = sanitise_stem(stem)
    candidate = directory / f"{safe_stem}.{extension}"
    counter = 1
    while candidate.exists():
        candidate = directory / f"{safe_stem}-{counter}.{extension}"
        counter += 1
    return candidate


def sanitise_stem(stem: str) -> str:
    cleaned = "".join("_" if ch in '/\\:*?"<>|' else ch for ch in stem).strip(" .")
    return (cleaned or "transcript")[:180]


def write_outputs(
    transcript: Transcript,
    *,
    output_dir: Path,
    formats: Iterable[str],
    stem: str | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Render and write every requested format. Returns ``{format: path}``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    base = stem or Path(transcript.source_name).stem
    written: dict[str, Path] = {}

    for output_format in normalise_formats(formats):
        body = render(transcript, output_format)
        if overwrite:
            target = output_dir / f"{sanitise_stem(base)}.{output_format}"
        else:
            target = unique_path(output_dir, base, output_format)
        target.write_text(body, encoding="utf-8")
        written[output_format] = target

    return written
