"""NVIDIA Parakeet TDT on MLX: the fast default engine.

Parakeet (via ``parakeet-mlx``) is a transducer, not a Whisper-style decoder. It
is several times faster than whisper-large-v3 and more accurate on English, but
it has no translation, no initial prompt and no language detection, and it
covers 25 European languages. :mod:`backend.transcriber` decides when to use it
and hands everything else to Whisper.

This module only adapts the library to the app's contract: it decodes long
audio in overlapping chunks (each boundary is a progress tick and a cancellation
point) and reshapes Parakeet's sub-word tokens into the same ``segments`` /
``words`` dict that ``mlx_whisper.transcribe`` returns, so the writers and the
UI do not know which engine ran. It deliberately does not import
:mod:`backend.transcriber`; the caller passes in callbacks instead.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, Protocol

import numpy as np

#: Languages parakeet-tdt-0.6b-v3 was trained on.
LANGUAGES = frozenset(
    {
        "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it",
        "lt", "lv", "mt", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk",
    }
)  # fmt: skip

#: Audio is decoded in windows this long. Two minutes is parakeet-mlx's own CLI
#: default: long enough for context, short enough to keep attention memory flat.
CHUNK_SECONDS = 120.0
#: Overlap between windows, re-aligned token by token when the chunks are merged.
OVERLAP_SECONDS = 15.0

#: Caption-sized segments. Whisper's segments come from its 30 s windows;
#: Parakeet has none, so cues are cut at sentence ends, pauses and these caps.
SEGMENT_MAX_WORDS = 20
SEGMENT_MAX_SECONDS = 10.0
SEGMENT_SILENCE_GAP = 1.0
_SENTENCE_END = (".", "?", "!", "…")


class Token(Protocol):
    """The fields read from ``parakeet_mlx.alignment.AlignedToken``."""

    text: str
    start: float
    end: float
    confidence: float


def is_parakeet_model(model: str) -> bool:
    return "parakeet" in model.lower()


# --------------------------------------------------------------------------
# Tokens → words → segments
# --------------------------------------------------------------------------


def group_words(tokens: Sequence[Token]) -> list[dict[str, Any]]:
    """Join sub-word tokens into Whisper-style word dicts.

    A token whose text starts with a space begins a new word (the tokenizer's
    ``▁`` marker); anything else, punctuation included, attaches to the word
    before it. Probability is the mean token confidence, as Whisper does it.
    """
    pieces: list[list[Token]] = []
    for token in tokens:
        if not pieces or token.text.startswith(" "):
            pieces.append([token])
        else:
            pieces[-1].append(token)

    words: list[dict[str, Any]] = []
    for piece in pieces:
        text = "".join(t.text for t in piece)
        if not text.strip():
            continue
        words.append(
            {
                "word": text,
                "start": float(piece[0].start),
                "end": float(piece[-1].end),
                "probability": float(np.mean([t.confidence for t in piece])),
            }
        )
    return words


def group_segments(
    words: Sequence[dict[str, Any]],
    *,
    max_words: int = SEGMENT_MAX_WORDS,
    max_seconds: float = SEGMENT_MAX_SECONDS,
    silence_gap: float = SEGMENT_SILENCE_GAP,
) -> list[dict[str, Any]]:
    """Cut words into segments at sentence ends, pauses, and the size caps.

    Works on whole words, so a segment boundary can never split one.
    """
    segments: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    def flush() -> None:
        if current:
            segments.append(
                {
                    "id": len(segments),
                    "start": current[0]["start"],
                    "end": current[-1]["end"],
                    "text": "".join(w["word"] for w in current),
                    "words": list(current),
                }
            )
            current.clear()

    for entry in words:
        if current and (
            entry["start"] - current[-1]["end"] >= silence_gap
            or len(current) >= max_words
            or entry["end"] - current[0]["start"] > max_seconds
        ):
            flush()
        current.append(entry)
        if entry["word"].rstrip().endswith(_SENTENCE_END):
            flush()
    flush()
    return segments


def to_raw(tokens: Sequence[Token], *, language: str | None) -> dict[str, Any]:
    """The ``mlx_whisper.transcribe``-shaped result :func:`build_transcript` reads."""
    segments = group_segments(group_words(tokens))
    text = "".join(segment["text"] for segment in segments).strip()
    return {"text": text, "language": language, "segments": segments}


# --------------------------------------------------------------------------
# Chunked decoding
# --------------------------------------------------------------------------


def decode_chunks(
    total_samples: int,
    decode: Callable[[int, int], list[Any]],
    merge: Callable[[list[Any], list[Any]], list[Any]],
    *,
    sample_rate: int,
    chunk_seconds: float = CHUNK_SECONDS,
    overlap_seconds: float = OVERLAP_SECONDS,
    on_progress: Callable[[float], None] | None = None,
    check_cancel: Callable[[], None] | None = None,
) -> list[Any]:
    """Decode ``[0, total_samples)`` in overlapping windows and merge the tokens.

    ``decode(start, end)`` returns tokens timed relative to ``start``; they are
    shifted to absolute time here. ``check_cancel`` runs before every window and
    stops the loop by raising.
    """
    chunk = int(chunk_seconds * sample_rate)
    step = chunk - int(overlap_seconds * sample_rate)
    if step <= 0:
        raise ValueError("overlap_seconds must be shorter than chunk_seconds")

    tokens: list[Any] = []
    start = 0
    while True:
        if check_cancel is not None:
            check_cancel()
        end = min(start + chunk, total_samples)
        offset = start / sample_rate
        shifted = [replace(t, start=t.start + offset) for t in decode(start, end)]
        if tokens and shifted:
            tokens = merge(tokens, shifted)
        elif shifted:
            tokens = shifted
        if on_progress is not None:
            on_progress(end / total_samples if total_samples else 1.0)
        if end >= total_samples:
            return tokens
        start += step


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


class _ModelHolder:
    """One resident Parakeet model, mirroring mlx-whisper's ``ModelHolder``."""

    lock = threading.Lock()
    model: Any = None
    path: str | None = None


def model_is_loaded(model: str) -> bool:
    return _ModelHolder.model is not None and _ModelHolder.path == model


def _load(model: str) -> Any:
    import mlx.core as mx  # noqa: PLC0415 - deliberately lazy
    from parakeet_mlx import from_pretrained  # noqa: PLC0415

    with _ModelHolder.lock:
        if not model_is_loaded(model):
            loaded = from_pretrained(model)
            # from_pretrained leaves the weights lazy, and a lazy MLX array is
            # bound to the streams of the thread that created it. The server
            # preloads on one thread and transcribes on another, which then
            # fails with "There is no Stream(cpu, 1) in current thread".
            # Materialise them here, as mlx-whisper's load_model does.
            mx.eval(loaded.parameters())
            _ModelHolder.model = loaded
            _ModelHolder.path = model
        return _ModelHolder.model


def run(
    samples: np.ndarray,
    model: str,
    *,
    on_progress: Callable[[float], None] | None = None,
    check_cancel: Callable[[], None] | None = None,
    on_start: Callable[[], None] | None = None,
) -> list[Any]:
    """Transcribe float32 mono 16 kHz ``samples``; returns absolute-timed tokens."""
    import mlx.core as mx  # noqa: PLC0415
    from parakeet_mlx.alignment import (  # noqa: PLC0415
        merge_longest_common_subsequence,
        merge_longest_contiguous,
    )
    from parakeet_mlx.audio import get_logmel  # noqa: PLC0415

    parakeet = _load(model)
    config = parakeet.preprocessor_config
    if on_start is not None:
        on_start()
    if len(samples) < config.hop_length:
        return []

    # get_logmel must see float32: it reinterprets the complex STFT as the input
    # dtype, so bfloat16 audio produces a mis-shaped spectrogram.
    audio = mx.array(np.asarray(samples, dtype=np.float32))

    def decode(start: int, end: int) -> list[Any]:
        if end - start < config.hop_length:
            return []
        return parakeet.generate(get_logmel(audio[start:end], config))[0].tokens

    def merge(previous: list[Any], new: list[Any]) -> list[Any]:
        try:
            return merge_longest_contiguous(previous, new, overlap_duration=OVERLAP_SECONDS)
        except RuntimeError:
            return merge_longest_common_subsequence(previous, new, overlap_duration=OVERLAP_SECONDS)

    return decode_chunks(
        len(samples),
        decode,
        merge,
        sample_rate=config.sample_rate,
        on_progress=on_progress,
        check_cancel=check_cancel,
    )
