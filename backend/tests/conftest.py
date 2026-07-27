"""Shared fixtures.

The whole suite runs without MLX and without model weights: the engine is
replaced by :class:`StubEngine`, which returns the same dict shape
``mlx_whisper.transcribe`` does and drives the same progress and cancellation
callbacks.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
import wave
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.config import SAMPLE_RATE, Settings  # noqa: E402
from backend.transcriber import (  # noqa: E402
    EngineStatus,
    TranscribeOptions,
    TranscribeResult,
    TranscriptionCancelled,
)

FFMPEG = shutil.which("ffmpeg") or shutil.which("/opt/homebrew/bin/ffmpeg")
requires_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg is not installed")

RAW_RESULT = {
    "text": " Hello there. This is a test.",
    "language": "en",
    "segments": [
        {
            "id": 0,
            "seek": 0,
            "start": 0.0,
            "end": 1.44,
            "text": " Hello there.",
            "tokens": [50364, 2425],
            "temperature": 0.0,
            "avg_logprob": -0.21,
            "compression_ratio": 1.1,
            "no_speech_prob": 0.01,
            "words": [
                {"word": " Hello", "start": 0.0, "end": 0.62, "probability": 0.98},
                {"word": " there.", "start": 0.62, "end": 1.44, "probability": 0.95},
            ],
        },
        {
            "id": 1,
            "seek": 0,
            "start": 1.44,
            "end": 3.9,
            "text": " This is a test.",
            "tokens": [50464, 1156],
            "temperature": 0.0,
            "avg_logprob": -0.18,
            "compression_ratio": 1.2,
            "no_speech_prob": 0.02,
            "words": [],
        },
        # Empty segments show up in real output; they must never reach a file.
        {"id": 2, "start": 3.9, "end": 4.0, "text": "   ", "tokens": [], "words": []},
    ],
}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return replace(
        Settings(),
        output_dir=tmp_path / "output",
        model_cache_dir=tmp_path / "models",
        temp_dir=tmp_path / "tmp",
        log_dir=tmp_path / "logs",
        model="stub/whisper-test",
        require_token=True,
        token="test-token",
        job_retention_seconds=3600,
        estimated_realtime_factor=0.1,
    )


class StubEngine:
    """Engine double: real progress and cancellation, no weights, no network."""

    def __init__(self, *, fail_with: Exception | None = None, steps: int = 4, step_delay: float = 0.0):
        self.fail_with = fail_with
        self.steps = steps
        self.step_delay = step_delay
        self.calls: list[TranscribeOptions] = []

    def model_is_local(self, model: str) -> bool:
        return True

    def model_is_loaded(self, model: str) -> bool:
        return True

    def ensure_model(self, model: str) -> str:
        return model

    def status(self) -> EngineStatus:
        return EngineStatus(available=True, device="stub", mlx_version="stub")

    def transcribe(
        self,
        samples: np.ndarray,
        options: TranscribeOptions,
        *,
        on_progress=None,
        should_cancel=None,
        on_inference_start=None,
    ) -> TranscribeResult:
        self.calls.append(options)
        if on_inference_start is not None:
            on_inference_start()
        if self.fail_with is not None:
            raise self.fail_with
        for step in range(1, self.steps + 1):
            if should_cancel is not None and should_cancel():
                raise TranscriptionCancelled("cancelled")
            if self.step_delay:
                time.sleep(self.step_delay)
            if on_progress is not None:
                on_progress(step / self.steps)
        return TranscribeResult(
            raw=RAW_RESULT, language="en", model=options.model, options=options
        )


@pytest.fixture
def engine() -> StubEngine:
    return StubEngine()


@pytest.fixture
def silent_wav(tmp_path: Path) -> Path:
    """A real 16 kHz mono 16-bit WAV — no ffmpeg required to create it."""
    path = tmp_path / "silence.wav"
    samples = (np.sin(np.linspace(0, 220 * 2 * np.pi, SAMPLE_RATE * 2)) * 8000).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(samples.tobytes())
    return path


@pytest.fixture
def tone_mp4(tmp_path: Path) -> Iterator[Path]:
    """A tiny video with an audio track, built by ffmpeg."""
    if FFMPEG is None:
        pytest.skip("ffmpeg is not installed")
    path = tmp_path / "clip.mp4"
    subprocess.run(
        [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-f", "lavfi", "-i", "color=c=navy:s=128x96:d=2",
            "-shortest", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )
    yield path
