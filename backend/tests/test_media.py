"""ffmpeg integration: probing, extraction, progress, cancellation, errors."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from backend import media
from backend.config import SAMPLE_RATE
from backend.tests.conftest import requires_ffmpeg


@pytest.mark.unit
class TestToolDiscovery:
    def test_finds_ffmpeg_outside_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(media.shutil, "which", lambda _name: None)
        found = media.find_tool("ffmpeg")
        # Either Homebrew's copy is discovered by the fallback, or nothing is.
        assert found is None or Path(found).name == "ffmpeg"

    def test_missing_tool_error_tells_the_user_what_to_do(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(media, "find_tool", lambda _name: None)
        with pytest.raises(media.FFmpegNotFoundError, match="brew install ffmpeg"):
            media.require_tool("ffmpeg")


@pytest.mark.unit
class TestLoadWav:
    def test_returns_normalised_float32_mono(self, silent_wav: Path) -> None:
        samples = media.load_pcm16_wav(silent_wav)
        assert samples.dtype.name == "float32"
        assert samples.ndim == 1
        assert len(samples) == SAMPLE_RATE * 2
        assert abs(samples).max() <= 1.0

    def test_duration_helper(self, silent_wav: Path) -> None:
        assert media.audio_duration(media.load_pcm16_wav(silent_wav)) == pytest.approx(2.0)


@requires_ffmpeg
@pytest.mark.integration
class TestProbe:
    def test_detects_video_with_audio(self, tone_mp4: Path) -> None:
        info = media.probe(tone_mp4)
        assert info.has_video and info.has_audio
        assert info.is_video
        assert info.duration == pytest.approx(2.0, abs=0.3)
        assert info.size_bytes > 0

    def test_audio_only_is_not_video(self, silent_wav: Path) -> None:
        info = media.probe(silent_wav)
        assert info.has_audio
        assert not info.is_video
        assert info.audio_codec == "pcm_s16le"

    def test_unreadable_file_raises_readable_error(self, tmp_path: Path) -> None:
        junk = tmp_path / "broken.mp4"
        junk.write_bytes(b"not a video")
        with pytest.raises(media.MediaError, match="broken.mp4"):
            media.probe(junk)


@requires_ffmpeg
@pytest.mark.integration
class TestExtractAudio:
    def test_extracts_16k_mono_from_video(self, tone_mp4: Path, tmp_path: Path) -> None:
        target = tmp_path / "out" / "audio.wav"
        media.extract_audio(tone_mp4, target, duration=2.0)

        assert target.exists()
        info = media.probe(target)
        assert info.audio_codec == "pcm_s16le"
        samples = media.load_pcm16_wav(target)
        assert media.audio_duration(samples) == pytest.approx(2.0, abs=0.2)

    def test_reports_monotonic_progress_ending_at_one(
        self, tone_mp4: Path, tmp_path: Path
    ) -> None:
        seen: list[float] = []
        media.extract_audio(
            tone_mp4, tmp_path / "a.wav", duration=2.0, on_progress=seen.append
        )
        assert seen and seen[-1] == 1.0
        assert seen == sorted(seen)
        assert all(0.0 <= value <= 1.0 for value in seen)

    def test_cancellation_stops_and_removes_partial_file(
        self, tone_mp4: Path, tmp_path: Path
    ) -> None:
        target = tmp_path / "cancelled.wav"
        flag = threading.Event()
        flag.set()
        with pytest.raises(media.OperationCancelled):
            media.extract_audio(
                tone_mp4, target, duration=2.0, should_cancel=flag.is_set
            )
        assert not target.exists()

    def test_video_without_audio_track_fails_clearly(self, tmp_path: Path) -> None:
        import subprocess

        from backend.tests.conftest import FFMPEG

        silent_video = tmp_path / "mute.mp4"
        subprocess.run(
            [
                FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=black:s=128x96:d=1",
                "-pix_fmt", "yuv420p", str(silent_video),
            ],
            check=True,
        )
        with pytest.raises(media.MediaError, match="(?i)no audio"):
            media.extract_audio(silent_video, tmp_path / "x.wav", duration=1.0)
