"""Writers must be exact: subtitle timing bugs are silent and user-visible."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.formats import (
    Segment,
    Transcript,
    UnsupportedFormatError,
    build_transcript,
    format_timestamp,
    normalise_formats,
    render,
    render_srt,
    render_txt,
    render_vtt,
    unique_path,
    wrap_caption,
    write_outputs,
)
from backend.tests.conftest import RAW_RESULT

pytestmark = pytest.mark.unit


def _transcript() -> Transcript:
    return build_transcript(
        source_name="meeting.mov", raw=RAW_RESULT, model="stub/whisper-test", duration=4.0
    )


class TestBuildTranscript:
    def test_drops_blank_segments_and_reindexes(self) -> None:
        transcript = _transcript()
        assert len(transcript.segments) == 2
        assert [s.index for s in transcript.segments] == [0, 1]

    def test_keeps_word_timings(self) -> None:
        words = _transcript().segments[0].words
        assert [w.text for w in words] == ["Hello", "there."]
        assert words[0].end == pytest.approx(0.62)

    def test_text_joins_segments(self) -> None:
        assert _transcript().text == "Hello there. This is a test."

    def test_word_count(self) -> None:
        assert _transcript().word_count == 6


class TestTimestamps:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.0, "00:00:00.000"),
            (1.5, "00:00:01.500"),
            (61.25, "00:01:01.250"),
            (3661.007, "01:01:01.007"),
            (-4.0, "00:00:00.000"),
        ],
    )
    def test_format_timestamp(self, seconds: float, expected: str) -> None:
        assert format_timestamp(seconds) == expected

    def test_srt_uses_comma_separator(self) -> None:
        assert format_timestamp(1.5, decimal_separator=",") == "00:00:01,500"

    def test_rounds_to_nearest_millisecond(self) -> None:
        assert format_timestamp(0.0006) == "00:00:00.001"


class TestWrapCaption:
    def test_wraps_at_line_budget(self) -> None:
        text = "the quick brown fox jumps over the lazy dog and keeps on running"
        lines = wrap_caption(text).split("\n")
        assert len(lines) == 2
        assert len(lines[0]) <= 42

    def test_never_drops_words_when_over_budget(self) -> None:
        text = " ".join(["word"] * 60)
        wrapped = wrap_caption(text)
        assert len(wrapped.split()) == 60
        assert len(wrapped.split("\n")) == 2

    def test_empty_text(self) -> None:
        assert wrap_caption("   ") == ""


class TestRenderers:
    def test_txt_is_one_line_per_segment(self) -> None:
        assert render_txt(_transcript()) == "Hello there.\nThis is a test.\n"

    def test_srt_structure(self) -> None:
        body = render_srt(_transcript())
        blocks = body.strip().split("\n\n")
        assert len(blocks) == 2
        assert blocks[0].splitlines()[0] == "1"
        assert blocks[0].splitlines()[1] == "00:00:00,000 --> 00:00:01,440"

    def test_vtt_has_header_and_language(self) -> None:
        body = render_vtt(_transcript())
        assert body.startswith("WEBVTT\nLanguage: en\n")
        assert "00:00:01.440 --> 00:00:03.900" in body

    def test_zero_length_segment_gets_nonzero_cue(self) -> None:
        transcript = Transcript(
            source_name="a.wav", segments=(Segment(index=0, start=2.0, end=2.0, text="hi"),)
        )
        assert "00:00:02,000 --> 00:00:02,001" in render_srt(transcript)

    def test_json_round_trips(self) -> None:
        payload = json.loads(render(_transcript(), "json"))
        assert payload["language"] == "en"
        assert payload["duration"] == 4.0
        assert payload["word_count"] == 6
        assert len(payload["segments"]) == 2
        assert payload["segments"][0]["words"][0]["word"] == "Hello"

    def test_speaker_labels_render_when_present(self) -> None:
        transcript = Transcript(
            source_name="a.wav",
            segments=(
                Segment(index=0, start=0.0, end=1.0, text="Hi", speaker="Speaker 1"),
                Segment(index=1, start=1.0, end=2.0, text="Hello", speaker="Speaker 2"),
            ),
        )
        assert transcript.has_speakers
        assert "Speaker 1: Hi" in render_txt(transcript)
        assert "Speaker 2: Hello" in render_srt(transcript)

    def test_unknown_format_rejected(self) -> None:
        with pytest.raises(UnsupportedFormatError, match="docx"):
            render(_transcript(), "docx")


class TestNormaliseFormats:
    def test_defaults_to_txt(self) -> None:
        assert normalise_formats(None) == ("txt",)
        assert normalise_formats([]) == ("txt",)

    def test_dedupes_and_orders(self) -> None:
        assert normalise_formats(["srt", "txt", "SRT", ".json"]) == ("txt", "srt", "json")

    def test_rejects_unknown(self) -> None:
        with pytest.raises(UnsupportedFormatError):
            normalise_formats(["txt", "pdf"])


class TestWriteOutputs:
    def test_writes_every_requested_format(self, tmp_path: Path) -> None:
        written = write_outputs(
            _transcript(), output_dir=tmp_path, formats=["txt", "srt", "vtt", "json"]
        )
        assert set(written) == {"txt", "srt", "vtt", "json"}
        for path in written.values():
            assert path.exists() and path.read_text(encoding="utf-8")

    def test_uses_source_stem(self, tmp_path: Path) -> None:
        written = write_outputs(_transcript(), output_dir=tmp_path, formats=["txt"])
        assert written["txt"].name == "meeting.txt"

    def test_does_not_clobber_existing_files(self, tmp_path: Path) -> None:
        first = write_outputs(_transcript(), output_dir=tmp_path, formats=["txt"])["txt"]
        second = write_outputs(_transcript(), output_dir=tmp_path, formats=["txt"])["txt"]
        assert first.name == "meeting.txt"
        assert second.name == "meeting-1.txt"

    def test_overwrite_reuses_the_name(self, tmp_path: Path) -> None:
        write_outputs(_transcript(), output_dir=tmp_path, formats=["txt"], overwrite=True)
        again = write_outputs(_transcript(), output_dir=tmp_path, formats=["txt"], overwrite=True)
        assert again["txt"].name == "meeting.txt"
        assert len(list(tmp_path.glob("*.txt"))) == 1

    def test_creates_missing_output_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "deeper"
        write_outputs(_transcript(), output_dir=target, formats=["txt"])
        assert (target / "meeting.txt").exists()

    def test_sanitises_path_separators_in_names(self, tmp_path: Path) -> None:
        path = unique_path(tmp_path, "some/nested:name", "txt")
        assert path.parent == tmp_path
        assert path.name == "some_nested_name.txt"
