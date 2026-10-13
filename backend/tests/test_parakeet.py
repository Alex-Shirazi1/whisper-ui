"""Parakeet adapter: token → word → segment grouping, chunked decoding, routing.

Parakeet emits sub-word tokens with their own timings rather than Whisper's
segment dicts, so everything the writers and the app rely on (words that are
whole words, caption-sized segments, offsets that survive chunking) is built in
:mod:`backend.parakeet`. None of these tests need MLX or weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from backend import parakeet
from backend.formats import build_transcript, render_srt

pytestmark = pytest.mark.unit


@dataclass
class Tok:
    """Shaped like ``parakeet_mlx.alignment.AlignedToken``: ``end`` derives from duration."""

    text: str
    start: float
    duration: float
    confidence: float = 1.0
    end: float = field(default=0.0)

    def __post_init__(self) -> None:
        self.end = self.start + self.duration


def tok(text: str, start: float, end: float, confidence: float = 1.0) -> Tok:
    return Tok(text=text, start=start, duration=end - start, confidence=confidence)


def word(text: str, start: float, end: float) -> dict:
    return {"word": text, "start": start, "end": end, "probability": 1.0}


# --------------------------------------------------------------------------
# Words
# --------------------------------------------------------------------------


class TestGroupWords:
    def test_subword_pieces_join_into_one_word(self) -> None:
        words = parakeet.group_words(
            [
                tok(" Hel", 0.0, 0.2, 0.9),
                tok("lo", 0.2, 0.4, 0.7),
                tok(" world", 0.5, 0.9),
                tok(".", 0.9, 1.0),
            ]
        )

        assert [w["word"] for w in words] == [" Hello", " world."]
        assert (words[0]["start"], words[0]["end"]) == (0.0, 0.4)
        assert words[0]["probability"] == pytest.approx(0.8)

    def test_a_first_token_without_a_leading_space_still_starts_a_word(self) -> None:
        words = parakeet.group_words([tok("Yes", 0.0, 0.3), tok(",", 0.3, 0.35)])
        assert [w["word"] for w in words] == ["Yes,"]

    def test_whitespace_only_tokens_never_become_words(self) -> None:
        words = parakeet.group_words([tok(" ", 0.0, 0.1), tok(" ok", 0.1, 0.3)])
        assert [w["word"] for w in words] == [" ok"]

    def test_no_tokens_means_no_words(self) -> None:
        assert parakeet.group_words([]) == []


# --------------------------------------------------------------------------
# Segments
# --------------------------------------------------------------------------


class TestGroupSegments:
    def test_splits_after_sentence_punctuation(self) -> None:
        segments = parakeet.group_segments(
            [word(" Hi.", 0, 1), word(" Are", 1, 2), word(" you?", 2, 3), word(" Yes!", 3, 4)]
        )
        assert [s["text"] for s in segments] == [" Hi.", " Are you?", " Yes!"]

    def test_splits_on_a_long_silence(self) -> None:
        segments = parakeet.group_segments(
            [word(" one", 0, 1), word(" two", 1, 2), word(" three", 9, 10)], silence_gap=1.0
        )
        assert [s["text"] for s in segments] == [" one two", " three"]

    def test_caps_words_per_segment(self) -> None:
        words = [word(f" w{i}", i * 0.1, i * 0.1 + 0.1) for i in range(5)]
        segments = parakeet.group_segments(words, max_words=2, max_seconds=100)
        assert [len(s["words"]) for s in segments] == [2, 2, 1]

    def test_caps_segment_duration(self) -> None:
        words = [word(f" w{i}", float(i), i + 0.9) for i in range(6)]
        segments = parakeet.group_segments(words, max_words=100, max_seconds=3.0, silence_gap=5)
        assert all(s["end"] - s["start"] <= 3.0 for s in segments)
        assert sum(len(s["words"]) for s in segments) == 6

    def test_segments_span_their_words(self) -> None:
        (segment,) = parakeet.group_segments([word(" a", 1.0, 1.5), word(" b.", 1.5, 2.25)])
        assert (segment["start"], segment["end"]) == (1.0, 2.25)
        assert segment["id"] == 0

    def test_no_words_means_no_segments(self) -> None:
        assert parakeet.group_segments([]) == []


# --------------------------------------------------------------------------
# Result shape
# --------------------------------------------------------------------------


class TestToRaw:
    def test_matches_the_shape_the_writers_consume(self) -> None:
        tokens = [tok(" Hello", 0.0, 0.5), tok(" there.", 0.5, 1.0), tok(" Bye.", 2.0, 2.4)]

        raw = parakeet.to_raw(tokens, language="en")
        transcript = build_transcript(source_name="a.wav", raw=raw, model="parakeet")

        assert raw["text"] == "Hello there. Bye."
        assert transcript.language == "en"
        assert [s.text for s in transcript.segments] == ["Hello there.", "Bye."]
        assert [w.text for w in transcript.segments[0].words] == ["Hello", "there."]
        assert "00:00:02,000 --> 00:00:02,400" in render_srt(transcript)

    def test_silence_produces_an_empty_transcript_not_an_error(self) -> None:
        raw = parakeet.to_raw([], language=None)
        assert raw == {"text": "", "language": None, "segments": []}


# --------------------------------------------------------------------------
# Chunked decoding
# --------------------------------------------------------------------------


def _concat(previous: list, new: list) -> list:
    return previous + new


class TestDecodeChunks:
    def test_short_audio_is_decoded_in_one_pass(self) -> None:
        calls: list[tuple[int, int]] = []

        def decode(start: int, end: int) -> list:
            calls.append((start, end))
            return [tok(" hi", 0.5, 0.9)]

        tokens = parakeet.decode_chunks(
            1_000, decode, _concat, sample_rate=100, chunk_seconds=20, overlap_seconds=5
        )

        assert calls == [(0, 1_000)]
        assert [(t.start, t.end) for t in tokens] == [(0.5, 0.9)]

    def test_long_audio_is_chunked_with_overlap_and_absolute_times(self) -> None:
        calls: list[tuple[int, int]] = []

        def decode(start: int, end: int) -> list:
            calls.append((start, end))
            return [tok(" x", 1.0, 1.5)]

        tokens = parakeet.decode_chunks(
            5_000, decode, _concat, sample_rate=100, chunk_seconds=20, overlap_seconds=5
        )

        assert calls == [(0, 2_000), (1_500, 3_500), (3_000, 5_000)]
        assert [t.start for t in tokens] == [1.0, 16.0, 31.0]
        assert [t.end for t in tokens] == [1.5, 16.5, 31.5]

    def test_reports_monotonic_progress_ending_at_one(self) -> None:
        seen: list[float] = []
        parakeet.decode_chunks(
            5_000,
            lambda s, e: [],
            _concat,
            sample_rate=100,
            chunk_seconds=20,
            overlap_seconds=5,
            on_progress=seen.append,
        )
        assert seen == sorted(seen)
        assert seen[-1] == 1.0
        assert len(seen) == 3

    def test_checks_for_cancellation_before_every_chunk(self) -> None:
        decoded: list[int] = []

        class Stop(Exception):
            pass

        def check_cancel() -> None:
            if decoded:
                raise Stop

        def decode(start: int, end: int) -> list:
            decoded.append(start)
            return []

        with pytest.raises(Stop):
            parakeet.decode_chunks(
                5_000,
                decode,
                _concat,
                sample_rate=100,
                chunk_seconds=20,
                overlap_seconds=5,
                check_cancel=check_cancel,
            )
        assert decoded == [0]

    def test_merge_only_runs_between_non_empty_chunks(self) -> None:
        merges: list[tuple[int, int]] = []

        def merge(previous: list, new: list) -> list:
            merges.append((len(previous), len(new)))
            return previous + new

        outputs = iter([[], [tok(" a", 0.0, 0.1)], [tok(" b", 0.0, 0.1)]])
        parakeet.decode_chunks(
            5_000,
            lambda s, e: next(outputs),
            merge,
            sample_rate=100,
            chunk_seconds=20,
            overlap_seconds=5,
        )
        assert merges == [(1, 1)]

    def test_rejects_an_overlap_that_would_never_advance(self) -> None:
        with pytest.raises(ValueError):
            parakeet.decode_chunks(
                5_000, lambda s, e: [], _concat, sample_rate=100, chunk_seconds=5, overlap_seconds=5
            )


# --------------------------------------------------------------------------
# Model + language support
# --------------------------------------------------------------------------


class TestSupport:
    @pytest.mark.parametrize(
        "model",
        ["mlx-community/parakeet-tdt-0.6b-v3", "/models/Parakeet-TDT-0.6b-v2", "parakeet"],
    )
    def test_recognises_parakeet_models(self, model: str) -> None:
        assert parakeet.is_parakeet_model(model)

    @pytest.mark.parametrize(
        "model", ["mlx-community/whisper-large-v3-mlx", "mlx-community/whisper-medium-mlx", ""]
    )
    def test_whisper_models_are_not_parakeet(self, model: str) -> None:
        assert not parakeet.is_parakeet_model(model)

    def test_covers_english_and_the_european_languages(self) -> None:
        for code in ("en", "es", "fr", "de", "it", "pt", "nl", "ru", "uk", "pl", "sv"):
            assert code in parakeet.LANGUAGES

    def test_does_not_claim_languages_it_cannot_transcribe(self) -> None:
        for code in ("fa", "ar", "zh", "ja", "ko", "hi", "tr", "vi", "id"):
            assert code not in parakeet.LANGUAGES


# --------------------------------------------------------------------------
# Contract with the installed library
# --------------------------------------------------------------------------


def _importable(name: str) -> bool:
    try:
        __import__(name)
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _importable("parakeet_mlx"), reason="parakeet-mlx is not installed")
class TestAgainstRealParakeetMlx:
    """Guards the parts of parakeet-mlx that backend.parakeet reaches into."""

    def test_entry_points_exist(self) -> None:
        import parakeet_mlx
        from parakeet_mlx import alignment, audio

        assert callable(parakeet_mlx.from_pretrained)
        assert callable(audio.get_logmel)
        assert callable(alignment.merge_longest_contiguous)
        assert callable(alignment.merge_longest_common_subsequence)

    def test_tokens_carry_the_fields_we_read(self) -> None:
        from parakeet_mlx.alignment import AlignedToken

        token = AlignedToken(id=1, text=" hi", start=1.0, duration=0.5, confidence=0.9)
        assert (token.text, token.start, token.end, token.confidence) == (" hi", 1.0, 1.5, 0.9)


@pytest.mark.skipif(not _importable("parakeet_mlx"), reason="parakeet-mlx is not installed")
def test_a_model_loaded_on_one_thread_runs_on_another(monkeypatch: pytest.MonkeyPatch) -> None:
    """The server preloads on one thread and transcribes on a long-lived worker.

    parakeet-mlx's from_pretrained leaves weights lazy, and lazy MLX arrays are
    bound to the creating thread's streams, so without materialising them at
    load the worker fails with "There is no Stream(cpu, 1) in current thread".
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import mlx.core as mx
    import mlx.nn as nn
    import parakeet_mlx

    def lazy_model(_model: str) -> nn.Module:
        layer = nn.Linear(4, 4)
        layer.update({"weight": layer.weight.astype(mx.float16) * 1, "bias": layer.bias * 1})
        return layer

    monkeypatch.setattr(parakeet_mlx, "from_pretrained", lazy_model)
    monkeypatch.setattr(parakeet._ModelHolder, "model", None)
    monkeypatch.setattr(parakeet._ModelHolder, "path", None)

    with ThreadPoolExecutor(max_workers=1) as worker:
        worker.submit(lambda: None).result()  # the worker exists before the preload
        loader = threading.Thread(target=parakeet._load, args=("tiny",))
        loader.start()
        loader.join()

        def use() -> float:
            return float(parakeet._load("tiny")(mx.ones((1, 4), dtype=mx.float16)).sum())

        assert isinstance(worker.submit(use).result(), float)
