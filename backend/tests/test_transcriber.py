"""Engine wiring: the progress bridge, kwarg filtering, option mapping.

The bridge is the fragile part of this project — it depends on an internal
detail of mlx-whisper (that it drives a `tqdm` bar once per decode window). If it
breaks, transcription still *works*, so nothing fails loudly: the UI just stops
showing real progress and Cancel stops working. Hence these tests.
"""

from __future__ import annotations

import types
from dataclasses import replace

import numpy as np
import pytest

from backend.config import Settings
from backend.transcriber import (
    TranscribeOptions,
    TranscriptionCancelled,
    _progress_bridge,
    _supported_kwargs,
    _transcribe_module,
)

pytestmark = pytest.mark.unit


def _importable(name: str) -> bool:
    try:
        __import__(name)
    except Exception:
        return False
    return True


requires_mlx = pytest.mark.skipif(
    not _importable("mlx_whisper"), reason="mlx-whisper is not installed"
)


def _module_with_tqdm_module() -> types.ModuleType:
    """Fake target shaped like mlx-whisper: ``import tqdm`` then ``tqdm.tqdm()``."""
    module = types.ModuleType("fake_whisper")
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = object  # placeholder we expect to be swapped out
    module.tqdm = tqdm_module
    return module


def _module_with_tqdm_class() -> types.ModuleType:
    """Fake target shaped like ``from tqdm import tqdm``."""
    module = types.ModuleType("fake_whisper_flat")
    module.tqdm = object
    return module


class TestProgressBridge:
    def test_forwards_fractions_for_module_style_import(self) -> None:
        module = _module_with_tqdm_module()
        seen: list[float] = []

        with _progress_bridge(
            module, on_progress=seen.append, should_cancel=None, on_start=None
        ):
            with module.tqdm.tqdm(total=200, unit="frames", disable=False) as bar:
                bar.update(50)
                bar.update(50)
                bar.update(100)

        assert seen == [0.25, 0.5, 1.0]

    def test_forwards_fractions_for_flat_import(self) -> None:
        module = _module_with_tqdm_class()
        seen: list[float] = []

        with _progress_bridge(
            module, on_progress=seen.append, should_cancel=None, on_start=None
        ):
            with module.tqdm(total=4, disable=False) as bar:
                bar.update(1)

        assert seen == [0.25]

    def test_fires_on_start_when_the_bar_is_created(self) -> None:
        module = _module_with_tqdm_module()
        started: list[bool] = []

        with _progress_bridge(
            module, on_progress=None, should_cancel=None, on_start=lambda: started.append(True)
        ):
            module.tqdm.tqdm(total=1, disable=False)

        assert started == [True]

    def test_cancellation_raises_from_inside_update(self) -> None:
        module = _module_with_tqdm_module()

        with _progress_bridge(
            module, on_progress=None, should_cancel=lambda: True, on_start=None
        ):
            bar = module.tqdm.tqdm(total=10, disable=False)
            with pytest.raises(TranscriptionCancelled):
                bar.update(1)

    def test_restores_the_original_attribute(self) -> None:
        module = _module_with_tqdm_module()
        original = module.tqdm

        with _progress_bridge(module, on_progress=None, should_cancel=None, on_start=None):
            assert module.tqdm is not original
        assert module.tqdm is original

    def test_restores_the_original_even_after_an_error(self) -> None:
        module = _module_with_tqdm_module()
        original = module.tqdm

        with pytest.raises(RuntimeError), _progress_bridge(
            module, on_progress=None, should_cancel=None, on_start=None
        ):
            raise RuntimeError("boom")
        assert module.tqdm is original

    def test_missing_tqdm_degrades_instead_of_failing(self) -> None:
        module = types.ModuleType("no_tqdm_here")
        with _progress_bridge(module, on_progress=None, should_cancel=None, on_start=None):
            pass  # must not raise
        assert not hasattr(module, "tqdm")

    def test_progress_is_capped_to_one(self) -> None:
        module = _module_with_tqdm_class()
        seen: list[float] = []
        with _progress_bridge(
            module, on_progress=seen.append, should_cancel=None, on_start=None
        ):
            bar = module.tqdm(total=10, disable=False)
            bar.update(50)
        assert seen == [1.0]


class TestSupportedKwargs:
    def test_drops_unknown_keys(self) -> None:
        def target(*, alpha: int = 1, beta: int = 2) -> None: ...

        filtered = _supported_kwargs(target, {"alpha": 1, "gamma": 3})
        assert filtered == {"alpha": 1}

    def test_keeps_decode_option_passthroughs_when_kwargs_are_accepted(self) -> None:
        def target(*, alpha: int = 1, **decode_options: object) -> None: ...

        filtered = _supported_kwargs(
            target, {"alpha": 1, "language": "en", "task": "transcribe", "fp16": True}
        )
        assert filtered == {"alpha": 1, "language": "en", "task": "transcribe", "fp16": True}

    def test_drops_passthroughs_when_kwargs_are_not_accepted(self) -> None:
        def target(*, alpha: int = 1) -> None: ...

        assert _supported_kwargs(target, {"alpha": 1, "language": "en"}) == {"alpha": 1}


class TestOptions:
    def test_auto_language_becomes_none(self) -> None:
        settings = Settings()
        for value in ("auto", "AUTO", "", "  ", "detect"):
            assert TranscribeOptions.from_settings(settings, language=value).language is None

    def test_language_is_normalised(self) -> None:
        assert TranscribeOptions.from_settings(Settings(), language="EN").language == "en"

    def test_unknown_task_falls_back_to_transcribe(self) -> None:
        assert TranscribeOptions.from_settings(Settings(), task="summarise").task == "transcribe"
        assert TranscribeOptions.from_settings(Settings(), task="translate").task == "translate"

    def test_verbose_false_is_what_enables_the_progress_bar(self) -> None:
        # mlx-whisper uses `disable=verbose is not False`, so False is the only
        # value that gives us a bar without printing every segment.
        assert TranscribeOptions.from_settings(Settings()).as_kwargs()["verbose"] is False

    def test_temperature_is_scalar_unless_fallback_is_enabled(self) -> None:
        settings = Settings()
        assert TranscribeOptions.from_settings(settings).temperature == 0.0

        laddered = replace(settings, temperature_fallback=True)
        assert TranscribeOptions.from_settings(laddered).temperature == (
            0.0, 0.2, 0.4, 0.6, 0.8, 1.0
        )

    def test_hallucination_threshold_requires_word_timestamps(self) -> None:
        off = replace(Settings(), word_timestamps=False)
        assert TranscribeOptions.from_settings(off).hallucination_silence_threshold is None
        assert TranscribeOptions.from_settings(Settings()).hallucination_silence_threshold == 2.0

    def test_fp16_is_on_so_metal_runs_half_precision(self) -> None:
        assert TranscribeOptions.from_settings(Settings()).as_kwargs()["fp16"] is True


@requires_mlx
class TestAgainstRealMlxWhisper:
    """Guards the assumptions the bridge makes about the installed library."""

    def test_transcribe_module_resolves_to_a_module_not_the_function(self) -> None:
        module = _transcribe_module()
        assert isinstance(module, types.ModuleType)
        assert module.__name__ == "mlx_whisper.transcribe"

    def test_the_module_still_exposes_tqdm(self) -> None:
        # If this ever fails, mlx-whisper changed how it reports progress and
        # the bridge in transcriber.py needs updating.
        assert hasattr(_transcribe_module(), "tqdm")

    def test_transcribe_accepts_every_option_we_send(self) -> None:
        import mlx_whisper

        candidate = TranscribeOptions.from_settings(Settings()).as_kwargs()
        filtered = _supported_kwargs(mlx_whisper.transcribe, candidate)
        assert set(filtered) == set(candidate), (
            f"mlx-whisper rejected: {set(candidate) - set(filtered)}"
        )

    def test_model_holder_is_where_we_think_it_is(self) -> None:
        # backend.transcriber.model_is_loaded reads this to report "weights hot".
        assert hasattr(_transcribe_module(), "ModelHolder")


def test_samples_contract_is_float32_mono() -> None:
    """What media.load_pcm16_wav promises the engine."""
    samples = np.zeros(16_000, dtype=np.float32)
    assert samples.dtype == np.float32
    assert samples.ndim == 1
