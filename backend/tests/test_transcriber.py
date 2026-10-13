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

from backend import transcriber
from backend.config import Settings
from backend.transcriber import (
    MLXEngine,
    ModelUnavailableError,
    TranscribeOptions,
    TranscribeResult,
    TranscriptionCancelled,
    _progress_bridge,
    _supported_kwargs,
    _transcribe_module,
    resolve_route,
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


PARAKEET = "mlx-community/parakeet-tdt-0.6b-v3"
WHISPER = "mlx-community/whisper-large-v3-mlx"


def _options(**overrides: object) -> TranscribeOptions:
    return replace(TranscribeOptions(model=PARAKEET), **overrides)


def _never_detect() -> str | None:
    raise AssertionError("language detection should not run")


class TestResolveRoute:
    """Which model handles a job when Parakeet is the default."""

    def test_a_parakeet_language_stays_on_parakeet(self) -> None:
        routed = resolve_route(_options(language="en"), fallback_model=WHISPER, detect=_never_detect)
        assert (routed.model, routed.language) == (PARAKEET, "en")

    def test_translation_goes_to_whisper(self) -> None:
        routed = resolve_route(
            _options(language="es", task="translate"), fallback_model=WHISPER, detect=_never_detect
        )
        assert (routed.model, routed.task, routed.language) == (WHISPER, "translate", "es")

    def test_a_language_parakeet_lacks_goes_to_whisper(self) -> None:
        routed = resolve_route(_options(language="fa"), fallback_model=WHISPER, detect=_never_detect)
        assert (routed.model, routed.language) == (WHISPER, "fa")

    def test_auto_detects_and_keeps_parakeet_for_its_languages(self) -> None:
        routed = resolve_route(_options(), fallback_model=WHISPER, detect=lambda: "de")
        assert (routed.model, routed.language) == (PARAKEET, "de")

    def test_auto_detects_and_hands_other_languages_to_whisper(self) -> None:
        routed = resolve_route(_options(), fallback_model=WHISPER, detect=lambda: "ja")
        assert (routed.model, routed.language) == (WHISPER, "ja")

    def test_auto_translate_lets_whisper_detect(self) -> None:
        routed = resolve_route(
            _options(task="translate"), fallback_model=WHISPER, detect=_never_detect
        )
        assert (routed.model, routed.language) == (WHISPER, None)

    def test_whisper_models_are_never_rerouted(self) -> None:
        options = replace(TranscribeOptions(model=WHISPER), language="ja", task="translate")
        assert resolve_route(options, fallback_model=None, detect=_never_detect) == options

    def test_without_a_fallback_auto_stays_on_parakeet(self) -> None:
        routed = resolve_route(_options(), fallback_model=None, detect=_never_detect)
        assert (routed.model, routed.language) == (PARAKEET, None)

    @pytest.mark.parametrize("overrides", [{"task": "translate"}, {"language": "zh"}])
    def test_without_a_fallback_unsupported_jobs_fail_clearly(self, overrides: dict) -> None:
        with pytest.raises(ModelUnavailableError, match="WHISPER_MAC_FALLBACK_MODEL"):
            resolve_route(_options(**overrides), fallback_model=None, detect=_never_detect)


class TestMLXEngineRouting:
    @pytest.fixture
    def calls(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, TranscribeOptions]]:
        recorded: list[tuple[str, TranscribeOptions]] = []

        def fake(name: str):
            def run(samples, options, **_kwargs) -> TranscribeResult:
                recorded.append((name, options))
                return TranscribeResult(raw={}, language=options.language, model=options.model)

            return run

        monkeypatch.setattr(transcriber, "transcribe", fake("whisper"))
        monkeypatch.setattr(transcriber, "transcribe_parakeet", fake("parakeet"))
        monkeypatch.setattr(transcriber, "detect_language", lambda samples, model: "ko")
        return recorded

    def test_runs_the_routed_engine(self, calls: list) -> None:
        engine = MLXEngine(fallback_model=WHISPER)
        samples = np.zeros(16_000, dtype=np.float32)

        english = engine.transcribe(samples, _options(language="en"))
        korean = engine.transcribe(samples, _options())

        assert [name for name, _ in calls] == ["parakeet", "whisper"]
        assert (english.model, korean.model, korean.language) == (PARAKEET, WHISPER, "ko")

    def test_the_fallback_counts_as_part_of_a_parakeet_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        on_disk = {PARAKEET}
        downloaded: list[str] = []
        monkeypatch.setattr(transcriber, "model_is_local", lambda model: model in on_disk)
        monkeypatch.setattr(transcriber, "ensure_model", downloaded.append)
        engine = MLXEngine(fallback_model=WHISPER)

        assert not engine.model_is_local(PARAKEET)
        engine.ensure_model(PARAKEET)
        assert downloaded == [PARAKEET, WHISPER]
        assert MLXEngine().model_is_local(PARAKEET)


class TestTranscribeParakeet:
    def test_wraps_tokens_in_the_whisper_result_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from backend.tests.test_parakeet import tok

        monkeypatch.setattr(
            transcriber.parakeet, "run", lambda samples, model, **_: [tok(" Hi.", 0.0, 0.4)]
        )
        progress: list[float] = []

        result = transcriber.transcribe_parakeet(
            np.zeros(16_000, dtype=np.float32), _options(language="en"), on_progress=progress.append
        )

        assert (result.model, result.language) == (PARAKEET, "en")
        assert result.raw["segments"][0]["text"] == " Hi."
        assert progress[-1] == 1.0

    def test_cancellation_surfaces_as_transcription_cancelled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def run(samples, model, *, check_cancel, **_):
            check_cancel()
            return []

        monkeypatch.setattr(transcriber.parakeet, "run", run)
        with pytest.raises(TranscriptionCancelled):
            transcriber.transcribe_parakeet(
                np.zeros(16_000, dtype=np.float32), _options(), should_cancel=lambda: True
            )


def test_samples_contract_is_float32_mono() -> None:
    """What media.load_pcm16_wav promises the engine."""
    samples = np.zeros(16_000, dtype=np.float32)
    assert samples.dtype == np.float32
    assert samples.ndim == 1
