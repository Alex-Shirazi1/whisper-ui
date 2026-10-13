"""MLX speech engines: inference, live progress, cancellation, routing.

Everything Apple-Silicon-specific lives here. Both engines run as Metal kernels
through MLX — there is no PyTorch, ONNX or CPU fallback path in this project, by
design.

The default model is NVIDIA Parakeet (:mod:`backend.parakeet`): faster and more
accurate on English than whisper-large-v3, but limited to 25 European languages
and unable to translate. :class:`MLXEngine` routes each job: Parakeet when it
can do the job, otherwise the Whisper fallback model — for translation, for a
language Parakeet lacks, and to identify the language when none was chosen.

Two things mlx-whisper does not offer out of the box are progress reporting and
cancellation. Both are obtained by swapping the ``tqdm`` module it uses for a
shim (:class:`_ProgressBar`): mlx-whisper calls ``pbar.update(...)`` once per
30-second decode window, so the shim sees real decode position and can raise
:class:`TranscriptionCancelled` from inside the loop to stop the work.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import threading
import types
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from . import parakeet
from .config import Settings
from .logging_setup import LoggerWriter, get_logger

log = get_logger(__name__)

ProgressCallback = Callable[[float], None]
CancelCheck = Callable[[], bool]
StageCallback = Callable[[str], None]

#: MLX keeps one model resident; serialise inference so two jobs never fight
#: over the GPU or thrash the weight cache.
_INFERENCE_LOCK = threading.Lock()


class TranscriptionCancelled(RuntimeError):
    pass


class EngineUnavailableError(RuntimeError):
    """mlx / mlx-whisper is missing or cannot run on this machine."""


class ModelUnavailableError(RuntimeError):
    """Weights are neither cached locally nor downloadable."""


# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TranscribeOptions:
    model: str
    #: ``None`` = auto-detect from the first 30 seconds.
    language: str | None = None
    task: str = "transcribe"
    initial_prompt: str | None = None
    temperature: float | tuple[float, ...] = 0.0
    word_timestamps: bool = True
    condition_on_previous_text: bool = True
    hallucination_silence_threshold: float | None = 2.0
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    #: float16 Metal kernels. Turning this off doubles memory for no gain.
    fp16: bool = True

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        model: str | None = None,
        language: str | None = None,
        task: str = "transcribe",
        initial_prompt: str | None = None,
    ) -> TranscribeOptions:
        return cls(
            model=model or settings.model,
            language=_normalise_language(language),
            task=task if task in {"transcribe", "translate"} else "transcribe",
            initial_prompt=initial_prompt or None,
            temperature=settings.temperatures,
            word_timestamps=settings.word_timestamps,
            condition_on_previous_text=settings.condition_on_previous_text,
            hallucination_silence_threshold=(
                settings.hallucination_silence_threshold if settings.word_timestamps else None
            ),
            compression_ratio_threshold=settings.compression_ratio_threshold,
            logprob_threshold=settings.logprob_threshold,
            no_speech_threshold=settings.no_speech_threshold,
        )

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "path_or_hf_repo": self.model,
            # False is the one value that enables mlx-whisper's progress bar
            # without also printing every decoded segment to stdout.
            "verbose": False,
            "language": self.language,
            "task": self.task,
            "initial_prompt": self.initial_prompt,
            "temperature": self.temperature,
            "word_timestamps": self.word_timestamps,
            "condition_on_previous_text": self.condition_on_previous_text,
            "hallucination_silence_threshold": self.hallucination_silence_threshold,
            "compression_ratio_threshold": self.compression_ratio_threshold,
            "logprob_threshold": self.logprob_threshold,
            "no_speech_threshold": self.no_speech_threshold,
            "fp16": self.fp16,
        }


def _normalise_language(language: str | None) -> str | None:
    if language is None:
        return None
    cleaned = language.strip().lower()
    return None if cleaned in {"", "auto", "detect", "automatic"} else cleaned


# --------------------------------------------------------------------------
# Progress bridge
# --------------------------------------------------------------------------


class _ProgressBar:
    """Minimal ``tqdm`` stand-in wired to our callbacks."""

    def __init__(
        self,
        *args: Any,
        total: float | None = None,
        on_progress: ProgressCallback | None = None,
        should_cancel: CancelCheck | None = None,
        on_start: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        self.total = total
        self.n = 0.0
        self._on_progress = on_progress
        self._should_cancel = should_cancel
        if on_start is not None:
            on_start()

    # tqdm API surface used by mlx-whisper (plus a little slack for versions).
    def __enter__(self) -> _ProgressBar:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[Any]:  # pragma: no cover - not used upstream
        return iter(())

    def update(self, advance: float = 1) -> None:
        self.n += float(advance or 0)
        self._emit()
        if self._should_cancel is not None and self._should_cancel():
            raise TranscriptionCancelled("Transcription cancelled.")

    def close(self) -> None:
        return None

    def refresh(self) -> None:  # pragma: no cover - cosmetic in tqdm
        self._emit()

    def set_description(self, *_args: Any, **_kwargs: Any) -> None:  # pragma: no cover
        return None

    set_postfix = set_description
    write = set_description

    def _emit(self) -> None:
        if self._on_progress is None or not self.total:
            return
        self._on_progress(max(0.0, min(1.0, self.n / float(self.total))))


def _transcribe_module() -> types.ModuleType:
    """Return the ``mlx_whisper.transcribe`` *module*.

    Deliberately not ``from mlx_whisper import transcribe``: the package's
    ``__init__`` rebinds that name to the *function*, so the obvious import (and
    ``import mlx_whisper.transcribe as m``, which prefers the attribute) hands
    back a callable with no module globals to patch — which silently disables
    progress and cancellation. Going through ``importlib`` always resolves
    ``sys.modules``.
    """
    return importlib.import_module("mlx_whisper.transcribe")


@contextlib.contextmanager
def _progress_bridge(
    module: types.ModuleType,
    *,
    on_progress: ProgressCallback | None,
    should_cancel: CancelCheck | None,
    on_start: Callable[[], None] | None,
) -> Iterator[None]:
    """Temporarily replace ``module.tqdm`` with a callback-driven shim.

    mlx-whisper does ``import tqdm`` and calls ``tqdm.tqdm(...)``, but older and
    newer releases have used ``from tqdm import tqdm``. Both shapes are handled,
    and if neither is present the transcription simply runs without live
    progress rather than failing.
    """
    original = getattr(module, "tqdm", None)
    if original is None:
        # Loud, because losing this silently downgrades the UI to a time estimate.
        log.warning(
            "%s has no 'tqdm' attribute; live progress and cancellation are "
            "unavailable and progress will be estimated from elapsed time.",
            getattr(module, "__name__", module),
        )
        yield
        return

    def factory(*args: Any, **kwargs: Any) -> _ProgressBar:
        kwargs.pop("on_progress", None)
        kwargs.pop("should_cancel", None)
        return _ProgressBar(
            *args,
            on_progress=on_progress,
            should_cancel=should_cancel,
            on_start=on_start,
            **kwargs,
        )

    replacement: Any
    if isinstance(original, types.ModuleType):
        replacement = types.SimpleNamespace(tqdm=factory)
    else:
        replacement = factory

    module.tqdm = replacement
    try:
        yield
    finally:
        module.tqdm = original


# --------------------------------------------------------------------------
# Engine introspection
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineStatus:
    available: bool
    mlx_version: str | None = None
    mlx_whisper_version: str | None = None
    device: str | None = None
    error: str | None = None


def engine_status() -> EngineStatus:
    """Import MLX and report what we found — never raises."""
    try:
        import mlx.core as mx  # noqa: PLC0415 - deliberately lazy
        import mlx_whisper  # noqa: PLC0415
        import parakeet_mlx  # noqa: F401, PLC0415
    except Exception as exc:  # pragma: no cover - depends on host
        return EngineStatus(available=False, error=f"{type(exc).__name__}: {exc}")

    try:
        device = str(mx.default_device())
    except Exception as exc:  # pragma: no cover
        return EngineStatus(available=False, error=f"MLX could not open a device: {exc}")

    return EngineStatus(
        available=True,
        mlx_version=getattr(mx, "__version__", None),
        mlx_whisper_version=getattr(mlx_whisper, "__version__", None),
        device=device,
    )


def require_engine() -> None:
    status = engine_status()
    if not status.available:
        raise EngineUnavailableError(
            "The MLX Whisper engine is not available.\n"
            f"Details: {status.error}\n"
            "Run ./scripts/install.sh to repair the Python environment."
        )


def model_is_local(model: str) -> bool:
    """True when weights are already on disk (no network needed)."""
    candidate = Path(model).expanduser()
    if candidate.is_dir() and (candidate / "config.json").exists():
        return True
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        snapshot_download(repo_id=model, local_files_only=True)
        return True
    except Exception:
        return False


def model_is_loaded(model: str) -> bool:
    """True when the weights are already resident in memory."""
    try:
        from mlx_whisper.transcribe import ModelHolder  # noqa: PLC0415
    except Exception:  # pragma: no cover - version drift
        return False
    return getattr(ModelHolder, "model", None) is not None and (
        getattr(ModelHolder, "model_path", None) == model
    )


def ensure_model(model: str) -> str:
    """Make sure weights exist locally, downloading once if needed."""
    candidate = Path(model).expanduser()
    if candidate.is_dir() and (candidate / "config.json").exists():
        return str(candidate)
    if model_is_local(model):
        return model

    log.info("Downloading model weights: %s", model)
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        snapshot_download(repo_id=model)
    except Exception as exc:
        raise ModelUnavailableError(
            f"Could not download the speech model '{model}'.\n"
            f"Details: {type(exc).__name__}: {exc}\n"
            "Check your internet connection, then retry. Once downloaded the "
            "model is cached in backend/models and never fetched again."
        ) from exc
    return model


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TranscribeResult:
    raw: Mapping[str, Any]
    language: str | None
    model: str
    options: TranscribeOptions = field(repr=False, default=None)  # type: ignore[assignment]


def transcribe(
    samples: np.ndarray,
    options: TranscribeOptions,
    *,
    on_progress: ProgressCallback | None = None,
    should_cancel: CancelCheck | None = None,
    on_inference_start: Callable[[], None] | None = None,
) -> TranscribeResult:
    """Run MLX Whisper over ``samples`` (float32 mono @ 16 kHz).

    ``on_inference_start`` fires once the weights are loaded and the first decode
    window begins, which is what lets the UI move from "Loading model" to
    "Transcribing" at the right moment.
    """
    require_engine()

    import mlx_whisper  # noqa: PLC0415

    kwargs = _supported_kwargs(mlx_whisper.transcribe, options.as_kwargs())
    transcribe_module = _transcribe_module()

    with _INFERENCE_LOCK:
        if should_cancel is not None and should_cancel():
            raise TranscriptionCancelled("Transcription cancelled before it started.")

        bridge = _progress_bridge(
            transcribe_module,
            on_progress=on_progress,
            should_cancel=should_cancel,
            on_start=on_inference_start,
        )
        # mlx-whisper prints the detected language to stdout; keep it in the log.
        with bridge, contextlib.redirect_stdout(LoggerWriter(log)):
            raw = mlx_whisper.transcribe(samples, **kwargs)

    if on_progress is not None:
        on_progress(1.0)

    language = raw.get("language") if isinstance(raw, Mapping) else None
    return TranscribeResult(
        raw=raw,
        language=str(language) if language else None,
        model=options.model,
        options=options,
    )


def detect_language(samples: np.ndarray, model: str) -> str | None:
    """Whisper's language identification over the first 30 seconds.

    The same check ``mlx_whisper.transcribe`` runs when no language is given,
    without the decode that follows it.
    """
    require_engine()

    import mlx.core as mx  # noqa: PLC0415
    from mlx_whisper.audio import (  # noqa: PLC0415
        N_FRAMES,
        N_SAMPLES,
        log_mel_spectrogram,
        pad_or_trim,
    )
    from mlx_whisper.transcribe import ModelHolder  # noqa: PLC0415

    with _INFERENCE_LOCK:
        whisper = ModelHolder.get_model(model, mx.float16)
        mel = log_mel_spectrogram(samples[:N_SAMPLES], n_mels=whisper.dims.n_mels, padding=N_SAMPLES)
        _, probs = whisper.detect_language(pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16))

    if not probs:
        return None
    language = max(probs, key=probs.get)
    log.info("Detected language: %s (p=%.2f)", language, probs[language])
    return language


def transcribe_parakeet(
    samples: np.ndarray,
    options: TranscribeOptions,
    *,
    on_progress: ProgressCallback | None = None,
    should_cancel: CancelCheck | None = None,
    on_inference_start: Callable[[], None] | None = None,
) -> TranscribeResult:
    """Run Parakeet over ``samples``; same contract as :func:`transcribe`.

    ``options.language`` is recorded on the result, not passed to the model:
    Parakeet has no language control.
    """

    def check_cancel() -> None:
        if should_cancel is not None and should_cancel():
            raise TranscriptionCancelled("Transcription cancelled.")

    with _INFERENCE_LOCK:
        check_cancel()
        try:
            tokens = parakeet.run(
                samples,
                options.model,
                on_progress=on_progress,
                check_cancel=check_cancel,
                on_start=on_inference_start,
            )
        except ImportError as exc:  # pragma: no cover - depends on host
            raise EngineUnavailableError(
                f"The Parakeet engine is not installed ({exc}).\n"
                "Run ./scripts/install.sh to repair the Python environment."
            ) from exc

    if on_progress is not None:
        on_progress(1.0)

    return TranscribeResult(
        raw=parakeet.to_raw(tokens, language=options.language),
        language=options.language,
        model=options.model,
        options=options,
    )


def resolve_route(
    options: TranscribeOptions,
    *,
    fallback_model: str | None,
    detect: Callable[[], str | None],
) -> TranscribeOptions:
    """Decide which model actually runs a job, and in which language.

    Whisper models run as asked. For a Parakeet model the job moves to
    ``fallback_model`` when Parakeet cannot do it: translation, or a language
    outside :data:`parakeet.LANGUAGES` — chosen by the user, or found by
    ``detect`` when the language is left on auto. Without a fallback, auto
    stays on Parakeet and its 25 languages.
    """
    if not parakeet.is_parakeet_model(options.model):
        return options

    language = options.language
    if options.task == "transcribe" and language is not None and language in parakeet.LANGUAGES:
        return options

    if not fallback_model:
        if options.task == "translate":
            raise ModelUnavailableError(
                "Translation needs a Whisper model, and WHISPER_MAC_FALLBACK_MODEL is not set."
            )
        if language is not None:
            raise ModelUnavailableError(
                f"Parakeet cannot transcribe '{language}', and WHISPER_MAC_FALLBACK_MODEL "
                "is not set. Set it to a Whisper model to handle other languages."
            )
        return options

    if options.task == "translate" or language is not None:
        return replace(options, model=fallback_model)

    detected = detect()
    if detected is not None and detected not in parakeet.LANGUAGES:
        return replace(options, model=fallback_model, language=detected)
    return replace(options, language=detected)


# --------------------------------------------------------------------------
# Engine seam
# --------------------------------------------------------------------------


class Engine(Protocol):
    """Everything the job pipeline needs from a speech-to-text backend.

    The pipeline depends on this protocol, not on MLX directly, which is what
    lets the test suite run without weights — and what will let a diarizing or
    remote engine drop in later without touching :mod:`backend.jobs`.
    """

    def model_is_local(self, model: str) -> bool: ...

    def model_is_loaded(self, model: str) -> bool: ...

    def ensure_model(self, model: str) -> str: ...

    def transcribe(
        self,
        samples: np.ndarray,
        options: TranscribeOptions,
        *,
        on_progress: ProgressCallback | None = None,
        should_cancel: CancelCheck | None = None,
        on_inference_start: Callable[[], None] | None = None,
    ) -> TranscribeResult: ...


class MLXEngine:
    """The real engine: Parakeet or mlx-whisper on Metal, routed per job.

    ``fallback_model`` is the Whisper model behind a Parakeet default (see
    :func:`resolve_route`). It is treated as part of the model: it is
    downloaded alongside it and must be on disk for the model to count as local.
    """

    def __init__(self, fallback_model: str | None = None) -> None:
        self._fallback_model = fallback_model or None

    def _required(self, model: str) -> list[str]:
        if parakeet.is_parakeet_model(model) and self._fallback_model:
            return [model, self._fallback_model]
        return [model]

    def model_is_local(self, model: str) -> bool:
        return all(model_is_local(required) for required in self._required(model))

    def model_is_loaded(self, model: str) -> bool:
        if parakeet.is_parakeet_model(model):
            return parakeet.model_is_loaded(model)
        return model_is_loaded(model)

    def ensure_model(self, model: str) -> str:
        for required in self._required(model):
            ensure_model(required)
        return model

    def transcribe(
        self,
        samples: np.ndarray,
        options: TranscribeOptions,
        *,
        on_progress: ProgressCallback | None = None,
        should_cancel: CancelCheck | None = None,
        on_inference_start: Callable[[], None] | None = None,
    ) -> TranscribeResult:
        fallback = self._fallback_model
        routed = resolve_route(
            options,
            fallback_model=fallback,
            detect=lambda: detect_language(samples, fallback) if fallback else None,
        )
        if routed.model != options.model:
            log.info(
                "Routing to %s (task=%s, language=%s)", routed.model, routed.task, routed.language
            )
        run = transcribe_parakeet if parakeet.is_parakeet_model(routed.model) else transcribe
        return run(
            samples,
            routed,
            on_progress=on_progress,
            should_cancel=should_cancel,
            on_inference_start=on_inference_start,
        )

    def status(self) -> EngineStatus:
        return engine_status()


def _supported_kwargs(func: Callable[..., Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Drop kwargs the installed mlx-whisper does not accept.

    ``transcribe`` forwards unknown keys into ``DecodingOptions``, which raises
    on anything it does not recognise — so filtering here keeps us compatible
    across mlx-whisper releases instead of pinning to one.
    """
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return dict(candidate)

    accepts_extras = any(
        param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
    )
    known = set(signature.parameters)
    # Keys that reach DecodingOptions through **decode_options.
    passthrough = {"language", "task", "fp16"}

    filtered: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in candidate.items():
        if key in known or (accepts_extras and key in passthrough):
            filtered[key] = value
        else:
            dropped.append(key)

    if dropped:
        log.debug("mlx-whisper does not accept %s; ignoring.", ", ".join(sorted(dropped)))
    return filtered
