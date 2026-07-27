"""MLX Whisper engine wrapper: inference, live progress, cancellation.

Everything Apple-Silicon-specific lives here. mlx-whisper runs the encoder and
decoder as Metal kernels in float16 — there is no PyTorch, ONNX or CPU fallback
path in this project, by design.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

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
            f"Could not download the Whisper model '{model}'.\n"
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
    """The real engine: mlx-whisper on Metal."""

    def model_is_local(self, model: str) -> bool:
        return model_is_local(model)

    def model_is_loaded(self, model: str) -> bool:
        return model_is_loaded(model)

    def ensure_model(self, model: str) -> str:
        return ensure_model(model)

    def transcribe(
        self,
        samples: np.ndarray,
        options: TranscribeOptions,
        *,
        on_progress: ProgressCallback | None = None,
        should_cancel: CancelCheck | None = None,
        on_inference_start: Callable[[], None] | None = None,
    ) -> TranscribeResult:
        return transcribe(
            samples,
            options,
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
