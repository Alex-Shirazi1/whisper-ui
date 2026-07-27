"""Transcription job queue: one pipeline, run off the event loop.

A job walks a fixed set of stages (inspect -> extract -> load -> transcribe ->
write). Each stage owns a slice of the 0..1 progress range so the UI gets a
single monotonic number it can render as one bar, plus a human-readable message
for the status line.

State is held as a frozen :class:`JobState`; mutating a job means building a
replacement snapshot under a lock, so a reader can never observe a half-updated
record.
"""

from __future__ import annotations

import shutil
import threading
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

import numpy as np

from . import media
from .config import Settings, is_video
from .formats import Transcript, build_transcript, normalise_formats, write_outputs
from .logging_setup import get_logger
from .transcriber import (
    Engine,
    EngineUnavailableError,
    MLXEngine,
    ModelUnavailableError,
    TranscribeOptions,
    TranscribeResult,
    TranscriptionCancelled,
)

log = get_logger(__name__)


class JobStage(StrEnum):
    QUEUED = "queued"
    INSPECTING = "inspecting"
    EXTRACTING_AUDIO = "extracting_audio"
    DOWNLOADING_MODEL = "downloading_model"
    LOADING_MODEL = "loading_model"
    TRANSCRIBING = "transcribing"
    WRITING_OUTPUT = "writing_output"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {JobStage.COMPLETED, JobStage.FAILED, JobStage.CANCELLED}


#: Copy shown in the app's status line. Phrased for a person, not a log.
STAGE_MESSAGES: Mapping[JobStage, str] = {
    JobStage.QUEUED: "Waiting in queue…",
    JobStage.INSPECTING: "Reading file…",
    JobStage.EXTRACTING_AUDIO: "Extracting audio…",
    JobStage.DOWNLOADING_MODEL: "Downloading Whisper model (one time only)…",
    JobStage.LOADING_MODEL: "Loading Whisper model…",
    JobStage.TRANSCRIBING: "Transcribing…",
    JobStage.WRITING_OUTPUT: "Writing transcript…",
    JobStage.COMPLETED: "Finished",
    JobStage.FAILED: "Failed",
    JobStage.CANCELLED: "Cancelled",
}

#: Fraction of the overall bar each stage occupies.
_STAGE_SPANS: Mapping[JobStage, tuple[float, float]] = {
    JobStage.QUEUED: (0.0, 0.0),
    JobStage.INSPECTING: (0.0, 0.02),
    JobStage.EXTRACTING_AUDIO: (0.02, 0.10),
    JobStage.DOWNLOADING_MODEL: (0.10, 0.13),
    JobStage.LOADING_MODEL: (0.13, 0.17),
    JobStage.TRANSCRIBING: (0.17, 0.97),
    JobStage.WRITING_OUTPUT: (0.97, 1.0),
}

_INDETERMINATE_STAGES = {JobStage.DOWNLOADING_MODEL, JobStage.LOADING_MODEL}


# --------------------------------------------------------------------------
# Requests & state
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class JobRequest:
    source_path: Path
    output_dir: Path
    formats: tuple[str, ...] = ("txt",)
    model: str | None = None
    language: str | None = None
    task: str = "transcribe"
    initial_prompt: str | None = None
    overwrite: bool = False
    #: True when ``source_path`` is a temp copy of an upload that we own.
    cleanup_source: bool = False
    display_name: str | None = None

    @property
    def name(self) -> str:
        return self.display_name or self.source_path.name


@dataclass(frozen=True)
class JobState:
    id: str
    name: str
    source_path: Path
    output_dir: Path
    formats: tuple[str, ...]
    model: str
    task: str
    stage: JobStage = JobStage.QUEUED
    progress: float = 0.0
    message: str = STAGE_MESSAGES[JobStage.QUEUED]
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    media_duration: float | None = None
    detected_language: str | None = None
    is_video: bool = False
    outputs: Mapping[str, Path] = field(default_factory=dict)
    transcript: Transcript | None = field(default=None, repr=False)
    #: Real decode progress reported by the engine (0..1), before blending.
    engine_progress: float = 0.0

    @property
    def indeterminate(self) -> bool:
        return self.stage in _INDETERMINATE_STAGES

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else time.time()
        return max(0.0, end - self.started_at)

    @property
    def eta_seconds(self) -> float | None:
        if self.stage.is_terminal or self.progress <= 0.02 or self.started_at is None:
            return None
        remaining = (self.elapsed / self.progress) - self.elapsed
        return round(remaining, 1) if remaining > 0 else None

    @property
    def speed_factor(self) -> float | None:
        """How many times faster than realtime the whole job ran."""
        if not self.media_duration or not self.finished_at or self.elapsed <= 0:
            return None
        return round(self.media_duration / self.elapsed, 2)


# --------------------------------------------------------------------------
# Job
# --------------------------------------------------------------------------


class Job:
    """Mutable holder around an immutable :class:`JobState`."""

    def __init__(self, state: JobState, request: JobRequest) -> None:
        self._state = state
        self.request = request
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self.future: Future | None = None

    @property
    def state(self) -> JobState:
        with self._lock:
            return self._state

    def update(self, **changes: object) -> JobState:
        with self._lock:
            self._state = replace(self._state, **changes)  # type: ignore[arg-type]
            return self._state

    def set_stage(self, stage: JobStage, *, message: str | None = None) -> JobState:
        with self._lock:
            floor, _ = _STAGE_SPANS.get(stage, (self._state.progress, 1.0))
            self._state = replace(
                self._state,
                stage=stage,
                message=message or STAGE_MESSAGES[stage],
                progress=max(self._state.progress, floor),
            )
            return self._state

    def report(self, stage: JobStage, fraction: float) -> None:
        """Record stage-local progress, mapped into the global bar."""
        low, high = _STAGE_SPANS.get(stage, (0.0, 1.0))
        scaled = low + (high - low) * max(0.0, min(1.0, fraction))
        with self._lock:
            if self._state.stage is not stage:
                return
            self._state = replace(
                self._state,
                progress=max(self._state.progress, scaled),
                engine_progress=(
                    max(self._state.engine_progress, fraction)
                    if stage is JobStage.TRANSCRIBING
                    else self._state.engine_progress
                ),
            )

    def blend_estimate(self, realtime_factor: float) -> JobState:
        """Advance the bar between decode windows using an elapsed-time model.

        Whisper reports progress once per 30-second window, which on a long file
        means a bar that sits still for seconds at a time. Interpolating with an
        estimate keeps it moving without ever going backwards or overtaking a
        real report by more than the cap.
        """
        with self._lock:
            state = self._state
            if state.stage is not JobStage.TRANSCRIBING or not state.media_duration:
                return state
            expected = max(1.0, state.media_duration * max(realtime_factor, 0.005))
            estimate = min(state.elapsed / expected, 0.95)
            low, high = _STAGE_SPANS[JobStage.TRANSCRIBING]
            scaled = low + (high - low) * estimate
            if scaled > state.progress:
                self._state = replace(state, progress=scaled)
            return self._state

    def request_cancel(self) -> None:
        self._cancel.set()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------


class JobStore:
    """Owns the worker thread, the job registry and the pipeline itself."""

    def __init__(
        self,
        settings: Settings,
        *,
        engine: Engine | None = None,
        max_workers: int = 1,
    ) -> None:
        self._settings = settings
        self._engine: Engine = engine or MLXEngine()
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="whisper-job"
        )

    # ----------------------------------------------------------- lifecycle

    def shutdown(self, *, wait: bool = False) -> None:
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            if not job.state.stage.is_terminal:
                job.request_cancel()
        self._executor.shutdown(wait=wait, cancel_futures=True)

    # -------------------------------------------------------------- public

    def submit(self, request: JobRequest) -> JobState:
        formats = normalise_formats(request.formats)
        job_id = uuid.uuid4().hex[:12]
        state = JobState(
            id=job_id,
            name=request.name,
            source_path=request.source_path,
            output_dir=request.output_dir,
            formats=formats,
            model=request.model or self._settings.model,
            task=request.task,
            is_video=is_video(request.source_path),
        )
        job = Job(state, replace(request, formats=formats))

        with self._lock:
            self._jobs[job_id] = job
        self._prune()

        job.future = self._executor.submit(self._run, job)
        log.info("Queued job %s for %s", job_id, request.name)
        return job.state

    def get(self, job_id: str) -> JobState | None:
        job = self._job(job_id)
        if job is None:
            return None
        return job.blend_estimate(self._settings.estimated_realtime_factor)

    def list(self) -> tuple[JobState, ...]:
        self._prune()
        with self._lock:
            jobs = list(self._jobs.values())
        states = [job.blend_estimate(self._settings.estimated_realtime_factor) for job in jobs]
        return tuple(sorted(states, key=lambda s: s.created_at))

    def cancel(self, job_id: str) -> JobState | None:
        job = self._job(job_id)
        if job is None:
            return None
        state = job.state
        if state.stage.is_terminal:
            return state
        job.request_cancel()
        if state.stage is JobStage.QUEUED and job.future is not None and job.future.cancel():
            return self._finish_cancelled(job)
        return job.state

    def transcript(self, job_id: str) -> Transcript | None:
        job = self._job(job_id)
        return job.state.transcript if job else None

    # ------------------------------------------------------------ internal

    def _job(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _prune(self) -> None:
        cutoff = time.time() - self._settings.job_retention_seconds
        with self._lock:
            stale = [
                job_id
                for job_id, job in self._jobs.items()
                if job.state.stage.is_terminal
                and (job.state.finished_at or job.state.created_at) < cutoff
            ]
            for job_id in stale:
                self._jobs.pop(job_id, None)
        if stale:
            log.debug("Pruned %d finished job(s)", len(stale))

    def _finish_cancelled(self, job: Job) -> JobState:
        self._cleanup(job)
        return job.update(
            stage=JobStage.CANCELLED,
            message=STAGE_MESSAGES[JobStage.CANCELLED],
            finished_at=time.time(),
        )

    # ------------------------------------------------------------ pipeline

    def _run(self, job: Job) -> None:
        job.update(started_at=time.time())
        workspace = self._settings.temp_dir / f"job-{job.state.id}"

        try:
            if job.cancel_requested:
                self._finish_cancelled(job)
                return

            info = self._inspect(job)
            samples = self._prepare_audio(job, info, workspace)
            result = self._run_engine(job, samples)
            self._write(job, result, duration=info.duration or media.audio_duration(samples))

        except TranscriptionCancelled:
            log.info("Job %s cancelled", job.state.id)
            job.update(
                stage=JobStage.CANCELLED,
                message=STAGE_MESSAGES[JobStage.CANCELLED],
                finished_at=time.time(),
            )
        except media.OperationCancelled:
            log.info("Job %s cancelled during extraction", job.state.id)
            job.update(
                stage=JobStage.CANCELLED,
                message=STAGE_MESSAGES[JobStage.CANCELLED],
                finished_at=time.time(),
            )
        except (media.MediaError, EngineUnavailableError, ModelUnavailableError) as exc:
            self._fail(job, str(exc))
        except MemoryError:
            self._fail(
                job,
                "Ran out of memory. Close other apps, or switch to a smaller model "
                "such as mlx-community/whisper-medium-mlx in Settings.",
            )
        except Exception as exc:  # noqa: BLE001 - last line of defence
            log.exception("Job %s failed unexpectedly", job.state.id)
            self._fail(job, f"Unexpected error: {type(exc).__name__}: {exc}")
        finally:
            self._cleanup(job, workspace)

    def _inspect(self, job: Job) -> media.MediaInfo:
        job.set_stage(JobStage.INSPECTING)
        source = job.request.source_path
        if not source.exists():
            raise media.MediaError(f"{source.name} no longer exists at {source.parent}.")

        info = media.probe(source)
        if not info.has_audio:
            raise media.MediaError(f"{source.name} has no audio track to transcribe.")

        job.update(media_duration=info.duration, is_video=info.is_video)
        job.report(JobStage.INSPECTING, 1.0)
        log.info(
            "Job %s: %s (%s, %s)",
            job.state.id,
            source.name,
            "video" if info.is_video else "audio",
            f"{info.duration:.1f}s" if info.duration else "unknown length",
        )
        return info

    def _prepare_audio(self, job: Job, info: media.MediaInfo, workspace: Path) -> np.ndarray:
        job.set_stage(
            JobStage.EXTRACTING_AUDIO,
            message="Extracting audio from video…" if info.is_video else "Preparing audio…",
        )
        workspace.mkdir(parents=True, exist_ok=True)
        wav_path = workspace / "audio.wav"

        media.extract_audio(
            info.path,
            wav_path,
            duration=info.duration,
            on_progress=lambda fraction: job.report(JobStage.EXTRACTING_AUDIO, fraction),
            should_cancel=lambda: job.cancel_requested,
        )
        samples = media.load_pcm16_wav(wav_path)
        if job.state.media_duration is None:
            job.update(media_duration=media.audio_duration(samples))
        return samples

    def _run_engine(self, job: Job, samples: np.ndarray) -> TranscribeResult:
        model = job.state.model
        if not self._engine.model_is_local(model):
            job.set_stage(JobStage.DOWNLOADING_MODEL)
            self._engine.ensure_model(model)

        job.set_stage(
            JobStage.LOADING_MODEL,
            message="Whisper model ready…" if self._engine.model_is_loaded(model) else None,
        )

        options = TranscribeOptions.from_settings(
            self._settings,
            model=model,
            language=job.request.language,
            task=job.request.task,
            initial_prompt=job.request.initial_prompt,
        )

        return self._engine.transcribe(
            samples,
            options,
            on_progress=lambda fraction: job.report(JobStage.TRANSCRIBING, fraction),
            should_cancel=lambda: job.cancel_requested,
            on_inference_start=lambda: job.set_stage(JobStage.TRANSCRIBING),
        )

    def _write(self, job: Job, result: TranscribeResult, *, duration: float | None) -> None:
        if job.cancel_requested:
            raise TranscriptionCancelled("Transcription cancelled.")

        job.set_stage(JobStage.WRITING_OUTPUT)
        request = job.request
        transcript = build_transcript(
            source_name=request.name,
            raw=result.raw,
            model=result.model,
            task=request.task,
            duration=duration,
        )
        outputs = write_outputs(
            transcript,
            output_dir=request.output_dir,
            formats=job.state.formats,
            stem=Path(request.name).stem,
            overwrite=request.overwrite,
        )

        state = job.update(
            stage=JobStage.COMPLETED,
            message=STAGE_MESSAGES[JobStage.COMPLETED],
            progress=1.0,
            outputs=outputs,
            transcript=transcript,
            detected_language=result.language,
            finished_at=time.time(),
        )
        log.info(
            "Job %s finished in %.1fs (%sx realtime) -> %s",
            state.id,
            state.elapsed,
            state.speed_factor or "?",
            ", ".join(str(p) for p in outputs.values()),
        )

    def _fail(self, job: Job, message: str) -> None:
        log.error("Job %s failed: %s", job.state.id, message)
        job.update(
            stage=JobStage.FAILED,
            message=STAGE_MESSAGES[JobStage.FAILED],
            error=message,
            finished_at=time.time(),
        )

    def _cleanup(self, job: Job, workspace: Path | None = None) -> None:
        if self._settings.keep_temp_files:
            return
        if workspace is not None and workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)
        if job.request.cleanup_source:
            source = job.request.source_path
            source.unlink(missing_ok=True)
            parent = source.parent
            if parent.name.startswith("upload-") and parent.parent == self._settings.temp_dir:
                shutil.rmtree(parent, ignore_errors=True)
