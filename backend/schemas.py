"""Pydantic request/response models — the contract the macOS app codes against."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .config import OUTPUT_FORMATS, SUPPORTED_EXTENSIONS
from .formats import Transcript
from .jobs import STAGE_MESSAGES, JobStage, JobState


class EngineInfo(BaseModel):
    available: bool
    device: str | None = None
    mlx_version: str | None = None
    mlx_whisper_version: str | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    model: str
    engine: EngineInfo
    ffmpeg: str | None = Field(default=None, description="ffmpeg version banner, if installed")
    model_cached: bool = Field(default=False, description="Weights present on disk")
    model_loaded: bool = Field(default=False, description="Weights resident in memory")
    active_jobs: int = 0

    model_config = ConfigDict(protected_namespaces=())


class ConfigResponse(BaseModel):
    """Defaults the app uses to seed its UI on first launch."""

    model: str
    output_dir: str
    formats: list[str] = Field(default_factory=lambda: list(OUTPUT_FORMATS))
    default_formats: list[str]
    supported_extensions: list[str] = Field(
        default_factory=lambda: sorted(SUPPORTED_EXTENSIONS)
    )
    max_upload_bytes: int
    word_timestamps: bool
    temperature: float
    log_path: str

    model_config = ConfigDict(protected_namespaces=())


class ModelStatusResponse(BaseModel):
    model: str
    cached: bool
    loaded: bool
    cache_dir: str
    preloading: bool = False
    error: str | None = None

    model_config = ConfigDict(protected_namespaces=())


class LocalTranscribeRequest(BaseModel):
    """Transcribe a file already on disk — no upload, no copy."""

    path: str = Field(description="Absolute path to an audio or video file")
    formats: list[str] = Field(default_factory=lambda: ["txt"])
    output_dir: str | None = None
    model: str | None = None
    language: str | None = Field(default=None, description="ISO code, or null to auto-detect")
    task: str = Field(default="transcribe", pattern="^(transcribe|translate)$")
    initial_prompt: str | None = None
    overwrite: bool = False

    model_config = ConfigDict(protected_namespaces=())


class OutputFile(BaseModel):
    format: str
    path: str
    filename: str


class JobResponse(BaseModel):
    id: str
    name: str
    stage: JobStage
    status_message: str
    progress: float
    indeterminate: bool
    source_path: str
    output_dir: str
    formats: list[str]
    model: str
    task: str
    outputs: list[OutputFile] = Field(default_factory=list)
    error: str | None = None
    media_duration: float | None = None
    detected_language: str | None = None
    is_video: bool = False
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None
    speed_factor: float | None = None
    created_at: float
    finished_at: float | None = None

    model_config = ConfigDict(protected_namespaces=())

    @classmethod
    def from_state(cls, state: JobState) -> JobResponse:
        return cls(
            id=state.id,
            name=state.name,
            stage=state.stage,
            status_message=state.error or state.message or STAGE_MESSAGES[state.stage],
            progress=round(state.progress, 4),
            indeterminate=state.indeterminate,
            source_path=str(state.source_path),
            output_dir=str(state.output_dir),
            formats=list(state.formats),
            model=state.model,
            task=state.task,
            outputs=[
                OutputFile(format=fmt, path=str(path), filename=Path(path).name)
                for fmt, path in state.outputs.items()
            ],
            error=state.error,
            media_duration=state.media_duration,
            detected_language=state.detected_language,
            is_video=state.is_video,
            elapsed_seconds=round(state.elapsed, 2),
            eta_seconds=state.eta_seconds,
            speed_factor=state.speed_factor,
            created_at=state.created_at,
            finished_at=state.finished_at,
        )


class JobListResponse(BaseModel):
    jobs: list[JobResponse]


class WordResponse(BaseModel):
    start: float
    end: float
    word: str
    probability: float | None = None


class SegmentResponse(BaseModel):
    id: int
    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[WordResponse] = Field(default_factory=list)


class TranscriptResponse(BaseModel):
    """Full transcript payload. Backs the in-app viewer, search and timestamps."""

    job_id: str
    source: str
    language: str | None
    duration: float | None
    model: str
    task: str
    word_count: int
    text: str
    segments: list[SegmentResponse]

    model_config = ConfigDict(protected_namespaces=())

    @classmethod
    def from_transcript(cls, job_id: str, transcript: Transcript) -> TranscriptResponse:
        return cls(
            job_id=job_id,
            source=transcript.source_name,
            language=transcript.language,
            duration=transcript.duration,
            model=transcript.model,
            task=transcript.task,
            word_count=transcript.word_count,
            text=transcript.text,
            segments=[
                SegmentResponse(
                    id=segment.index,
                    start=segment.start,
                    end=segment.end,
                    text=segment.text.strip(),
                    speaker=segment.speaker,
                    words=[
                        WordResponse(
                            start=word.start,
                            end=word.end,
                            word=word.text,
                            probability=word.probability,
                        )
                        for word in segment.words
                    ],
                )
                for segment in transcript.segments
            ],
        )


class ErrorResponse(BaseModel):
    detail: str
    hint: str | None = None
    context: Mapping[str, Any] | None = None
