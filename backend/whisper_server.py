"""FastAPI application exposing local Whisper transcription.

Run it directly (``python -m backend.whisper_server``), through
``scripts/run_backend.sh``, or let the macOS app start it on demand. It only
ever binds to loopback.

    GET    /health                  liveness + engine/ffmpeg/model report
    GET    /config                  defaults for the UI
    GET    /models/status           are the weights on disk / in memory?
    POST   /models/preload          warm the model before the first file
    POST   /transcribe              multipart upload -> job
    POST   /transcribe/local        JSON {path} -> job (no copy; app uses this)
    GET    /jobs                    every job this session
    GET    /jobs/{id}               progress, stage, outputs
    DELETE /jobs/{id}               cancel
    GET    /jobs/{id}/transcript    segments + words for the viewer
"""

from __future__ import annotations

import shutil
import threading
import uuid
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from . import __version__, auth, media, transcriber
from .config import SESSION_FILE, SUPPORTED_EXTENSIONS, Settings, get_settings
from .formats import UnsupportedFormatError, normalise_formats
from .jobs import JobRequest, JobStore
from .logging_setup import LOG_FILENAME, configure_logging, get_logger
from .schemas import (
    ConfigResponse,
    EngineInfo,
    HealthResponse,
    JobListResponse,
    JobResponse,
    LocalTranscribeRequest,
    ModelStatusResponse,
    TranscriptResponse,
)

log = get_logger(__name__)

_UPLOAD_CHUNK = 4 * 1024 * 1024


# --------------------------------------------------------------------------
# Lifespan
# --------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    settings.ensure_dirs()
    configure_logging(settings.log_dir, settings.log_level)
    # Must happen before mlx-whisper imports huggingface_hub.
    settings.apply_model_cache_env()

    app.state.engine = getattr(app.state, "engine", None) or transcriber.MLXEngine(
        settings.fallback_model
    )
    app.state.jobs = JobStore(settings, engine=app.state.engine)
    app.state.preloading = threading.Event()
    app.state.preload_error = None

    auth.write_session_file(settings, version=__version__)
    log.info(
        "WhisperMac backend %s listening on %s (model=%s)",
        __version__,
        settings.base_url,
        settings.model,
    )
    if media.find_tool("ffmpeg") is None:
        log.warning("ffmpeg was not found on PATH — transcription will fail until it is installed.")

    try:
        yield
    finally:
        log.info("Shutting down; cancelling in-flight jobs.")
        app.state.jobs.shutdown(wait=False)
        auth.clear_session_file()


def create_app(
    settings: Settings | None = None, *, engine: transcriber.Engine | None = None
) -> FastAPI:
    """Build the app.

    ``engine`` swaps the MLX backend for an alternative implementation — the test
    suite passes a stub so the API can be exercised without 3 GB of weights.
    """
    resolved = settings or get_settings()
    app = FastAPI(
        title="WhisperMac",
        version=__version__,
        summary="Local, private speech-to-text on Apple Silicon.",
        lifespan=lifespan,
    )
    app.state.settings = resolved
    app.state.engine = engine

    @app.exception_handler(UnsupportedFormatError)
    async def _bad_format(_request: Request, exc: UnsupportedFormatError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(media.MediaError)
    async def _bad_media(_request: Request, exc: media.MediaError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    _register_routes(app)
    return app


# --------------------------------------------------------------------------
# Dependencies & helpers
# --------------------------------------------------------------------------


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _store(request: Request) -> JobStore:
    return request.app.state.jobs


def _parse_formats(raw: Iterable[str] | None) -> tuple[str, ...]:
    """Accept ``?formats=txt&formats=srt`` and ``formats=txt,srt`` alike."""
    if not raw:
        return ("txt",)
    flattened: list[str] = []
    for value in raw:
        flattened.extend(part for part in str(value).replace(" ", "").split(",") if part)
    return normalise_formats(flattened)


def _resolve_source(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    try:
        path = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=404, detail=f"No such file: {raw_path}") from exc

    if not path.is_file():
        raise HTTPException(
            status_code=400, detail=f"{path.name} is not a file. Drop a single media file."
        )
    _check_extension(path.name)
    return path


def _check_extension(filename: str) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"'{suffix or filename}' is not a supported media type. "
                "Supported: " + ", ".join(sorted(e.lstrip('.') for e in SUPPORTED_EXTENSIONS)) + "."
            ),
        )


def _resolve_output_dir(raw: str | None, settings: Settings) -> Path:
    directory = Path(raw).expanduser() if raw else settings.output_dir
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot write to {directory}: {exc.strerror}. Pick a different output folder.",
        ) from exc
    return directory.resolve()


async def _save_upload(upload: UploadFile, settings: Settings) -> Path:
    """Stream an upload to a temp file, enforcing the size cap as we go."""
    filename = Path(upload.filename or "upload").name
    _check_extension(filename)

    workspace = settings.temp_dir / f"upload-{uuid.uuid4().hex[:12]}"
    workspace.mkdir(parents=True, exist_ok=True)
    destination = workspace / filename

    written = 0
    try:
        with destination.open("wb") as sink:
            while chunk := await upload.read(_UPLOAD_CHUNK):
                written += len(chunk)
                if written > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"{filename} is larger than the "
                            f"{settings.max_upload_bytes // 1024**3} GB upload limit. "
                            "Use the app (or /transcribe/local) to transcribe it in place instead."
                        ),
                    )
                sink.write(chunk)
    except HTTPException:
        shutil.rmtree(workspace, ignore_errors=True)
        raise
    except OSError as exc:
        shutil.rmtree(workspace, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Could not save upload: {exc}") from exc
    finally:
        await upload.close()

    if written == 0:
        shutil.rmtree(workspace, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"{filename} is empty.")
    return destination


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


def _register_routes(app: FastAPI) -> None:
    protected = [Depends(auth.require_token)]

    # ------------------------------------------------------------- health

    @app.get("/health", response_model=HealthResponse, tags=["status"])
    async def health(request: Request) -> HealthResponse:
        settings = _settings(request)
        engine = request.app.state.engine
        status_info = _engine_status(engine)
        store: JobStore | None = getattr(request.app.state, "jobs", None)
        active = (
            sum(1 for job in store.list() if not job.stage.is_terminal) if store is not None else 0
        )
        return HealthResponse(
            version=__version__,
            model=settings.model,
            engine=EngineInfo(
                available=status_info.available,
                device=status_info.device,
                mlx_version=status_info.mlx_version,
                mlx_whisper_version=status_info.mlx_whisper_version,
                error=status_info.error,
            ),
            ffmpeg=media.ffmpeg_version(),
            model_cached=engine.model_is_local(settings.model),
            model_loaded=engine.model_is_loaded(settings.model),
            active_jobs=active,
        )

    @app.get("/config", response_model=ConfigResponse, tags=["status"], dependencies=protected)
    async def read_config(request: Request) -> ConfigResponse:
        settings = _settings(request)
        return ConfigResponse(
            model=settings.model,
            output_dir=str(settings.output_dir),
            default_formats=["txt", "srt"],
            max_upload_bytes=settings.max_upload_bytes,
            word_timestamps=settings.word_timestamps,
            temperature=settings.temperature,
            log_path=str(settings.log_dir / LOG_FILENAME),
        )

    # ------------------------------------------------------------- models

    @app.get(
        "/models/status", response_model=ModelStatusResponse, tags=["models"], dependencies=protected
    )
    async def model_status(request: Request) -> ModelStatusResponse:
        settings = _settings(request)
        engine = request.app.state.engine
        return ModelStatusResponse(
            model=settings.model,
            cached=engine.model_is_local(settings.model),
            loaded=engine.model_is_loaded(settings.model),
            cache_dir=str(settings.model_cache_dir),
            preloading=request.app.state.preloading.is_set(),
            error=request.app.state.preload_error,
        )

    @app.post(
        "/models/preload",
        response_model=ModelStatusResponse,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["models"],
        dependencies=protected,
    )
    async def preload_model(request: Request) -> ModelStatusResponse:
        """Download (once) and warm the weights so the first file starts instantly."""
        settings = _settings(request)
        app_state = request.app.state
        engine = app_state.engine

        if not app_state.preloading.is_set() and not engine.model_is_loaded(settings.model):
            app_state.preloading.set()
            app_state.preload_error = None
            threading.Thread(
                target=_preload_worker,
                args=(app_state, settings, engine),
                name="whisper-preload",
                daemon=True,
            ).start()

        return ModelStatusResponse(
            model=settings.model,
            cached=engine.model_is_local(settings.model),
            loaded=engine.model_is_loaded(settings.model),
            cache_dir=str(settings.model_cache_dir),
            preloading=app_state.preloading.is_set(),
            error=app_state.preload_error,
        )

    # --------------------------------------------------------- transcribe

    @app.post(
        "/transcribe",
        response_model=JobResponse,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["transcribe"],
        dependencies=protected,
    )
    async def transcribe_upload(
        request: Request,
        file: Annotated[UploadFile, File(description="Audio or video file")],
        formats: Annotated[list[str] | None, Form()] = None,
        output_dir: Annotated[str | None, Form()] = None,
        model: Annotated[str | None, Form()] = None,
        language: Annotated[str | None, Form()] = None,
        task: Annotated[str, Form()] = "transcribe",
        initial_prompt: Annotated[str | None, Form()] = None,
        overwrite: Annotated[bool, Form()] = False,
    ) -> JobResponse:
        settings = _settings(request)
        requested_formats = _parse_formats(formats)
        target_dir = _resolve_output_dir(output_dir, settings)
        saved = await _save_upload(file, settings)

        state = _store(request).submit(
            JobRequest(
                source_path=saved,
                output_dir=target_dir,
                formats=requested_formats,
                model=model or None,
                language=language or None,
                task=task,
                initial_prompt=initial_prompt or None,
                overwrite=overwrite,
                cleanup_source=True,
                display_name=saved.name,
            )
        )
        return JobResponse.from_state(state)

    @app.post(
        "/transcribe/local",
        response_model=JobResponse,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["transcribe"],
        dependencies=protected,
    )
    async def transcribe_local(request: Request, body: LocalTranscribeRequest) -> JobResponse:
        """Transcribe a file in place — what the app uses, so nothing is copied."""
        settings = _settings(request)
        source = _resolve_source(body.path)
        state = _store(request).submit(
            JobRequest(
                source_path=source,
                output_dir=_resolve_output_dir(body.output_dir, settings),
                formats=_parse_formats(body.formats),
                model=body.model or None,
                language=body.language or None,
                task=body.task,
                initial_prompt=body.initial_prompt or None,
                overwrite=body.overwrite,
            )
        )
        return JobResponse.from_state(state)

    # --------------------------------------------------------------- jobs

    @app.get("/jobs", response_model=JobListResponse, tags=["jobs"], dependencies=protected)
    async def list_jobs(request: Request) -> JobListResponse:
        return JobListResponse(
            jobs=[JobResponse.from_state(state) for state in _store(request).list()]
        )

    @app.get("/jobs/{job_id}", response_model=JobResponse, tags=["jobs"], dependencies=protected)
    async def get_job(request: Request, job_id: str) -> JobResponse:
        state = _store(request).get(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Unknown job '{job_id}'.")
        return JobResponse.from_state(state)

    @app.delete("/jobs/{job_id}", response_model=JobResponse, tags=["jobs"], dependencies=protected)
    async def cancel_job(request: Request, job_id: str) -> JobResponse:
        state = _store(request).cancel(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Unknown job '{job_id}'.")
        return JobResponse.from_state(state)

    @app.get(
        "/jobs/{job_id}/transcript",
        response_model=TranscriptResponse,
        tags=["jobs"],
        dependencies=protected,
    )
    async def get_transcript(request: Request, job_id: str) -> TranscriptResponse:
        store = _store(request)
        if store.get(job_id) is None:
            raise HTTPException(status_code=404, detail=f"Unknown job '{job_id}'.")
        transcript = store.transcript(job_id)
        if transcript is None:
            raise HTTPException(
                status_code=409, detail="This job has not produced a transcript yet."
            )
        return TranscriptResponse.from_transcript(job_id, transcript)


def _engine_status(engine: transcriber.Engine) -> transcriber.EngineStatus:
    reporter = getattr(engine, "status", None)
    return reporter() if callable(reporter) else transcriber.engine_status()


def _preload_worker(app_state, settings: Settings, engine: transcriber.Engine) -> None:
    """Download + load the model off the request path."""
    import numpy as np

    try:
        engine.ensure_model(settings.model)
        # 100 ms of silence is enough to force a full weight load.
        engine.transcribe(
            np.zeros(1600, dtype=np.float32),
            transcriber.TranscribeOptions.from_settings(settings, language="en"),
        )
        log.info("Model preloaded: %s", settings.model)
    except Exception as exc:  # noqa: BLE001 - surfaced through /models/status
        app_state.preload_error = str(exc)
        log.error("Model preload failed: %s", exc)
    finally:
        app_state.preloading.clear()


app = create_app()


def main() -> None:
    settings = get_settings()
    settings.ensure_dirs()
    configure_logging(settings.log_dir, settings.log_level)
    print(f"WhisperMac backend on {settings.base_url}  (token in {SESSION_FILE})")
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        access_log=False,
    )


if __name__ == "__main__":
    main()
