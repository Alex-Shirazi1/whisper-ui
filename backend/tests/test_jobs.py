"""Job pipeline: stages, progress mapping, cancellation, cleanup, failures."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from backend import media
from backend.config import Settings
from backend.jobs import STAGE_MESSAGES, Job, JobRequest, JobStage, JobState, JobStore
from backend.tests.conftest import StubEngine, requires_ffmpeg


def _wait_for_terminal(store: JobStore, job_id: str, timeout: float = 30.0) -> JobState:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = store.get(job_id)
        assert state is not None
        if state.stage.is_terminal:
            return state
        time.sleep(0.02)
    raise AssertionError(f"Job {job_id} never finished; last stage was {state.stage}")


@pytest.fixture
def store(settings: Settings, engine: StubEngine) -> Iterator[JobStore]:
    instance = JobStore(settings, engine=engine)
    yield instance
    instance.shutdown(wait=True)


@pytest.mark.unit
class TestProgressMapping:
    def _job(self) -> Job:
        request = JobRequest(source_path=Path("a.wav"), output_dir=Path("/tmp"))
        state = JobState(
            id="x",
            name="a.wav",
            source_path=request.source_path,
            output_dir=request.output_dir,
            formats=("txt",),
            model="stub",
            task="transcribe",
        )
        return Job(state, request)

    def test_stage_progress_is_scaled_into_the_global_bar(self) -> None:
        job = self._job()
        job.set_stage(JobStage.TRANSCRIBING)
        job.report(JobStage.TRANSCRIBING, 0.5)
        # Transcription spans 0.17..0.97, so halfway is ~0.57.
        assert job.state.progress == pytest.approx(0.57, abs=0.01)

    def test_progress_never_goes_backwards(self) -> None:
        job = self._job()
        job.set_stage(JobStage.TRANSCRIBING)
        job.report(JobStage.TRANSCRIBING, 0.8)
        job.report(JobStage.TRANSCRIBING, 0.1)
        assert job.state.progress == pytest.approx(0.81, abs=0.01)

    def test_reports_for_a_stale_stage_are_ignored(self) -> None:
        job = self._job()
        job.set_stage(JobStage.TRANSCRIBING)
        job.report(JobStage.EXTRACTING_AUDIO, 1.0)
        assert job.state.progress == pytest.approx(0.17, abs=0.001)

    def test_time_estimate_advances_the_bar_between_windows(self) -> None:
        job = self._job()
        job.update(started_at=time.time() - 5.0, media_duration=100.0)
        job.set_stage(JobStage.TRANSCRIBING)
        before = job.state.progress
        # 5s elapsed against an expected 10s (100s * 0.1) is ~50% of the stage.
        after = job.blend_estimate(0.1)
        assert after.progress > before
        assert after.progress < 0.97

    def test_time_estimate_is_capped_below_completion(self) -> None:
        job = self._job()
        job.update(started_at=time.time() - 10_000, media_duration=10.0)
        job.set_stage(JobStage.TRANSCRIBING)
        assert job.blend_estimate(0.1).progress < 0.97

    def test_eta_and_speed_factor(self) -> None:
        job = self._job()
        job.update(started_at=time.time() - 10, media_duration=100.0)
        job.set_stage(JobStage.TRANSCRIBING)
        job.report(JobStage.TRANSCRIBING, 0.5)
        assert job.state.eta_seconds is not None and job.state.eta_seconds > 0
        finished = job.update(finished_at=job.state.started_at + 20)
        assert finished.speed_factor == pytest.approx(5.0)


@requires_ffmpeg
@pytest.mark.integration
class TestPipeline:
    def test_audio_file_produces_all_formats(self, store: JobStore, silent_wav: Path) -> None:
        state = store.submit(
            JobRequest(
                source_path=silent_wav,
                output_dir=store._settings.output_dir,
                formats=("txt", "srt", "vtt", "json"),
            )
        )
        final = _wait_for_terminal(store, state.id)

        assert final.stage is JobStage.COMPLETED
        assert final.message == STAGE_MESSAGES[JobStage.COMPLETED]
        assert final.progress == 1.0
        assert set(final.outputs) == {"txt", "srt", "vtt", "json"}
        assert final.outputs["txt"].read_text().startswith("Hello there.")
        assert final.detected_language == "en"
        assert final.media_duration == pytest.approx(2.0, abs=0.2)
        assert final.error is None

    def test_video_file_goes_through_extraction(self, store: JobStore, tone_mp4: Path) -> None:
        state = store.submit(
            JobRequest(source_path=tone_mp4, output_dir=store._settings.output_dir)
        )
        final = _wait_for_terminal(store, state.id)
        assert final.stage is JobStage.COMPLETED
        assert final.is_video

    def test_transcript_is_retrievable_after_completion(
        self, store: JobStore, silent_wav: Path
    ) -> None:
        state = store.submit(
            JobRequest(source_path=silent_wav, output_dir=store._settings.output_dir)
        )
        _wait_for_terminal(store, state.id)
        transcript = store.transcript(state.id)
        assert transcript is not None
        assert len(transcript.segments) == 2

    def test_temp_workspace_is_cleaned_up(self, store: JobStore, silent_wav: Path) -> None:
        state = store.submit(
            JobRequest(source_path=silent_wav, output_dir=store._settings.output_dir)
        )
        _wait_for_terminal(store, state.id)
        leftovers = list(store._settings.temp_dir.glob("job-*"))
        assert leftovers == []

    def test_uploaded_source_copy_is_deleted(self, store: JobStore, silent_wav: Path) -> None:
        upload_dir = store._settings.temp_dir / "upload-abc123"
        upload_dir.mkdir(parents=True)
        copy = upload_dir / "clip.wav"
        copy.write_bytes(silent_wav.read_bytes())

        state = store.submit(
            JobRequest(
                source_path=copy,
                output_dir=store._settings.output_dir,
                cleanup_source=True,
            )
        )
        _wait_for_terminal(store, state.id)
        assert not copy.exists()
        assert not upload_dir.exists()

    def test_missing_file_fails_with_a_readable_message(self, store: JobStore) -> None:
        state = store.submit(
            JobRequest(
                source_path=store._settings.temp_dir / "ghost.wav",
                output_dir=store._settings.output_dir,
            )
        )
        final = _wait_for_terminal(store, state.id)
        assert final.stage is JobStage.FAILED
        assert final.error and "ghost.wav" in final.error

    def test_engine_failure_is_reported_not_raised(
        self, settings: Settings, silent_wav: Path
    ) -> None:
        store = JobStore(settings, engine=StubEngine(fail_with=RuntimeError("metal exploded")))
        try:
            state = store.submit(
                JobRequest(source_path=silent_wav, output_dir=settings.output_dir)
            )
            final = _wait_for_terminal(store, state.id)
            assert final.stage is JobStage.FAILED
            assert "metal exploded" in (final.error or "")
        finally:
            store.shutdown(wait=True)

    def test_cancellation_marks_the_job_cancelled(
        self, settings: Settings, silent_wav: Path
    ) -> None:
        store = JobStore(settings, engine=StubEngine(steps=400, step_delay=0.01))
        try:
            state = store.submit(
                JobRequest(source_path=silent_wav, output_dir=settings.output_dir)
            )
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if store.get(state.id).stage is JobStage.TRANSCRIBING:
                    break
                time.sleep(0.02)

            store.cancel(state.id)
            final = _wait_for_terminal(store, state.id, timeout=10)
            assert final.stage is JobStage.CANCELLED
            assert not list(settings.output_dir.glob("*.txt"))
        finally:
            store.shutdown(wait=True)


@requires_ffmpeg
@pytest.mark.integration
class TestStore:
    def test_jobs_run_in_submission_order(self, store: JobStore, silent_wav: Path) -> None:
        ids = [
            store.submit(
                JobRequest(source_path=silent_wav, output_dir=store._settings.output_dir)
            ).id
            for _ in range(3)
        ]
        for job_id in ids:
            _wait_for_terminal(store, job_id)
        assert [state.id for state in store.list()] == ids

    def test_unknown_job_returns_none(self, store: JobStore) -> None:
        assert store.get("nope") is None
        assert store.cancel("nope") is None

    def test_finished_jobs_are_pruned_after_retention(
        self, settings: Settings, engine: StubEngine, silent_wav: Path
    ) -> None:
        store = JobStore(replace(settings, job_retention_seconds=0), engine=engine)
        try:
            first = store.submit(
                JobRequest(source_path=silent_wav, output_dir=settings.output_dir)
            )
            _wait_for_terminal(store, first.id)
            store.submit(JobRequest(source_path=silent_wav, output_dir=settings.output_dir))
            assert store.get(first.id) is None
        finally:
            store.shutdown(wait=True)

    def test_no_audio_track_fails_before_the_engine_runs(
        self, store: JobStore, tmp_path: Path
    ) -> None:
        broken = tmp_path / "corrupt.mp4"
        broken.write_bytes(b"\x00" * 2048)
        state = store.submit(
            JobRequest(source_path=broken, output_dir=store._settings.output_dir)
        )
        final = _wait_for_terminal(store, state.id)
        assert final.stage is JobStage.FAILED
        assert isinstance(final.error, str) and "corrupt.mp4" in final.error


@pytest.mark.unit
def test_stage_messages_cover_every_stage() -> None:
    assert set(STAGE_MESSAGES) == set(JobStage)


@pytest.mark.unit
def test_media_error_messages_are_user_facing() -> None:
    message = media._explain_ffmpeg_failure(Path("clip.mov"), "Invalid data found when processing input")
    assert "clip.mov" in message
    assert "corrupt" in message
