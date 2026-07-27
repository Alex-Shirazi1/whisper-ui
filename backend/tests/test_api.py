"""HTTP contract tests. No MLX, no weights, no network."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import auth, config
from backend.auth import TOKEN_HEADER
from backend.config import Settings
from backend.tests.conftest import StubEngine, requires_ffmpeg
from backend.whisper_server import create_app

AUTH = {TOKEN_HEADER: "test-token"}


@pytest.fixture
def client(
    settings: Settings,
    engine: StubEngine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    # Never touch the real ~/.whisper-mac while testing.
    state_dir = tmp_path / "state"
    monkeypatch.setattr(config, "STATE_DIR", state_dir)
    monkeypatch.setattr(auth, "STATE_DIR", state_dir)
    monkeypatch.setattr(auth, "SESSION_FILE", state_dir / "session.json")

    app = create_app(settings, engine=engine)
    with TestClient(app) as test_client:
        yield test_client


def _wait_for(client: TestClient, job_id: str, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        payload = client.get(f"/jobs/{job_id}", headers=AUTH).json()
        if payload["stage"] in {"completed", "failed", "cancelled"}:
            return payload
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} stuck at {payload['stage']}")


@pytest.mark.unit
class TestStatusEndpoints:
    def test_health_needs_no_token(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model"] == "stub/whisper-test"
        assert "available" in body["engine"]

    def test_config_reports_defaults(self, client: TestClient, settings: Settings) -> None:
        body = client.get("/config", headers=AUTH).json()
        assert body["output_dir"] == str(settings.output_dir)
        assert body["formats"] == ["txt", "srt", "vtt", "json"]
        assert body["default_formats"] == ["txt", "srt"]
        assert ".mp4" in body["supported_extensions"]

    def test_model_status(self, client: TestClient) -> None:
        body = client.get("/models/status", headers=AUTH).json()
        assert body["model"] == "stub/whisper-test"
        assert body["cached"] is True
        assert body["loaded"] is True
        assert body["preloading"] is False

    def test_session_file_is_written_with_owner_only_permissions(
        self, client: TestClient
    ) -> None:
        assert auth.SESSION_FILE.exists()
        assert auth.SESSION_FILE.stat().st_mode & 0o777 == 0o600
        assert auth.read_session_file()["token"] == "test-token"


@pytest.mark.unit
class TestAuth:
    def test_protected_route_rejects_missing_token(self, client: TestClient) -> None:
        assert client.get("/jobs").status_code == 401

    def test_protected_route_rejects_wrong_token(self, client: TestClient) -> None:
        response = client.get("/jobs", headers={TOKEN_HEADER: "nope"})
        assert response.status_code == 401
        assert "session token" in response.json()["detail"]

    def test_bearer_header_is_accepted(self, client: TestClient) -> None:
        response = client.get("/jobs", headers={"Authorization": "Bearer test-token"})
        assert response.status_code == 200


@pytest.mark.unit
class TestValidation:
    def test_unsupported_extension_is_rejected(self, client: TestClient, tmp_path: Path) -> None:
        document = tmp_path / "notes.pdf"
        document.write_bytes(b"%PDF-1.4")
        response = client.post(
            "/transcribe/local", json={"path": str(document)}, headers=AUTH
        )
        assert response.status_code == 415
        assert "not a supported media type" in response.json()["detail"]

    def test_missing_file_is_404(self, client: TestClient, tmp_path: Path) -> None:
        response = client.post(
            "/transcribe/local", json={"path": str(tmp_path / "ghost.wav")}, headers=AUTH
        )
        assert response.status_code == 404

    def test_directory_is_rejected(self, client: TestClient, tmp_path: Path) -> None:
        response = client.post("/transcribe/local", json={"path": str(tmp_path)}, headers=AUTH)
        assert response.status_code in {400, 415}

    def test_bad_output_format_is_400(self, client: TestClient, silent_wav: Path) -> None:
        response = client.post(
            "/transcribe/local",
            json={"path": str(silent_wav), "formats": ["txt", "pdf"]},
            headers=AUTH,
        )
        assert response.status_code == 400
        assert "pdf" in response.json()["detail"]

    def test_invalid_task_is_422(self, client: TestClient, silent_wav: Path) -> None:
        response = client.post(
            "/transcribe/local",
            json={"path": str(silent_wav), "task": "summarise"},
            headers=AUTH,
        )
        assert response.status_code == 422

    def test_unknown_job_is_404(self, client: TestClient) -> None:
        assert client.get("/jobs/deadbeef", headers=AUTH).status_code == 404
        assert client.delete("/jobs/deadbeef", headers=AUTH).status_code == 404
        assert client.get("/jobs/deadbeef/transcript", headers=AUTH).status_code == 404


@requires_ffmpeg
@pytest.mark.integration
class TestTranscribeLocal:
    def test_happy_path(self, client: TestClient, silent_wav: Path, settings: Settings) -> None:
        response = client.post(
            "/transcribe/local",
            json={"path": str(silent_wav), "formats": ["txt", "srt"]},
            headers=AUTH,
        )
        assert response.status_code == 202
        job = response.json()
        assert job["stage"] in {"queued", "inspecting", "extracting_audio"}
        assert job["formats"] == ["txt", "srt"]

        final = _wait_for(client, job["id"])
        assert final["stage"] == "completed"
        assert final["progress"] == 1.0
        assert final["status_message"] == "Finished"
        assert final["detected_language"] == "en"
        assert {output["format"] for output in final["outputs"]} == {"txt", "srt"}
        for output in final["outputs"]:
            assert Path(output["path"]).exists()
            assert Path(output["path"]).parent == settings.output_dir

    def test_custom_output_dir(self, client: TestClient, silent_wav: Path, tmp_path: Path) -> None:
        target = tmp_path / "Transcripts" / "2026"
        job = client.post(
            "/transcribe/local",
            json={"path": str(silent_wav), "output_dir": str(target)},
            headers=AUTH,
        ).json()
        final = _wait_for(client, job["id"])
        assert final["stage"] == "completed"
        assert Path(final["outputs"][0]["path"]).parent == target

    def test_transcript_endpoint_returns_segments_and_words(
        self, client: TestClient, silent_wav: Path
    ) -> None:
        job = client.post("/transcribe/local", json={"path": str(silent_wav)}, headers=AUTH).json()
        _wait_for(client, job["id"])

        body = client.get(f"/jobs/{job['id']}/transcript", headers=AUTH).json()
        assert body["language"] == "en"
        assert body["word_count"] == 6
        assert len(body["segments"]) == 2
        assert body["segments"][0]["words"][0]["word"] == "Hello"
        assert body["segments"][0]["speaker"] is None

    def test_transcript_is_409_before_completion(
        self, client: TestClient, silent_wav: Path
    ) -> None:
        job = client.post("/transcribe/local", json={"path": str(silent_wav)}, headers=AUTH).json()
        early = client.get(f"/jobs/{job['id']}/transcript", headers=AUTH)
        assert early.status_code in {200, 409}

    def test_jobs_list_includes_the_job(self, client: TestClient, silent_wav: Path) -> None:
        job = client.post("/transcribe/local", json={"path": str(silent_wav)}, headers=AUTH).json()
        _wait_for(client, job["id"])
        listing = client.get("/jobs", headers=AUTH).json()["jobs"]
        assert [entry["id"] for entry in listing] == [job["id"]]

    def test_cancel_is_idempotent_after_completion(
        self, client: TestClient, silent_wav: Path
    ) -> None:
        job = client.post("/transcribe/local", json={"path": str(silent_wav)}, headers=AUTH).json()
        _wait_for(client, job["id"])
        response = client.delete(f"/jobs/{job['id']}", headers=AUTH)
        assert response.status_code == 200
        assert response.json()["stage"] == "completed"


@requires_ffmpeg
@pytest.mark.integration
class TestTranscribeUpload:
    def test_upload_is_transcribed_and_the_copy_removed(
        self, client: TestClient, silent_wav: Path, settings: Settings
    ) -> None:
        with silent_wav.open("rb") as handle:
            response = client.post(
                "/transcribe",
                files={"file": ("dropped.wav", handle, "audio/wav")},
                data={"formats": "txt,vtt"},
                headers=AUTH,
            )
        assert response.status_code == 202
        job = response.json()
        assert job["name"] == "dropped.wav"

        final = _wait_for(client, job["id"])
        assert final["stage"] == "completed"
        assert {output["filename"] for output in final["outputs"]} == {
            "dropped.txt",
            "dropped.vtt",
        }
        assert list(settings.temp_dir.glob("upload-*")) == []

    def test_upload_with_unsupported_extension_is_415(self, client: TestClient) -> None:
        response = client.post(
            "/transcribe",
            files={"file": ("notes.pdf", b"%PDF-1.4", "application/pdf")},
            headers=AUTH,
        )
        assert response.status_code == 415

    def test_empty_upload_is_400(self, client: TestClient) -> None:
        response = client.post(
            "/transcribe", files={"file": ("empty.wav", b"", "audio/wav")}, headers=AUTH
        )
        assert response.status_code == 400

    def test_upload_over_the_size_cap_is_413(
        self, settings: Settings, engine: StubEngine, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dataclasses import replace

        state_dir = tmp_path / "state2"
        monkeypatch.setattr(config, "STATE_DIR", state_dir)
        monkeypatch.setattr(auth, "STATE_DIR", state_dir)
        monkeypatch.setattr(auth, "SESSION_FILE", state_dir / "session.json")

        tiny_limit = replace(settings, max_upload_bytes=1024)
        app = create_app(tiny_limit, engine=engine)
        with TestClient(app) as small_client:
            response = small_client.post(
                "/transcribe",
                files={"file": ("big.wav", b"\x00" * 4096, "audio/wav")},
                headers=AUTH,
            )
        assert response.status_code == 413
        assert list(tiny_limit.temp_dir.glob("upload-*")) == []
