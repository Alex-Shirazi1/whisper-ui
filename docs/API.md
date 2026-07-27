# Backend API

Base URL: `http://127.0.0.1:8756` (loopback only — it is never reachable from the
network).

## Authentication

Every route except `GET /health` requires the session token. The backend writes
it to `~/.whisper-mac/session.json` (mode `0600`) at startup and deletes it on
shutdown; the macOS app reads it from there, so there is nothing to configure.

```bash
TOKEN=$(python3 -c "import json;print(json.load(open('$HOME/.whisper-mac/session.json'))['token'])")
curl -H "X-Whisper-Token: $TOKEN" http://127.0.0.1:8756/jobs
```

`Authorization: Bearer <token>` works too. Set `WHISPER_MAC_REQUIRE_TOKEN=0` to
disable auth entirely (not recommended: any local process, including a web page
you visit, could then read your transcripts).

Interactive docs: <http://127.0.0.1:8756/docs>

---

## `GET /health`

No auth. Used by `run.sh`, the app's status pill, and the offline banner.

```json
{
  "status": "ok",
  "version": "1.0.0",
  "model": "mlx-community/whisper-large-v3-mlx",
  "engine": {
    "available": true,
    "device": "Device(gpu, 0)",
    "mlx_version": "0.32.0",
    "mlx_whisper_version": "0.4.3",
    "error": null
  },
  "ffmpeg": "ffmpeg version 7.1.1 Copyright (c) 2000-2025 the FFmpeg developers",
  "model_cached": true,
  "model_loaded": false,
  "active_jobs": 0
}
```

`model_cached` = weights on disk. `model_loaded` = weights resident in memory.

## `GET /config`

UI defaults: model, default output directory, the four output formats, every
supported file extension, upload cap, and the log path.

## `GET /models/status` · `POST /models/preload`

`preload` downloads the weights if needed and runs one tiny inference to warm
them, in a background thread; it returns `202` immediately. Poll `status` for
`{cached, loaded, preloading, error}`. The app calls `preload` once on connect so
the first real file doesn't wait on a cold load.

## `POST /transcribe/local`

Transcribe a file **in place** — nothing is copied or uploaded. This is what the
app uses.

```bash
curl -X POST http://127.0.0.1:8756/transcribe/local \
  -H "X-Whisper-Token: $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
        "path": "/Users/me/Movies/standup.mov",
        "formats": ["txt", "srt", "vtt", "json"],
        "output_dir": "/Users/me/Documents/Transcripts",
        "language": null,
        "task": "transcribe",
        "initial_prompt": null,
        "overwrite": false
      }'
```

| Field | Default | Notes |
|---|---|---|
| `path` | required | Absolute path; `~` is expanded |
| `formats` | `["txt"]` | Any of `txt`, `srt`, `vtt`, `json` |
| `output_dir` | backend default | Created if missing |
| `model` | configured model | Any MLX Whisper repo id |
| `language` | `null` | ISO code, or `null`/`"auto"` to detect |
| `task` | `"transcribe"` | Or `"translate"` (to English) |
| `initial_prompt` | `null` | Bias vocabulary, names, jargon |
| `overwrite` | `false` | When false, `name.txt` → `name-1.txt` |

Returns `202` with a job. Errors: `404` (no such file), `400` (not a file /
unwritable output dir / bad media), `415` (unsupported extension), `422`
(malformed body).

## `POST /transcribe`

Multipart upload for clients that can't share a filesystem. The copy is deleted
once the job ends.

```bash
curl -X POST http://127.0.0.1:8756/transcribe \
  -H "X-Whisper-Token: $TOKEN" \
  -F "file=@meeting.m4a" \
  -F "formats=txt,srt"
```

Rejects uploads over `WHISPER_MAC_MAX_UPLOAD_BYTES` (8 GB default) with `413`,
mid-stream, without buffering the whole body.

## `GET /jobs` · `GET /jobs/{id}` · `DELETE /jobs/{id}`

```json
{
  "id": "ef38505650a1",
  "name": "long_meeting.m4a",
  "stage": "transcribing",
  "status_message": "Transcribing…",
  "progress": 0.69,
  "indeterminate": false,
  "source_path": "/Users/me/Movies/long_meeting.m4a",
  "output_dir": "/Users/me/Desktop/whisper-mac/output",
  "formats": ["txt", "srt"],
  "model": "mlx-community/whisper-large-v3-mlx",
  "task": "transcribe",
  "outputs": [],
  "error": null,
  "media_duration": 283.3,
  "detected_language": "en",
  "is_video": false,
  "elapsed_seconds": 36.4,
  "eta_seconds": 9.1,
  "speed_factor": null,
  "created_at": 1785000000.0,
  "finished_at": null
}
```

Stages: `queued` → `inspecting` → `extracting_audio` → `downloading_model` →
`loading_model` → `transcribing` → `writing_output` → `completed`, plus `failed`
and `cancelled`. `indeterminate` is true for the two model stages, where there is
no measurable position. `progress` is monotonic. `speed_factor` is filled in on
completion (`media_duration / elapsed`).

`DELETE` cancels: a queued job never starts, a running one stops at its next
decode window. No partial files are ever written. Cancelling a finished job is a
no-op that returns its final state.

Finished jobs are kept for `WHISPER_MAC_JOB_RETENTION_SECONDS` (6 h) so the UI
can still read them, then pruned.

## `GET /jobs/{id}/transcript`

Full transcript for the in-app viewer. `409` if the job hasn't produced one yet.

```json
{
  "job_id": "ef38505650a1",
  "source": "long_meeting.m4a",
  "language": "en",
  "duration": 283.3,
  "model": "mlx-community/whisper-large-v3-mlx",
  "task": "transcribe",
  "word_count": 803,
  "text": "Good morning everyone…",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.64,
      "text": "Good morning everyone, and thanks for joining the weekly engineering sync.",
      "speaker": null,
      "words": [{ "start": 0.0, "end": 0.1, "word": "Good", "probability": 0.9141 }]
    }
  ]
}
```

`speaker` is always `null` today and is where diarization will land. `words` is
populated whenever `word_timestamps` is on (the default).

## Errors

Every failure is `{"detail": "…"}` with a message written for a person, e.g.

```json
{ "detail": "standup.mov has no audio track to transcribe." }
{ "detail": "'.pdf' is not a supported media type. Supported: aac, aif, aiff, …" }
{ "detail": "ffprobe was not found on this Mac. Install it with:\n\n    brew install ffmpeg\n" }
```

The app shows `detail` verbatim, so the backend never emits a stack trace or an
error code as user-facing text.
