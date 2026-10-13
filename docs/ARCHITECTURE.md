# Architecture

WhisperMac is two processes that talk over loopback HTTP:

```
┌──────────────────────────────┐        ┌────────────────────────────────────┐
│  WhisperMac.app (SwiftUI)    │        │  backend (FastAPI + MLX engines)   │
│                              │        │                                    │
│  ContentView                 │        │  whisper_server.py   routes        │
│   ├─ HeaderBar               │  HTTP  │  jobs.py             pipeline      │
│   ├─ DropZone / QueueView    │ ─────► │  transcriber.py      MLX / Metal   │
│   ├─ TranscriptInspector     │  :8756 │  media.py            ffmpeg        │
│   └─ ControlBar              │ ◄───── │  formats.py          txt/srt/vtt   │
│                              │  JSON  │  auth.py             session token │
│  AppModel  (single source    │        │                                    │
│  of truth, @MainActor)       │        │  ~/.whisper-mac/session.json       │
└──────────────────────────────┘        └────────────────────────────────────┘
```

Two processes rather than one because the transcription stack is Python (MLX has
no Swift API) and because a crash in a decode loop should never take the window
with it. The app can start, monitor and stop the backend itself, so the split is
invisible in normal use.

## Request flow

1. A file is dropped. `AppModel.add(urls:)` validates the extension against the
   list `/config` advertised and appends a `QueueItem`.
2. `AppModel.submit(_:)` posts `POST /transcribe/local` with the file's **path**.
   Nothing is copied — a 4 GB screen recording never moves.
3. `JobStore.submit` returns a job id immediately and queues the work on a
   single-worker thread pool. One decode at a time keeps the GPU and the weight
   cache uncontended.
4. The pipeline walks its stages, each owning a slice of the progress bar:

   | Stage               | Range       | What happens |
   |---------------------|-------------|--------------|
   | `inspecting`        | 0.00 – 0.02 | `ffprobe` for duration and stream layout |
   | `extracting_audio`  | 0.02 – 0.10 | `ffmpeg` → 16 kHz mono PCM, real progress from `-progress` |
   | `downloading_model` | 0.10 – 0.13 | first run only; indeterminate |
   | `loading_model`     | 0.13 – 0.17 | weights into unified memory; indeterminate |
   | `transcribing`      | 0.17 – 0.97 | MLX decode; one report per 2 min chunk (Parakeet) or 30 s window (Whisper) |
   | `writing_output`    | 0.97 – 1.00 | render and write every requested format |

5. The app polls `GET /jobs` twice a second while anything is running, and
   `GET /jobs/{id}/transcript` once a file finishes.

## Design decisions worth knowing

**Parakeet by default, Whisper behind it.** The default model is NVIDIA
Parakeet TDT 0.6B v3 (`backend/parakeet.py`, via `parakeet-mlx`): far faster
than whisper-large-v3 and more accurate on English, but it covers 25 European
languages and cannot translate or detect a language. `transcriber.resolve_route`
keeps a job on Parakeet when it can do it and otherwise hands it to the
fallback Whisper model: translation, a language Parakeet lacks, or — with the
language on auto — whatever Whisper's own language ID on the first 30 seconds
says is not a Parakeet language. Parakeet decodes in 2-minute chunks with 15 s
of overlap; each boundary is a progress report and a cancellation check, and
its sub-word tokens are regrouped into Whisper-shaped words and caption-sized
segments so the writers and the app never know which engine ran.

**Progress and cancellation come from a `tqdm` shim.** mlx-whisper offers
neither. It calls `tqdm.tqdm(total=content_frames)` and `pbar.update(...)` once
per decode window, so `transcriber._progress_bridge` temporarily swaps that
module attribute for `_ProgressBar`, which forwards fractions to the job and
raises `TranscriptionCancelled` from inside `update()` when the user cancels.
The shim handles both `import tqdm` and `from tqdm import tqdm` shapes and
degrades to no live progress rather than failing if a future release drops the
bar entirely.

**Progress is monotonic and interpolated.** A 30-second reporting granularity
makes a bar look frozen on long files, so `Job.blend_estimate` advances it from
elapsed time against a realtime-factor estimate, capped at 95 % of the stage and
never allowed to move backwards or overtake a real report.

**Immutable state.** `JobState` is a frozen dataclass. Mutation means building a
replacement under a lock (`Job.update`), so a poll can never observe a
half-written record. Same for `Settings`.

**The engine is behind a protocol.** `transcriber.Engine` declares the four
things the pipeline needs (`model_is_local`, `model_is_loaded`, `ensure_model`,
`transcribe`). `MLXEngine` is the real one; the test suite passes a `StubEngine`,
which is why the whole suite runs in CI on Linux with no weights and no network.

**Kwargs are filtered by signature.** `_supported_kwargs` inspects
`mlx_whisper.transcribe` and drops anything the installed version does not
accept, because unknown keys otherwise fall through to `DecodingOptions` and
raise. That keeps the project working across mlx-whisper releases instead of
pinning to one.

**Loopback is not the same as private.** Any local process could otherwise POST
a path and read the transcript back, so the server writes a token to
`~/.whisper-mac/session.json` (mode 0600) and requires it on every route except
`/health`. The app reads the same file, so the user never sees it.

## Where the roadmap features attach

The seams already exist; none of these require reshaping the pipeline.

| Feature | Where it goes |
|---|---|
| **Speaker diarization** | `formats.Segment.speaker` is already carried through the model, the JSON writer, the SRT/VTT renderers (`display_text`) and `SegmentDTO`. Add a post-processor between `_run_engine` and `_write` in `jobs.py` that assigns labels, e.g. from `pyannote` or a lightweight embedding clusterer over the extracted WAV. Nothing downstream changes. |
| **Multiple speakers in the UI** | `SegmentRow` already renders a speaker chip when the field is non-nil. |
| **Batch transcription** | The queue is already a list and `JobStore` already serialises N jobs. Raise `max_workers` only if you want concurrent decodes (you probably don't — one model, one GPU). Folder drops just need a recursive expand in `AppModel.add(urls:)`. |
| **Local LLM summaries** | Add `POST /jobs/{id}/summarise` that reads `state.transcript` and calls an MLX LLM (`mlx-lm`). The transcript is retained in the job for `job_retention_seconds`, so no re-read is needed. Surface it as a tab in `TranscriptInspector`. |
| **Search across transcripts** | `TranscriptInspector` already filters the loaded transcript. For cross-file search, index `output/*.json` (they contain full segment text plus timings) into SQLite FTS5 and add a `GET /search` route. |
| **Timestamp navigation** | Segments carry `start`/`end`, and word-level timings are stored when `word_timestamps` is on (the default). Add an `AVPlayer` beside the inspector and seek to `segment.start` on click — the row already handles the click. |
| **Subtitle editing** | `formats.render_srt` is pure (`Transcript → String`), and `Transcript.with_segments` returns a modified copy. Edit segments in the app, `PUT` them back, re-render. Word timings make split/merge operations possible without re-running the model. |

## Module map

| File | Responsibility | Lines |
|---|---|---|
| `backend/config.py` | Frozen `Settings`, env overrides, HF cache location | ~230 |
| `backend/media.py` | ffmpeg/ffprobe discovery, probing, extraction, WAV → float32 | ~300 |
| `backend/formats.py` | `Transcript`/`Segment`/`Word`, four renderers, file naming | ~360 |
| `backend/transcriber.py` | MLX engines, routing, progress bridge, model cache, `Engine` protocol | ~650 |
| `backend/parakeet.py` | Parakeet adapter: chunked decode, tokens → words → segments | ~260 |
| `backend/jobs.py` | Stages, progress mapping, cancellation, cleanup | ~460 |
| `backend/whisper_server.py` | FastAPI routes, uploads, validation | ~430 |
| `backend/auth.py` | Session token and handshake file | ~100 |
| `app/WhisperMac/ViewModels/AppModel.swift` | All app state and intents | ~430 |
| `app/WhisperMac/Services/` | HTTP client, session discovery, backend launcher | ~380 |
| `app/WhisperMac/Views/` | Nine focused views | ~1200 |
| `app/WhisperMac/Theme/Theme.swift` | Dynamic palette, type scale, metrics | ~180 |
