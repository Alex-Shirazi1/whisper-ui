<h1 align="center">WhisperMac</h1>

<p align="center">
  <strong>Drag a video in. Get a transcript out. Nothing leaves your Mac.</strong>
</p>

<p align="center">
  A native macOS transcription app running NVIDIA <code>parakeet-tdt-0.6b-v3</code>,
  with OpenAI <code>whisper-large-v3</code> behind it, on Apple Silicon through
  <a href="https://github.com/ml-explore/mlx">MLX</a> — Metal GPU acceleration, no
  cloud, no API keys, no terminal.
</p>

<p align="center">
  <a href="../../actions/workflows/test.yml"><img alt="tests" src="../../actions/workflows/test.yml/badge.svg"></a>
  <img alt="platform" src="https://img.shields.io/badge/platform-macOS%2014%2B%20·%20Apple%20Silicon-black">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-blue">
</p>

---

## Why this exists

Local speech recognition is excellent and MLX makes it fast on Apple Silicon —
but using it means remembering a command, a model id, a flag for every output
format, and where you left the file. WhisperMac is that command turned into an app you can
keep in your Dock:

- **Drag and drop** a video or audio file onto the window. That's the whole flow.
- **Genuinely local.** Audio never leaves the machine. The server binds to
  `127.0.0.1`, needs no network after the one-time model download, and holds a
  session token so nothing else on your Mac can read your transcripts.
- **Apple Silicon native.** MLX float16 kernels on the Metal GPU. No CUDA, no
  PyTorch inference, no CPU fallback.
- **Four formats at once** — `.txt`, `.srt`, `.vtt`, `.json` — with word-level
  timings in the JSON.
- **Real progress**, not a spinner: extraction percentage from ffmpeg, decode
  position from the model, an ETA, and a Cancel button that actually stops work.

## Screenshots

> **TODO — add real captures.** Run `./run.sh`, press ⌘⇧4, and save into
> `docs/images/`, then replace this block with the table below it. Worth capturing
> in both appearances (System Settings ▸ Appearance), since light and dark are
> styled deliberately rather than one being derived from the other.
>
> | File | Shot |
> |---|---|
> | `dropzone.png` | Empty state — the drop target with the waveform backdrop |
> | `queue.png` | Two or three files mid-transcription, progress bars live |
> | `transcript.png` | A finished file with the transcript inspector open and a search term entered |
> | `dark.png` | The same window in dark mode |
>
> ```markdown
> |  |  |
> |---|---|
> | **Empty state** <br><img src="docs/images/dropzone.png" width="420"> | **Transcribing** <br><img src="docs/images/queue.png" width="420"> |
> | **Transcript inspector** <br><img src="docs/images/transcript.png" width="420"> | **Dark mode** <br><img src="docs/images/dark.png" width="420"> |
> ```

## Requirements

| | |
|---|---|
| Mac | Apple Silicon (M1 or newer). MLX has no Intel build. |
| macOS | 14 Sonoma or newer |
| Disk | ~5.5 GB of weights (2.5 GB Parakeet + 3 GB Whisper), ~1.5 GB for the Python environment |
| Memory | 16 GB comfortably runs both models; on 8 GB set `WHISPER_MAC_FALLBACK_MODEL=mlx-community/whisper-medium-mlx` |
| Tools | [Homebrew](https://brew.sh), `ffmpeg`, Python 3.11+, Xcode command line tools |

## Install

```bash
git clone https://github.com/Alex-Shirazi1/whisper-ui.git whisper-mac
cd whisper-mac
./scripts/install.sh
```

The installer checks your hardware, installs `ffmpeg` if it's missing, creates
`.venv`, installs the MLX stack, verifies Metal actually works, downloads the
model, transcribes a generated test clip end to end, and builds the app. Every
step is idempotent — re-run it any time something looks broken.

```bash
./scripts/install.sh --help

  --yes           never prompt (CI, unattended re-runs)
  --skip-model    don't download the ~5 GB of weights yet
  --skip-app      backend only
  --slim          remove the unused PyTorch dependency (~2.5 GB)
  --model REPO    use a different Parakeet or MLX Whisper model
```

Then:

```bash
./run.sh
```

The backend starts, the app opens, and quitting either one shuts down both. Drag
a file onto the window.

> `--slim` is worth knowing about: `mlx-whisper` declares `torch` as a
> dependency, but only its weight-conversion module imports it. Inference here is
> 100 % MLX/Metal, so the flag removes it and re-verifies the import — saving
> about 2.5 GB of disk for zero functional change.

## Usage

**Transcribe** — drop files on the window, use **Add Files…** (⌘O), or right-click
a file in Finder ▸ Open With ▸ WhisperMac. With *Start transcribing immediately*
on (the default) work begins on drop; otherwise press **Transcribe** (⌘R).

**Choose formats** — the TXT / SRT / VTT / JSON toggles in the bottom bar. Your
selection is remembered.

**Choose where files land** — click the folder path in the bottom bar. Also
remembered. Defaults to `output/`.

**Read the result** — click a finished row to load the transcript on the right:
searchable, timestamped, click any line to copy it.

**Cancel** — ⌘. or the stop button on the row. Nothing half-written is left
behind.

### Keyboard

| | |
|---|---|
| ⌘O | Add files |
| ⌘R | Transcribe queued files |
| ⌘. | Cancel all |
| ⇧⌘K | Clear finished |
| ⇧⌘C | Copy transcript |
| ⇧⌘O | Open output folder |
| ⇧⌘R | Reconnect to the engine |

### Options menu (in the header)

Auto-start on drop · reveal in Finder when done · language (auto-detect or pick
one of 20) · translate to English · output folder.

## Supported formats

**Video** — `.mp4` `.mov` `.m4v` `.mkv` `.webm` `.avi` `.mpg` `.mpeg` `.wmv` `.flv`

**Audio** — `.m4a` `.wav` `.mp3` `.aac` `.flac` `.ogg` `.oga` `.opus` `.aiff` `.aif` `.wma` `.caf`

Anything ffmpeg can decode, really — audio is extracted to 16 kHz mono before the
model sees it, so the container and codec don't matter.

**Output**

| Format | Contents |
|---|---|
| `.txt` | Plain text, one line per segment |
| `.srt` | SubRip subtitles, wrapped at 42 characters over 2 lines |
| `.vtt` | WebVTT captions with a language header |
| `.json` | Everything: segments, per-word timings and probabilities, language, duration, model |

Existing files are never overwritten — `meeting.txt` becomes `meeting-1.txt`.

## Performance

Measured on a **MacBook Air, Apple M4, 16 GB**, both models through MLX on the
same audio. Word error rate uses Whisper's English text normaliser on 200 random
LibriSpeech utterances per set.

| | Parakeet TDT 0.6B v3 (default) | whisper-large-v3 |
|---|---|---|
| WER, LibriSpeech test-clean | **1.65 %** | 2.82 % |
| WER, LibriSpeech test-other (noisy, accented) | **3.55 %** | 6.72 % |
| 10 min of a real screen recording | **19 s (31× realtime)** | several minutes |

So a one-hour recording takes roughly **2 minutes**. Notes:

- **Auto language costs one Whisper load.** With the language on auto, Whisper
  identifies it from the first 30 seconds (~3 s to load once per session), then
  Parakeet transcribes. Pick a language in Options to skip that.
- **Translation and non-European languages run on Whisper**, at Whisper speed
  (about 5× realtime on an M3 Pro).
- **Word timestamps come free with Parakeet**; on Whisper they cost 15–20 %.

## Configuration

Everything is environment-driven; the defaults are what `run.sh` uses.

| Variable | Default | Purpose |
|---|---|---|
| `WHISPER_MAC_MODEL` | `mlx-community/parakeet-tdt-0.6b-v3` | A Parakeet or MLX Whisper repo id, or a local path |
| `WHISPER_MAC_FALLBACK_MODEL` | `mlx-community/whisper-large-v3-mlx` | Whisper model behind Parakeet (translation, other languages, language ID). Empty disables it |
| `WHISPER_MAC_PORT` | `8756` | Backend port |
| `WHISPER_MAC_OUTPUT_DIR` | `output/` | Default transcript folder |
| `WHISPER_MAC_MODEL_DIR` | `backend/models/` | Weight cache (shared, so downloads happen once) |
| `WHISPER_MAC_TEMPERATURE` | `0.0` | Deterministic decoding |
| `WHISPER_MAC_TEMPERATURE_FALLBACK` | `0` | Enable Whisper's retry ladder — slower, but rescues the rare repetition loop |
| `WHISPER_MAC_WORD_TIMESTAMPS` | `1` | Word-level DTW alignment (Whisper; Parakeet always times words) |
| `WHISPER_MAC_ESTIMATED_RTF` | `0.19` | Calibration for progress interpolation |
| `WHISPER_MAC_MAX_UPLOAD_BYTES` | 8 GB | Cap for the multipart upload route |
| `WHISPER_MAC_REQUIRE_TOKEN` | `1` | Session-token auth |
| `WHISPER_MAC_KEEP_TEMP` | `0` | Keep extracted WAVs for debugging |
| `WHISPER_MAC_LOG_LEVEL` | `INFO` | `DEBUG` for the full picture |

## How it works

Two processes over loopback HTTP: a SwiftUI app, and a FastAPI backend that owns
the MLX model.

```
drop file → POST /transcribe/local (path, not bytes)
          → ffprobe          duration, stream layout
          → ffmpeg           16 kHz mono WAV, live progress
          → Parakeet         Metal decode, progress per 2-minute chunk
            (or mlx-whisper  for translation / other languages, per 30 s window)
          → writers          txt / srt / vtt / json
          → GET /jobs        polled twice a second by the app
```

Two processes because MLX has no Swift API, and because a bad decode should never
take the window down with it. The app can start, monitor and stop the backend
itself, so you never see the seam.

The interesting parts — how progress and cancellation are extracted from a
library that offers neither, why state is immutable, and exactly where each
roadmap feature attaches — are in **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.
The HTTP contract is in **[docs/API.md](docs/API.md)**.

## Development

```bash
./scripts/setup_python.sh --dev     # venv + pytest/ruff
.venv/bin/python -m pytest          # 107 tests, ~5 s, no model download needed
.venv/bin/python -m ruff check .
./scripts/build.sh --debug          # rebuild the app
./run.sh --server-only              # backend only; interactive docs at /docs
```

The test suite runs against a `StubEngine` rather than MLX, so it's fast, needs
no weights and no network, and runs on Linux in CI. A separate macOS job installs
the real MLX stack and asserts that the assumptions the progress bridge makes
about `mlx-whisper` still hold — that's the thing most likely to break on a
dependency bump.

```
backend/           config · media · formats · transcriber · jobs · whisper_server · auth
backend/tests/     107 tests · 92% coverage: formats, media, jobs, API, engine wiring
app/WhisperMac/    SwiftUI app (SwiftPM — open app/Package.swift in Xcode if you prefer)
scripts/           install · setup_python · build · run_backend · make_icon
```

## Troubleshooting

**"Engine offline" in the app.**
Click **Start Engine** in the banner, or run `./run.sh` from the repo. If it still
fails, look at `logs/whisper-mac.log` and `logs/backend.out`. Starting the app by
double-clicking works because `scripts/build.sh` stamps the repo path into the
bundle — if you moved the `.app` elsewhere, re-run `./scripts/build.sh`.

**"ffmpeg is not installed."**
`brew install ffmpeg`. A GUI-launched app inherits a minimal `PATH`, so
WhisperMac also looks in `/opt/homebrew/bin` and `/usr/local/bin` directly.

**"MLX engine unavailable."**
`./scripts/setup_python.sh --recreate`. This also appears on an Intel Mac, which
MLX doesn't support.

**The model downloads every time.**
It shouldn't — weights are cached in `backend/models/`. If `WHISPER_MAC_MODEL_DIR`
or `HF_HOME` differ between runs you'll end up with two caches. `run.sh` pins it
for you.

**First transcription is slow, later ones are fast.**
Expected: weights load once per backend session (~3 s), and numba compiles the
word-alignment kernel on first use.

**Repeated or looping text in the output.**
Rare, and usually caused by long silences. Try
`WHISPER_MAC_TEMPERATURE_FALLBACK=1`, which lets Whisper retry a bad window at a
higher temperature.

**"No audio track to transcribe."**
The file is video-only or silent. Check with `ffprobe -show_streams yourfile.mp4`.

**macOS won't open the app** ("unidentified developer").
It's ad-hoc signed, not notarised, because it's built locally. Right-click ▸
**Open** once, or `xattr -dr com.apple.quarantine build/WhisperMac.app`.

**Out of memory.**
Use a smaller fallback: `WHISPER_MAC_FALLBACK_MODEL=mlx-community/whisper-medium-mlx ./run.sh`.
Picking a Parakeet language in Options (rather than auto) also keeps Whisper out
of memory entirely.

## Roadmap

The architecture leaves room for these, and
[ARCHITECTURE.md](docs/ARCHITECTURE.md) says exactly where each one plugs in:

- [ ] **Speaker diarization** — `Segment.speaker` already flows through the model,
      the writers and the UI; it needs a labelling post-processor
- [ ] **Batch / folder drops** — the queue already handles N files serially
- [ ] **Local LLM summaries** via `mlx-lm`
- [ ] **Cross-transcript search** over the emitted JSON (SQLite FTS5)
- [ ] **Timestamp navigation** with an inline player
- [ ] **Subtitle editing** — word timings are already persisted

## License

MIT — see [LICENSE](LICENSE).

Built on [MLX](https://github.com/ml-explore/mlx) and
[mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) by
Apple, [parakeet-mlx](https://github.com/senstella/parakeet-mlx),
[Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) by NVIDIA, and
[Whisper](https://github.com/openai/whisper) by OpenAI.
