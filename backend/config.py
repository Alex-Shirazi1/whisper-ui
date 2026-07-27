"""Runtime configuration for the WhisperMac backend.

Every value can be overridden with a ``WHISPER_MAC_*`` environment variable so
the same code runs from ``run.sh``, from a test suite, or from a launchd job
without edits. :class:`Settings` is frozen — call :func:`dataclasses.replace`
to derive a variant instead of mutating the shared instance.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = Path(__file__).resolve().parent

DEFAULT_MODEL = "mlx-community/whisper-large-v3-mlx"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8756

#: Whisper's native sample rate. Audio is always normalised to this.
SAMPLE_RATE = 16_000

VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".mpg", ".mpeg", ".wmv", ".flv"}
)
AUDIO_EXTENSIONS = frozenset(
    {".m4a", ".wav", ".mp3", ".aac", ".flac", ".ogg", ".oga", ".opus", ".aiff", ".aif", ".wma", ".caf"}
)
SUPPORTED_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

#: Formats the writer layer understands (see :mod:`backend.formats`).
OUTPUT_FORMATS = ("txt", "srt", "vtt", "json")

#: Shared handshake file. The server writes it on boot; the macOS app reads it
#: to discover the port and session token without any user configuration.
STATE_DIR = Path(os.environ.get("WHISPER_MAC_STATE_DIR", str(Path.home() / ".whisper-mac")))
SESSION_FILE = STATE_DIR / "session.json"

_GIB = 1024**3


# --------------------------------------------------------------------------
# Env helpers
# --------------------------------------------------------------------------


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser().resolve() if value else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of backend configuration."""

    # --- server ---
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    log_level: str = "INFO"

    # --- filesystem ---
    output_dir: Path = REPO_ROOT / "output"
    model_cache_dir: Path = BACKEND_ROOT / "models"
    temp_dir: Path = BACKEND_ROOT / ".tmp"
    log_dir: Path = REPO_ROOT / "logs"

    # --- model / decoding ---
    model: str = DEFAULT_MODEL
    #: 0.0 is deterministic and what the CLI docs recommend for accuracy.
    temperature: float = 0.0
    #: When true, fall back up the temperature ladder on a failed window. Costs
    #: time but rescues the occasional repetition loop.
    temperature_fallback: bool = False
    #: Word-level DTW alignment. Slower, but yields tight subtitle timings and
    #: the word data that future subtitle editing will need.
    word_timestamps: bool = True
    condition_on_previous_text: bool = True
    #: Only applies when ``word_timestamps`` is on; suppresses hallucinated
    #: speech inside long silences.
    hallucination_silence_threshold: float | None = 2.0
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6

    # --- behaviour ---
    #: Fraction of realtime the model needs, used to interpolate progress between
    #: decode windows. Measured at ~0.19 for whisper-large-v3 on an M3 with word
    #: timestamps on (≈5.4× realtime); lower it for a smaller model or a faster chip.
    estimated_realtime_factor: float = 0.19
    max_upload_bytes: int = 8 * _GIB
    #: Finished jobs are kept this long so the UI can still read results.
    job_retention_seconds: int = 6 * 60 * 60
    keep_temp_files: bool = False

    # --- auth ---
    require_token: bool = True
    token: str = ""

    # ---------------------------------------------------------------- build

    @classmethod
    def from_env(cls) -> Settings:
        defaults = cls()
        settings = replace(
            defaults,
            host=_env_str("WHISPER_MAC_HOST", defaults.host),
            port=_env_int("WHISPER_MAC_PORT", defaults.port),
            log_level=_env_str("WHISPER_MAC_LOG_LEVEL", defaults.log_level).upper(),
            output_dir=_env_path("WHISPER_MAC_OUTPUT_DIR", defaults.output_dir),
            model_cache_dir=_env_path("WHISPER_MAC_MODEL_DIR", defaults.model_cache_dir),
            temp_dir=_env_path("WHISPER_MAC_TEMP_DIR", defaults.temp_dir),
            log_dir=_env_path("WHISPER_MAC_LOG_DIR", defaults.log_dir),
            model=_env_str("WHISPER_MAC_MODEL", defaults.model),
            temperature=_env_float("WHISPER_MAC_TEMPERATURE", defaults.temperature),
            temperature_fallback=_env_bool(
                "WHISPER_MAC_TEMPERATURE_FALLBACK", defaults.temperature_fallback
            ),
            word_timestamps=_env_bool("WHISPER_MAC_WORD_TIMESTAMPS", defaults.word_timestamps),
            condition_on_previous_text=_env_bool(
                "WHISPER_MAC_CONDITION_ON_PREVIOUS_TEXT", defaults.condition_on_previous_text
            ),
            estimated_realtime_factor=_env_float(
                "WHISPER_MAC_ESTIMATED_RTF", defaults.estimated_realtime_factor
            ),
            max_upload_bytes=_env_int("WHISPER_MAC_MAX_UPLOAD_BYTES", defaults.max_upload_bytes),
            job_retention_seconds=_env_int(
                "WHISPER_MAC_JOB_RETENTION_SECONDS", defaults.job_retention_seconds
            ),
            keep_temp_files=_env_bool("WHISPER_MAC_KEEP_TEMP", defaults.keep_temp_files),
            require_token=_env_bool("WHISPER_MAC_REQUIRE_TOKEN", defaults.require_token),
            token=_env_str("WHISPER_MAC_TOKEN", ""),
        )
        if settings.require_token and not settings.token:
            settings = replace(settings, token=secrets.token_urlsafe(32))
        return settings

    # ---------------------------------------------------------------- misc

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def temperatures(self) -> float | tuple[float, ...]:
        """Value handed to ``mlx_whisper.transcribe(temperature=...)``."""
        if not self.temperature_fallback:
            return self.temperature
        ladder = tuple(round(self.temperature + 0.2 * i, 1) for i in range(6))
        return tuple(t for t in ladder if t <= 1.0)

    def ensure_dirs(self) -> None:
        for directory in (
            self.output_dir,
            self.model_cache_dir,
            self.temp_dir,
            self.log_dir,
            STATE_DIR,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        STATE_DIR.chmod(0o700)

    def apply_model_cache_env(self) -> None:
        """Point Hugging Face at ``backend/models`` before mlx-whisper loads.

        Keeping the cache inside the repo means the 3 GB download happens once
        and is reused by every entry point — server, CLI, tests. HF's cache
        layout dedupes by revision hash, so re-running never re-downloads.
        """
        hub = self.model_cache_dir / "hub"
        hub.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(self.model_cache_dir))
        os.environ["HF_HUB_CACHE"] = str(hub)
        # Never let a stray offline flag turn a first run into a crash loop.
        os.environ.pop("HF_HUB_OFFLINE", None)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings.from_env()


def reset_settings_cache() -> None:
    """Drop the singleton. Used by tests that patch the environment."""
    get_settings.cache_clear()


def is_video(path: Path | str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def is_supported(path: Path | str) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS
