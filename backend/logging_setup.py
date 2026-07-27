"""Logging for the backend: readable on the console, durable on disk."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

LOG_FILENAME = "whisper-mac.log"
_CONSOLE_FORMAT = "%(asctime)s  %(levelname)-7s %(name)-22s %(message)s"
_FILE_FORMAT = "%(asctime)s  %(levelname)-7s %(name)-22s [%(threadName)s] %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_configured = False


def configure_logging(log_dir: Path, level: str = "INFO") -> Path:
    """Attach console + rotating-file handlers to the root logger.

    Idempotent: calling it twice (uvicorn reloads, tests) will not duplicate
    handlers. Returns the path of the log file so callers can surface it in
    error messages.
    """
    global _configured

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / LOG_FILENAME

    if _configured:
        return log_path

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(console)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=4 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
    root.addHandler(file_handler)

    # These are chatty and add nothing over our own request logs.
    for noisy in ("httpx", "urllib3", "filelock", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True
    logging.getLogger(__name__).debug("Logging to %s", log_path)
    return log_path


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class LoggerWriter:
    """File-like shim that funnels third-party ``print`` output into a logger.

    mlx-whisper prints the detected language straight to stdout; wrapping the
    call in ``contextlib.redirect_stdout(LoggerWriter(...))`` keeps that out of
    the terminal while preserving it in the log file.
    """

    def __init__(self, logger: logging.Logger, level: int = logging.DEBUG) -> None:
        self._logger = logger
        self._level = level
        self._buffer = ""

    def write(self, message: str) -> int:
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._logger.log(self._level, "%s", line.rstrip())
        return len(message)

    def flush(self) -> None:
        if self._buffer.strip():
            self._logger.log(self._level, "%s", self._buffer.strip())
        self._buffer = ""

    def isatty(self) -> bool:
        return False
