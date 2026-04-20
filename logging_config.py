"""
logging_config.py
=================
Centralized logging setup for the LangGraph financial agent.

Configures two handlers:
  Console — INFO and above, colored by level
  File    — DEBUG and above, written to output/agent.log

Usage (call once at startup in main.py):
    from logging_config import setup_logging
    setup_logging()

Then in each module:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("message")
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from config.settings import settings


# ── ANSI color codes for console output ──────────────────────
_COLORS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    """Adds ANSI color to the levelname in console output."""

    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname:<8}{_RESET}"
        return super().format(record)


def setup_logging(log_filename: str = "agent.log") -> None:
    """
    Configure root logger with console + rotating file handlers.
    Call once at the top of main.py before anything else runs.

    Args:
        log_filename: name of the log file inside OUTPUT_DIR
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # capture everything; handlers filter by level

    # ── Console handler — INFO and above ──────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(_ColorFormatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(console)

    # ── File handler — DEBUG and above ────────────────────────
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(settings.OUTPUT_DIR, log_filename)
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,   # 5 MB per file
        backupCount=3,               # keep last 3 rotated files
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(file_handler)

    # ── Suppress noisy third-party loggers ───────────────────
    for noisy in ("httpx", "httpcore", "anthropic", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root.info(f"Logging initialized — file: {log_path}")
