"""
MarketShield — Centralised logging configuration.

Usage in any module:
    from logger import get_logger
    log = get_logger(__name__)
    log.info("Fetching RSS feeds...")

Built on loguru for structured, coloured, level-filtered output.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


# ---------------------------------------------------------------------------
# Log file location
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).parent / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def configure_logging(level: str = "INFO", log_to_file: bool = True) -> None:
    """
    Configure loguru sinks.

    Call once at application startup (main.py / dashboard entry point).
    Subsequent ``get_logger()`` calls reuse the same configuration.
    """
    logger.remove()  # Remove default sink

    # ── Console sink ──────────────────────────────────────────────────────
    logger.add(
        sys.stderr,
        level=level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        enqueue=True,  # Thread-safe
    )

    # ── Rotating file sink ────────────────────────────────────────────────
    if log_to_file:
        logger.add(
            LOG_DIR / "marketshield_{time:YYYY-MM-DD}.log",
            level="DEBUG",
            rotation="00:00",       # New file at midnight
            retention="14 days",    # Keep two weeks
            compression="zip",
            enqueue=True,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} — {message}",
        )


def get_logger(name: str):
    """
    Return a loguru logger bound to *name* (typically ``__name__``).

    Example
    -------
    >>> log = get_logger(__name__)
    >>> log.info("Starting data ingestion pipeline")
    """
    return logger.bind(name=name)


# ---------------------------------------------------------------------------
# Default configuration — INFO to stderr, no file (overridden by main.py)
# ---------------------------------------------------------------------------
configure_logging(level="INFO", log_to_file=False)
