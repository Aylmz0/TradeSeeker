"""Centralized loguru configuration for TradeSeeker.

Provides structured logging with 4 sinks:
1. Console — human-readable, colored output
2. File — daily rotation, 30 days retention
3. JSON — structured, machine-readable, 10MB rotation
4. Crash — ERROR and above, 90 days retention

Usage:
    from src.core.log_config import setup_logging
    setup_logging()
"""

import os
import sys

from loguru import logger

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_logging(log_level: str = "INFO", log_dir: str | None = None) -> None:
    """Configure loguru with console, file, and JSON sinks.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Directory for log files. Defaults to PROJECT_ROOT/data/logs.
    """
    if log_dir is None:
        log_dir = os.path.join(PROJECT_ROOT, "data", "logs")

    os.makedirs(log_dir, exist_ok=True)

    # Remove default handler
    logger.remove()

    # 1. Console — human-readable, colored
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "{message}"
        ),
        colorize=True,
    )

    # 2. File — daily rotation, 30 days retention, gzipped
    logger.add(
        os.path.join(log_dir, "{time:YYYY-MM-DD}.log"),
        level="DEBUG",
        rotation="1 day",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        compression="gz",
    )

    # 3. JSON — structured, machine-readable
    logger.add(
        os.path.join(log_dir, "structured.jsonl"),
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        format="{time:ISO}",
        serialize=True,
    )

    # 4. Crash reports — ERROR and above only
    logger.add(
        os.path.join(log_dir, "crash_reports.jsonl"),
        level="ERROR",
        rotation="10 MB",
        retention="90 days",
        format="{time:ISO}",
        serialize=True,
    )

    logger.info("Logging initialized", level=log_level, log_dir=log_dir)
