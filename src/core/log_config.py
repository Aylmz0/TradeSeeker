"""Centralized loguru configuration for TradeSeeker.

Provides structured logging with 5 sinks:
1. Console — human-readable, colored output
2. File — daily rotation, 30 days retention
3. AI Reasoning — full LLM chain-of-thought, daily rotation
4. Structured — machine-readable, 10MB rotation (jq-queryable)
5. Crash — ERROR and above, 90 days retention

Usage:
    from src.core.log_config import setup_logging
    setup_logging()
"""

import logging
import os
import sys

from loguru import logger

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _InterceptHandler(logging.Handler):
    """Route Python stdlib logging (litellm, xgboost, etc.) through loguru."""

    _LEVEL_MAP = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: "WARNING",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "CRITICAL",
    }

    def emit(self, record: logging.LogRecord) -> None:
        """Forward a stdlib logging record to loguru.

        Args:
            record: Standard library log record to intercept and re-emit.
        """
        try:
            level = self._LEVEL_MAP.get(record.levelno, "INFO")
            logger.opt(depth=6).log(level, "{}", record.getMessage())
        except Exception:
            pass


def _ai_reasoning_filter(record) -> bool:
    """Loguru filter that passes only messages bound with ``kind='ai_reasoning'``.

    Args:
        record: Loguru record dict containing an ``extra`` field.

    Returns:
        True if the record should be logged, False to suppress it.
    """
    return record["extra"].get("kind") == "ai_reasoning"


def setup_logging(log_level: str = "INFO", log_dir: str | None = None) -> None:
    """Configure loguru with console, file, and JSON sinks.

    Sets up five sinks: human-readable console output, daily-rotating file logs,
    AI reasoning logs, structured JSON logs, and crash reports. Also routes stdlib
    logging from third-party libraries (litellm, httpx) through loguru.

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

    # 3. AI Reasoning — full LLM chain-of-thought (bind with kind="ai_reasoning")
    logger.add(
        os.path.join(log_dir, "ai_reasoning.log"),
        level="DEBUG",
        rotation="1 day",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{line} | {message}",
        filter=_ai_reasoning_filter,
        compression="gz",
    )

    # 4. Structured — machine-readable, 10MB rotation, gzipped
    logger.add(
        os.path.join(log_dir, "structured.jsonl"),
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD}T{time:HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        compression="gz",
    )

    # 5. Crash reports — ERROR and above only
    logger.add(
        os.path.join(log_dir, "crash_reports.jsonl"),
        level="ERROR",
        rotation="10 MB",
        retention="90 days",
        format="{time:YYYY-MM-DD}T{time:HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        compression="gz",
    )

    # Route stdlib logging (litellm, xgboost, httpx) through loguru
    # Set litellm to ERROR to suppress noisy register_model cost-map warnings
    intercept_handler = _InterceptHandler()
    for name in (
        "litellm",
        "litellm.utils",
        "litellm.router",
        "litellm.proxy",
        "LiteLLM",
        "LiteLLM Proxy",
        "LiteLLM Router",
        "httpx",
        "httpcore",
    ):
        stdlib_logger = logging.getLogger(name)
        stdlib_logger.handlers = [intercept_handler]
        stdlib_logger.setLevel(logging.ERROR)
        stdlib_logger.propagate = False

    logger.info("Logging initialized", level=log_level, log_dir=log_dir)
