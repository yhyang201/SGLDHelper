"""Structured logging setup using structlog."""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

import structlog


def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> None:
    """Configure structlog with JSON output to stderr and rotating log files."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "sgldhelper.log")

    # Rotating file handler: one file per day, keep 30 days
    file_handler = TimedRotatingFileHandler(
        log_file, when="midnight", backupCount=30, encoding="utf-8"
    )
    file_handler.setLevel(level)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(level)

    # Standard logging config (captures logs from third-party libs too)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stderr_handler)

    # Enable slack_bolt debug logging when at DEBUG level
    if level <= logging.DEBUG:
        for name in ("slack_bolt", "slack_sdk"):
            lib_logger = logging.getLogger(name)
            lib_logger.setLevel(logging.DEBUG)

    # structlog: JSON to both stderr and file via stdlib integration
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            # Route through stdlib so file_handler also gets the output
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Formatter for stdlib handlers — renders structlog events as JSON
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
    )
    file_handler.setFormatter(formatter)
    stderr_handler.setFormatter(formatter)
