"""Structured logging helpers used across the package.

``get_logger`` is a thin, dependency-free convenience (safe to call anywhere).
``configure_logging`` applies the dictConfig in ``config/logging.yaml`` and is a
stub until the implementation step.
"""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import yaml


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under ``ragforce`` (e.g. ``ragforce.pipeline``)."""
    return logging.getLogger(name if name.startswith("ragforce") else f"ragforce.{name}")


def configure_logging(config_file: str = "config/logging.yaml") -> None:
    """Apply the logging dictConfig from ``config_file`` (idempotent).

    Falls back to a sensible ``basicConfig`` if the file is missing or unreadable, so
    logging never blocks a run.
    """
    path = Path(config_file)
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        logging.config.dictConfig(cfg)
    except (OSError, ValueError, KeyError, TypeError):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
