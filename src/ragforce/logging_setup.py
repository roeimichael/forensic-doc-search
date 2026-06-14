"""Structured logging helpers used across the package.

``get_logger`` is a thin, dependency-free convenience (safe to call anywhere).
``configure_logging`` applies the dictConfig in ``config/logging.yaml`` and is a
stub until the implementation step.
"""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under ``ragforce`` (e.g. ``ragforce.pipeline``)."""
    return logging.getLogger(name if name.startswith("ragforce") else f"ragforce.{name}")


def configure_logging(config_file: str = "config/logging.yaml") -> None:
    """Apply the logging dictConfig from ``config_file``.

    TODO(X5): read the YAML and call ``logging.config.dictConfig(...)``; fall back
    to ``logging.basicConfig(level=INFO)`` if the file is missing.
    """
    raise NotImplementedError("logging configuration is implemented in the next step (X5)")
