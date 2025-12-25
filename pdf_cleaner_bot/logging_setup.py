"""Logging setup utilities."""

from __future__ import annotations

import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Configure Python logging.

    The bot is typically run under Docker with stdout/stderr captured. A single
    basicConfig is sufficient and keeps logs structured.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
