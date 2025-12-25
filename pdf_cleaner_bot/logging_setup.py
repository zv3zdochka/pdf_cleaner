from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


class RequestIdFilter(logging.Filter):
    """
    Ensures every log record has request_id attribute for consistent formatting.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True


def setup_logging(
        log_dir: Path,
        level: str = "INFO",
        file_name: str = "bot.log",
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 10,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicated handlers on reloads
    if root.handlers:
        return

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s request_id=%(request_id)s %(message)s"
    )

    request_filter = RequestIdFilter()

    # Console
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.addFilter(request_filter)
    root.addHandler(sh)

    # File with rotation
    fh = RotatingFileHandler(
        filename=str(log_dir / file_name),
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    fh.setFormatter(fmt)
    fh.addFilter(request_filter)
    root.addHandler(fh)

    # Reduce noisy libs a bit if desired
    logging.getLogger("aiogram").setLevel(logging.INFO)
    logging.getLogger("onnxruntime").setLevel(logging.WARNING)
