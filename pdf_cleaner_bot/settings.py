from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


@dataclass(frozen=True)
class Settings:
    bot_token: str
    model_path: Path
    work_dir: Path
    storage_dir: Path
    logs_dir: Path

    telegram_max_file_size: int
    internal_max_file_size: int

    storage_max_bytes: int
    storage_max_age_days: int

    log_level: str

    @staticmethod
    def from_env() -> "Settings":
        bot_token = os.getenv("BOT_TOKEN", "").strip()

        model_path = Path(os.getenv("MODEL_PATH", "weights.onnx"))
        work_dir = Path(os.getenv("WORK_DIR", "data"))
        storage_dir = Path(os.getenv("STORAGE_DIR", "storage"))
        logs_dir = Path(os.getenv("LOGS_DIR", "logs"))

        telegram_max = _env_int("TELEGRAM_MAX_FILE_SIZE", 50 * 1024 * 1024)
        internal_max = _env_int("INTERNAL_MAX_FILE_SIZE", 1 * 1024 * 1024 * 1024)

        storage_max_bytes = _env_int("STORAGE_MAX_BYTES", 5 * 1024 * 1024 * 1024)  # 5 GiB
        storage_max_age_days = _env_int("STORAGE_MAX_AGE_DAYS", 14)

        log_level = os.getenv("LOG_LEVEL", "INFO").strip()

        return Settings(
            bot_token=bot_token,
            model_path=model_path,
            work_dir=work_dir,
            storage_dir=storage_dir,
            logs_dir=logs_dir,
            telegram_max_file_size=telegram_max,
            internal_max_file_size=internal_max,
            storage_max_bytes=storage_max_bytes,
            storage_max_age_days=storage_max_age_days,
            log_level=log_level,
        )
