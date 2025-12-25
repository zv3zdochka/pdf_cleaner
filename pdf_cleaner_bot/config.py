"""Application configuration.

Notes
-----
- All filesystem paths are resolved relative to the working directory inside the container.
- The bot token MUST be provided via the BOT_TOKEN environment variable in production.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """ONNX detector configuration."""
    model_path: Path
    input_size: Tuple[int, int] = (736, 736)
    conf_threshold: float = 0.5
    render_zoom: float = 2.0
    providers: List[str] = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class LimitsConfig:
    """File size limits."""
    telegram_max_file_size: int = 50 * 1024 * 1024           # ~50 MiB
    internal_max_file_size: int = 1 * 1024 * 1024 * 1024     # 1 GiB


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Top-level application configuration."""
    bot_token: str
    work_dir: Path
    model: ModelConfig
    limits: LimitsConfig = LimitsConfig()


def load_config() -> AppConfig:
    """Load configuration from environment variables (with safe defaults)."""
    # Security: do not hardcode tokens in the source code.
    # The user can set BOT_TOKEN in docker run / compose / systemd environment.
    bot_token = os.getenv("BOT_TOKEN", "").strip()

    # Filesystem
    work_dir = Path(os.getenv("WORK_DIR", "data"))
    work_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(os.getenv("MODEL_PATH", "weights.onnx"))

    providers_env = os.getenv("ONNX_PROVIDERS", "").strip()
    if providers_env:
        providers = [p.strip() for p in providers_env.split(",") if p.strip()]
    else:
        providers = ["CPUExecutionProvider"]

    model = ModelConfig(
        model_path=model_path,
        input_size=(736, 736),
        conf_threshold=float(os.getenv("CONF_THRESHOLD", "0.5")),
        render_zoom=float(os.getenv("RENDER_ZOOM", "2.0")),
        providers=providers,
    )

    limits = LimitsConfig(
        telegram_max_file_size=int(os.getenv("TELEGRAM_MAX_FILE_SIZE", str(50 * 1024 * 1024))),
        internal_max_file_size=int(os.getenv("INTERNAL_MAX_FILE_SIZE", str(1 * 1024 * 1024 * 1024))),
    )

    return AppConfig(
        bot_token=bot_token,
        work_dir=work_dir,
        model=model,
        limits=limits,
    )
