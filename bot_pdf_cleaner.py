"""Backward-compatible entrypoint.

This file is kept to preserve the existing Docker CMD and user habits.
All real logic lives in the pdf_cleaner_bot package.
"""

from __future__ import annotations

import asyncio

from pdf_cleaner_bot.config import load_config
from pdf_cleaner_bot.logging_setup import setup_logging
from pdf_cleaner_bot.bot.app import run_bot


async def main() -> None:
    setup_logging()
    cfg = load_config()
    await run_bot(cfg)


if __name__ == "__main__":
    asyncio.run(main())
