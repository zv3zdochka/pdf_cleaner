"""Bot application wiring (dispatcher, dependencies, polling)."""

from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart

from pdf_cleaner_bot.config import AppConfig
from pdf_cleaner_bot.processor import PDFRegionProcessor
from pdf_cleaner_bot.bot.handlers import BotHandlers


def build_processor(cfg: AppConfig) -> PDFRegionProcessor:
    """Create and configure the global PDF processor instance."""
    return PDFRegionProcessor(
        model_path=cfg.model.model_path,
        input_size=cfg.model.input_size,
        conf_threshold=cfg.model.conf_threshold,
        render_zoom=cfg.model.render_zoom,
        providers=cfg.model.providers,
    )


async def run_bot(cfg: AppConfig) -> None:
    """Run aiogram polling loop."""
    if not cfg.bot_token:
        raise RuntimeError("BOT_TOKEN is empty. Provide it via environment variables.")

    bot = Bot(cfg.bot_token)
    dp = Dispatcher()

    process_lock = asyncio.Lock()
    processor = build_processor(cfg)
    handlers = BotHandlers(
        processor=processor,
        work_dir=cfg.work_dir,
        telegram_max_file_size=cfg.limits.telegram_max_file_size,
        internal_max_file_size=cfg.limits.internal_max_file_size,
        process_lock=process_lock,
    )

    dp.message.register(handlers.cmd_start, CommandStart())
    dp.message.register(handlers.handle_document, F.document)

    logging.getLogger("Main").info("Bot started. Waiting for updates...")
    await dp.start_polling(bot)
