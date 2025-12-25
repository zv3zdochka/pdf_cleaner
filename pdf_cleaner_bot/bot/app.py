from __future__ import annotations

import asyncio
import logging

from aiogram import Dispatcher, F
from aiogram.filters import CommandStart

from pdf_cleaner_bot.bot.handlers import cmd_start, handle_document
from pdf_cleaner_bot.storage.manager import StorageConfig, StorageManager


def build_dispatcher(*, processor, shrink_pdf, settings) -> Dispatcher:
    """
    Build and configure aiogram dispatcher with injected dependencies.
    """
    dp = Dispatcher()

    process_lock = asyncio.Lock()

    storage = StorageManager(
        StorageConfig(
            root_dir=settings.storage_dir,
            max_bytes=settings.storage_max_bytes,
            max_age_days=settings.storage_max_age_days,  # keep 0 => TTL disabled
        ),
        logger=logging.getLogger("pdf_cleaner.storage"),
    )

    dp.message.register(cmd_start, CommandStart())

    async def _doc_handler(message):
        # aiogram v3 гарантирует message.bot
        return await handle_document(
            message,
            message.bot,
            processor=processor,
            shrink_pdf=shrink_pdf,
            process_lock=process_lock,
            telegram_max_file_size=settings.telegram_max_file_size,
            internal_max_file_size=settings.internal_max_file_size,
            storage=storage,
        )

    dp.message.register(_doc_handler, F.document)
    return dp
