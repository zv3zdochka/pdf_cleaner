import asyncio
import logging

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart

from pdf_cleaner_bot.storage.manager import StorageManager, StorageConfig
from pdf_cleaner_bot.bot.handlers import cmd_start, handle_document


def build_dispatcher(
    *,
    processor,
    shrink_pdf,
    settings,
) -> Dispatcher:
    dp = Dispatcher()

    process_lock = asyncio.Lock()

    storage = StorageManager(
        StorageConfig(
            root_dir=settings.storage_dir,
            max_bytes=settings.storage_max_bytes,
            max_age_days=settings.storage_max_age_days,
        ),
        logger=logging.getLogger("pdf_cleaner.storage"),
    )

    dp.message.register(cmd_start, CommandStart())

    # lambda-обвязка, чтобы пробросить зависимости и не менять сигнатуры aiogram
    dp.message.register(
        lambda m, b: handle_document(
            m,
            b,
            processor=processor,
            shrink_pdf=shrink_pdf,
            process_lock=process_lock,
            telegram_max_file_size=settings.telegram_max_file_size,
            internal_max_file_size=settings.internal_max_file_size,
            storage=storage,
        ),
        F.document,
    )
    return dp
