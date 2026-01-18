from __future__ import annotations

import asyncio
import logging

from aiogram import Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import ContentType

from pdf_cleaner_bot.bot.handlers import (
    cmd_start,
    handle_callback,
    handle_document,
    handle_pages_text,
)
from pdf_cleaner_bot.storage.manager import StorageConfig, StorageManager
from pdf_cleaner_bot.storage.user_db import UserDatabase
from pdf_cleaner_bot.bot.user_tracker import UserTrackingMiddleware, handle_contact


def build_dispatcher(*, processor, shrink_pdf, settings) -> Dispatcher:
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

    user_db_path = settings.storage_dir / "users.db"
    user_db_logger = logging.getLogger("pdf_cleaner.user_db")
    user_db_logger.info("Initializing user database at: %s", user_db_path)

    user_db = UserDatabase(
        db_path=user_db_path,
        logger=user_db_logger,
    )

    user_tracker_logger = logging.getLogger("pdf_cleaner.user_tracker")
    user_tracker_logger.info("Registering user tracking middleware")

    user_tracker = UserTrackingMiddleware(
        user_db=user_db,
        logger=user_tracker_logger,
    )
    dp.message.middleware(user_tracker)
    dp.callback_query.middleware(user_tracker)

    # Contact handler - register FIRST with content_type filter
    async def _contact_handler(message):
        return await handle_contact(message, user_db)

    dp.message.register(_contact_handler, F.content_type == ContentType.CONTACT)

    # /start
    dp.message.register(cmd_start, CommandStart())

    # PDF document
    async def _doc_handler(message):
        return await handle_document(
            message,
            message.bot,
            telegram_max_file_size=settings.telegram_max_file_size,
            internal_max_file_size=settings.internal_max_file_size,
            storage=storage,
        )

    dp.message.register(_doc_handler, F.document)

    # Inline buttons callbacks
    async def _cb_handler(query):
        return await handle_callback(
            query,
            processor=processor,
            shrink_pdf=shrink_pdf,
            process_lock=process_lock,
            telegram_max_file_size=settings.telegram_max_file_size,
            internal_max_file_size=settings.internal_max_file_size,
            storage=storage,
        )

    dp.callback_query.register(_cb_handler, F.data.startswith("pdfc:"))

    # Text input for pages
    async def _pages_text_handler(message):
        return await handle_pages_text(
            message,
            storage=storage,
        )

    dp.message.register(_pages_text_handler, F.text & ~F.text.startswith("/"))

    return dp