"""Telegram message handlers."""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Optional

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import FSInputFile, Message

from pdf_cleaner_bot.processor import PDFRegionProcessor, shrink_pdf


class BotHandlers:
    """A thin stateful wrapper around handlers.

    This object carries shared dependencies (processor, limits, work_dir, lock),
    making the handlers easy to test and easier to extend later.
    """

    def __init__(
            self,
            processor: PDFRegionProcessor,
            work_dir: Path,
            telegram_max_file_size: int,
            internal_max_file_size: int,
            process_lock: asyncio.Lock,
    ) -> None:
        self.processor = processor
        self.work_dir = work_dir
        self.telegram_max_file_size = telegram_max_file_size
        self.internal_max_file_size = internal_max_file_size
        self.process_lock = process_lock
        self.logger = logging.getLogger(self.__class__.__name__)

    async def cmd_start(self, message: Message) -> None:
        """/start handler: greeting and short instructions."""
        await message.answer(
            "Привет! Пришлите мне PDF-файл (до 50 МБ — это ограничение Telegram), "
            "я автоматически найду и очищу нужные зоны, "
            "а затем пришлю вам обработанный PDF с тем же именем файла."
        )

    async def handle_document(self, message: Message, bot: Bot) -> None:
        """Handle incoming documents. Only PDF files are processed.

        Flow
        ----
        1) Validate file and size limits
        2) Download file to disk (UUID-based name)
        3) Run PDFRegionProcessor (under process_lock)
        4) Compress the result
        5) Send processed PDF back with the original filename
        """
        document = message.document
        if not document:
            return

        original_filename = document.file_name or "document.pdf"
        file_size = document.file_size or 0

        # 1) Hard Telegram limit
        if file_size > self.telegram_max_file_size:
            await message.reply(
                "Этот файл слишком большой для Telegram-бота. "
                "Ограничение Telegram для ботов — примерно 50 МБ на файл.\n\n"
                "Попробуйте отправить более компактный PDF "
                "или предварительно сжать его."
            )
            return

        # 2) Internal safety limit
        if file_size > self.internal_max_file_size:
            await message.reply("Файл слишком большой для обработки (внутренний лимит 1 ГБ).")
            return

        if not original_filename.lower().endswith(".pdf"):
            await message.reply("Пожалуйста, пришлите PDF-файл.")
            return

        await message.reply(
            "Файл получен, начинаю обработку. На больших документах это может занять время..."
        )

        unique_id = uuid.uuid4().hex
        input_path = self.work_dir / f"{message.from_user.id}_{unique_id}.pdf"

        # Download file to local storage (streaming)
        try:
            tg_file = await bot.get_file(document.file_id)
        except TelegramBadRequest as e:
            self.logger.exception("TelegramBadRequest on get_file: %s", e)
            await message.reply(
                "Telegram отказался отдавать файл (скорее всего, он слишком большой)."
            )
            return

        await bot.download_file(tg_file.file_path, destination=input_path)
        self.logger.info("Downloaded file to %s (original name: %s)", input_path, original_filename)

        cleaned_path: Optional[Path] = None
        final_path: Optional[Path] = None

        try:
            async with self.process_lock:
                # Heavy work in a separate thread
                cleaned_path = await asyncio.to_thread(
                    self.processor.process_pdf,
                    pdf_path=input_path,
                    output_path=None,
                )

                final_path = cleaned_path.parent / f"{cleaned_path.stem}_small.pdf"
                await asyncio.to_thread(shrink_pdf, cleaned_path, final_path)

            self.logger.info("Processed file saved to %s", final_path)

            # Telegram output size check
            try:
                result_size = final_path.stat().st_size
            except FileNotFoundError:
                result_size = 0

            if result_size > self.telegram_max_file_size:
                await message.reply(
                    "Обработанный PDF получился больше лимита Telegram (≈50 МБ), "
                    "поэтому я не могу отправить его обратно через бот.\n\n"
                    "Файл сохранён на сервере, но для такого размера нужно использовать "
                    "другой канал передачи (например, облако и ссылку)."
                )
                return

            await message.reply_document(
                FSInputFile(path=str(final_path), filename=original_filename),
                caption="Готово! Вот ваш обработанный PDF.",
            )

        except Exception as e:
            self.logger.exception("Error while processing PDF: %s", e)
            await message.reply("Произошла ошибка при обработке PDF. Проверьте лог сервера.")
        finally:
            # Best-effort cleanup
            for path in (input_path, cleaned_path, final_path):
                try:
                    if isinstance(path, Path):
                        path.unlink(missing_ok=True)
                except Exception as cleanup_err:
                    self.logger.warning("Failed to delete temporary file %s: %s", path, cleanup_err)
