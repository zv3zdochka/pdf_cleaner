from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

from aiogram.types import Message, FSInputFile
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest

# добавь эти импорты
from pdf_cleaner_bot.storage.manager import StorageManager, StorageConfig


async def handle_document(
    message: Message,
    bot: Bot,
    *,
    processor,
    shrink_pdf,
    process_lock: asyncio.Lock,
    telegram_max_file_size: int,
    internal_max_file_size: int,
    storage: StorageManager,
) -> None:
    document = message.document
    if not document:
        return

    user_id = message.from_user.id if message.from_user else 0
    original_filename = document.file_name or "document.pdf"
    file_size = document.file_size or 0

    log = logging.getLogger("pdf_cleaner.bot.handlers")

    # Best-effort cleanup before accepting new work
    try:
        storage.cleanup()
    except Exception:
        pass

    # 1) Hard Telegram limit
    if file_size > telegram_max_file_size:
        await message.reply(
            "Этот файл слишком большой для Telegram-бота. "
            "Ограничение Telegram для ботов — примерно 50 МБ на файл.\n\n"
            "Попробуйте отправить более компактный PDF "
            "или предварительно сжать его."
        )
        return

    # 2) Internal safety limit
    if file_size > internal_max_file_size:
        await message.reply("Файл слишком большой для обработки (внутренний лимит 1 ГБ).")
        return

    if not original_filename.lower().endswith(".pdf"):
        await message.reply("Пожалуйста, пришлите PDF-файл.")
        return

    await message.reply("Файл получен, начинаю обработку. На больших документах это может занять время...")

    request_id = uuid.uuid4().hex
    req_dir = storage.request_dir(user_id, request_id)
    req_dir.mkdir(parents=True, exist_ok=True)

    input_path = req_dir / "input.pdf"
    cleaned_path: Optional[Path] = None
    final_path: Optional[Path] = None

    base_meta: Dict[str, Any] = {
        "request_id": request_id,
        "user_id": user_id,
        "original_filename": original_filename,
        "status": "received",
        "created_at": int(time.time()),
        "input": {"path": str(input_path.relative_to(storage.cfg.root_dir))},
        "output": {},
        "errors": [],
    }
    storage.write_meta(user_id, request_id, base_meta)

    adapter = logging.LoggerAdapter(log, {"request_id": request_id})
    adapter.info("Incoming file: name=%s size=%s bytes", original_filename, file_size)

    # Download file to storage
    try:
        tg_file = await bot.get_file(document.file_id)
    except TelegramBadRequest as e:
        adapter.exception("TelegramBadRequest on get_file: %s", e)
        base_meta["status"] = "failed_get_file"
        base_meta["errors"].append({"stage": "get_file", "error": str(e)})
        storage.write_meta(user_id, request_id, base_meta)
        await message.reply("Telegram отказался отдавать файл (скорее всего, он слишком большой).")
        return

    await bot.download_file(tg_file.file_path, destination=input_path)
    adapter.info("Downloaded to %s", input_path)

    base_meta["status"] = "downloaded"
    base_meta["input"]["size_bytes"] = input_path.stat().st_size if input_path.exists() else 0
    storage.write_meta(user_id, request_id, base_meta)

    try:
        async with process_lock:
            base_meta["status"] = "processing"
            storage.write_meta(user_id, request_id, base_meta)

            # Run heavy PDF processing
            cleaned_path = await asyncio.to_thread(
                processor.process_pdf,
                pdf_path=input_path,
                output_path=req_dir / "cleaned.pdf",
            )

            # Shrink
            final_path = req_dir / "cleaned_small.pdf"
            await asyncio.to_thread(shrink_pdf, cleaned_path, final_path)

        # Update meta
        base_meta["status"] = "done"
        base_meta["output"] = {
            "cleaned": {
                "path": str((req_dir / "cleaned.pdf").relative_to(storage.cfg.root_dir)),
                "size_bytes": (req_dir / "cleaned.pdf").stat().st_size if (req_dir / "cleaned.pdf").exists() else 0,
            },
            "cleaned_small": {
                "path": str(final_path.relative_to(storage.cfg.root_dir)),
                "size_bytes": final_path.stat().st_size if final_path.exists() else 0,
            },
        }
        storage.write_meta(user_id, request_id, base_meta)

        adapter.info("Processing finished. final=%s bytes=%s", final_path, final_path.stat().st_size)

        # Telegram output size check
        result_size = final_path.stat().st_size if final_path and final_path.exists() else 0
        if result_size > telegram_max_file_size:
            await message.reply(
                "Обработанный PDF получился больше лимита Telegram (≈50 МБ), "
                "поэтому я не могу отправить его обратно через бот.\n\n"
                "Файл сохранён на сервере. Позже можно будет скачать его через веб-интерфейс."
            )
            return

        await message.reply_document(
            FSInputFile(path=str(final_path), filename=original_filename),
            caption="Готово! Вот ваш обработанный PDF.",
        )

    except Exception as e:
        adapter.exception("Error while processing PDF: %s", e)
        base_meta["status"] = "failed_processing"
        base_meta["errors"].append({"stage": "processing", "error": str(e)})
        storage.write_meta(user_id, request_id, base_meta)
        await message.reply("Произошла ошибка при обработке PDF. Проверьте лог сервера.")
    finally:
        # IMPORTANT: больше НЕ удаляем вход/выход — они остаются для вебки
        # Но запускаем уборку по квоте/TTL
        try:
            storage.cleanup()
        except Exception:
            pass
