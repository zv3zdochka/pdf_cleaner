from __future__ import annotations

import asyncio
import logging
import re
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import fitz  # PyMuPDF
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from pdf_cleaner_bot.storage.manager import StorageManager

# user_id -> request_id (ожидание ввода строкой страниц)
_PENDING_PAGES_INPUT: Dict[int, str] = {}


def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(max(0, n))
    i = 0
    while v >= 1024 and i < len(units) - 1:
        v /= 1024
        i += 1
    if i == 0:
        return f"{int(v)} {units[i]}"
    return f"{v:.2f} {units[i]}"


def _kb_for_request(request_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="✅ Обработать", callback_data=f"pdfc:proc:{request_id}")],
            [InlineKeyboardButton(text="🗑 Задать страницы для удаления", callback_data=f"pdfc:pages:{request_id}")],
        ]
    )


def _parse_pages_spec(spec: str, max_page: int) -> List[int]:
    """
    "1, 2, 4-6" -> [1,2,4,5,6] (1-based)
    строгая валидация: любая страница вне 1..max_page -> ошибка
    """
    s = (spec or "").strip()
    if not s or s in {"0", "нет", "none", "no"}:
        return []

    if "," not in s and " " in s:
        s = re.sub(r"\s+", ",", s)

    out: Set[int] = set()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return []

    for p in parts:
        if "-" in p:
            a_str, b_str = [x.strip() for x in p.split("-", 1)]
            if not a_str or not b_str:
                raise ValueError(f"Неверный диапазон: '{p}'. Пример: 4-6")
            if not a_str.isdigit() or not b_str.isdigit():
                raise ValueError(f"Неверный диапазон: '{p}'. Пример: 4-6")
            a = int(a_str)
            b = int(b_str)
            if a <= 0 or b <= 0:
                raise ValueError("Номера страниц должны быть >= 1")
            if a > b:
                a, b = b, a
            for x in range(a, b + 1):
                out.add(x)
        else:
            if not p.isdigit():
                raise ValueError(f"Неверный номер страницы: '{p}'")
            x = int(p)
            if x <= 0:
                raise ValueError("Номера страниц должны быть >= 1")
            out.add(x)

    bad = [x for x in out if x > max_page]
    if bad:
        raise ValueError(f"Есть страницы вне диапазона 1..{max_page}: {sorted(bad)}")

    res = sorted(out)
    if len(res) == max_page:
        raise ValueError("Нельзя удалить все страницы целиком (документ станет пустым).")
    return res


def _pdf_page_count(pdf_path: Path) -> int:
    d = fitz.open(pdf_path)
    n = d.page_count
    d.close()
    return n


def _remove_pages_copy(
    src_pdf: Path,
    dst_pdf: Path,
    pages_to_delete_1based: List[int],
) -> Tuple[int, int]:
    """
    Создаёт копию dst_pdf из src_pdf, удаляя указанные страницы (1-based).
    Исходник не трогаем.
    Возвращает (old_pages, new_pages).
    """
    src = fitz.open(src_pdf)
    old_n = src.page_count

    if not pages_to_delete_1based:
        # если страниц нет — просто копируем целиком
        dst = fitz.open()
        dst.insert_pdf(src)
        dst.save(dst_pdf, garbage=3, deflate=True, clean=True)
        dst.close()
        src.close()
        return old_n, old_n

    del_set = set(pages_to_delete_1based)
    keep = [i for i in range(old_n) if (i + 1) not in del_set]
    if not keep:
        src.close()
        raise ValueError("После удаления страниц документ стал бы пустым.")

    dst = fitz.open()
    for i in keep:
        dst.insert_pdf(src, from_page=i, to_page=i)

    tmp = dst_pdf.with_suffix(".tmp.pdf")
    dst.save(tmp, garbage=3, deflate=True, clean=True)
    dst.close()
    src.close()
    tmp.replace(dst_pdf)

    new_n = _pdf_page_count(dst_pdf)
    return old_n, new_n


def _split_pdf_to_parts_under_limit(
    pdf_path: Path,
    *,
    max_bytes: int,
    tmp_root: Path,
    base_filename_stem: str,
    logger: logging.LoggerAdapter,
) -> List[Path]:
    tmp_root.mkdir(parents=True, exist_ok=True)
    src = fitz.open(pdf_path)
    n = src.page_count

    # небольшой запас
    limit = max(1, int(max_bytes) - 256 * 1024)

    parts: List[Path] = []
    cur_pages: List[int] = []

    def save_pages(pages: List[int], part_idx: int) -> Path:
        out = fitz.open()
        for pi in pages:
            out.insert_pdf(src, from_page=pi, to_page=pi)
        out_path = tmp_root / f"{base_filename_stem}_part{part_idx:02d}.pdf"
        out.save(out_path, garbage=3, deflate=True, clean=True)
        out.close()
        return out_path

    part_idx = 1
    i = 0
    while i < n:
        trial = cur_pages + [i]
        trial_path = save_pages(trial, part_idx)
        sz = trial_path.stat().st_size

        if sz <= limit:
            cur_pages = trial
            i += 1
            continue

        trial_path.unlink(missing_ok=True)

        if not cur_pages:
            src.close()
            raise ValueError("Одна из страниц слишком большая и не помещается в лимит Telegram.")

        final_path = save_pages(cur_pages, part_idx)
        final_sz = final_path.stat().st_size
        logger.info(
            "Split part %s: pages=%s..%s size=%s",
            part_idx,
            cur_pages[0] + 1,
            cur_pages[-1] + 1,
            final_sz,
        )
        parts.append(final_path)
        part_idx += 1
        cur_pages = []

    if cur_pages:
        final_path = save_pages(cur_pages, part_idx)
        final_sz = final_path.stat().st_size
        logger.info(
            "Split part %s: pages=%s..%s size=%s",
            part_idx,
            cur_pages[0] + 1,
            cur_pages[-1] + 1,
            final_sz,
        )
        parts.append(final_path)

    src.close()
    return parts


async def cmd_start(message: Message) -> None:
    await message.answer(
        "Привет! Пришли PDF.\n\n"
        "Я сохраню его и покажу карточку (имя/размер/страницы/время) + кнопки:\n"
        "— Обработать\n"
        "— Задать страницы для удаления\n\n"
        "Файлы хранятся на сервере и доступны через веб-интерфейс."
    )


async def handle_document(
    message: Message,
    bot: Bot,
    *,
    telegram_max_file_size: int,
    internal_max_file_size: int,
    storage: StorageManager,
) -> None:
    """
    При получении PDF:
      - сохраняем input_original.pdf
      - пишем meta
      - показываем карточку + кнопки
    """
    document = message.document
    if not document:
        return

    user_id = message.from_user.id if message.from_user else 0
    original_filename = document.file_name or "document.pdf"
    file_size = document.file_size or 0

    log = logging.getLogger("pdf_cleaner.bot.handlers")

    if file_size > telegram_max_file_size:
        await message.reply(
            "Этот файл слишком большой для Telegram-бота (≈50 МБ). "
            "Сожмите PDF и попробуйте снова."
        )
        return

    if file_size > internal_max_file_size:
        await message.reply("Файл слишком большой для обработки (внутренний лимит 1 ГБ).")
        return

    if not original_filename.lower().endswith(".pdf"):
        await message.reply("Пожалуйста, пришлите PDF-файл.")
        return

    # квота до скачивания (примерно оцениваем +file_size)
    if storage.would_exceed_quota(file_size):
        await message.reply(
            "Хранилище на сервере заполнено (лимит 30 ГБ). "
            "Удалите старые файлы через веб-интерфейс и попробуйте снова."
        )
        return

    request_id = uuid.uuid4().hex
    rd = storage.request_dir(user_id, request_id)
    rd.mkdir(parents=True, exist_ok=True)

    input_original = rd / "input_original.pdf"
    input_trimmed = rd / "input_trimmed.pdf"
    cleaned = rd / "cleaned.pdf"
    cleaned_small = rd / "cleaned_small.pdf"

    meta: Dict[str, Any] = {
        "request_id": request_id,
        "user_id": user_id,
        "original_filename": original_filename,
        "status": "received",
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
        "pages_total_original": None,
        "pages_total_effective": None,
        "pages_to_delete": [],
        "input": {
            "original": {"path": str(input_original.relative_to(storage.cfg.root_dir)), "size_bytes": file_size},
            "trimmed": None,
        },
        "output": {},
        "errors": [],
    }
    storage.write_meta(user_id, request_id, meta)

    adapter = logging.LoggerAdapter(log, {"request_id": request_id})
    adapter.info("Incoming file: name=%s size=%s bytes", original_filename, file_size)

    # download
    try:
        tg_file = await bot.get_file(document.file_id)
    except TelegramBadRequest as e:
        adapter.exception("TelegramBadRequest on get_file: %s", e)
        meta["status"] = "failed_get_file"
        meta["updated_at"] = int(time.time())
        meta["errors"].append({"stage": "get_file", "error": str(e)})
        storage.write_meta(user_id, request_id, meta)
        await message.reply("Telegram отказался отдавать файл (скорее всего, он слишком большой).")
        return

    await bot.download_file(tg_file.file_path, destination=input_original)

    # page count
    try:
        pages_total = _pdf_page_count(input_original)
    except Exception as e:
        adapter.exception("Failed to open PDF for page count: %s", e)
        meta["status"] = "failed_open_pdf"
        meta["updated_at"] = int(time.time())
        meta["errors"].append({"stage": "open_pdf", "error": str(e)})
        storage.write_meta(user_id, request_id, meta)
        await message.reply("Не смог открыть PDF (возможно файл повреждён).")
        return

    # update meta
    meta["status"] = "ready"
    meta["updated_at"] = int(time.time())
    meta["pages_total_original"] = pages_total
    meta["pages_total_effective"] = pages_total
    meta["input"]["original"]["size_bytes"] = input_original.stat().st_size if input_original.exists() else 0
    meta["input"]["trimmed"] = None
    meta["output"] = {}
    storage.write_meta(user_id, request_id, meta)

    # show card
    sent_dt = message.date
    sent_str = sent_dt.strftime("%Y-%m-%d %H:%M:%S") if sent_dt else datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    size_str = _human_bytes(int(meta["input"]["original"]["size_bytes"]))

    text = (
        "Файл сохранён.\n\n"
        f"Имя: {original_filename}\n"
        f"Размер: {size_str}\n"
        f"Страниц: {pages_total}\n"
        f"Время: {sent_str} (UTC)\n\n"
        "Выберите действие:"
    )
    await message.reply(text, reply_markup=_kb_for_request(request_id))


async def handle_pages_text(
    message: Message,
    *,
    storage: StorageManager,
) -> None:
    """
    Пользователь вводит строку страниц.
    Мы создаём input_trimmed.pdf как отдельную копию.
    """
    user_id = message.from_user.id if message.from_user else 0
    if user_id not in _PENDING_PAGES_INPUT:
        return

    request_id = _PENDING_PAGES_INPUT[user_id]
    rd = storage.request_dir(user_id, request_id)
    meta = storage.read_meta(user_id, request_id) or {}
    input_original = rd / "input_original.pdf"
    input_trimmed = rd / "input_trimmed.pdf"

    if not input_original.exists():
        _PENDING_PAGES_INPUT.pop(user_id, None)
        await message.reply("Исходный файл не найден. Отправьте PDF заново.")
        return

    pages_total = int(meta.get("pages_total_original") or 0)
    if pages_total <= 0:
        try:
            pages_total = _pdf_page_count(input_original)
        except Exception:
            _PENDING_PAGES_INPUT.pop(user_id, None)
            await message.reply("Не удалось определить количество страниц. Отправьте файл заново.")
            return

    spec = (message.text or "").strip()

    try:
        pages = _parse_pages_spec(spec, pages_total)
    except ValueError as e:
        await message.reply(
            f"Не понял формат.\nОшибка: {e}\n\n"
            "Введите так: 1,2,4-6\n"
            f"Диапазон допустимых страниц: 1..{pages_total}\n"
            "Или отправьте 0, чтобы очистить список удаления."
        )
        return

    log = logging.getLogger("pdf_cleaner.bot.handlers")
    adapter = logging.LoggerAdapter(log, {"request_id": request_id})

    # Если очищаем список (pages == []): удаляем trimmed-файл (если был) и сбрасываем output
    if not pages:
        input_trimmed.unlink(missing_ok=True)

        # также удалим старые результаты, чтобы не было путаницы
        (rd / "cleaned.pdf").unlink(missing_ok=True)
        (rd / "cleaned_small.pdf").unlink(missing_ok=True)

        meta["pages_to_delete"] = []
        meta["input"]["trimmed"] = None
        meta["pages_total_effective"] = int(meta.get("pages_total_original") or pages_total)
        meta["output"] = {}
        meta["status"] = "ready"
        meta["updated_at"] = int(time.time())
        storage.write_meta(user_id, request_id, meta)

        _PENDING_PAGES_INPUT.pop(user_id, None)
        await message.reply("Ок. Список удаления очищен.\n\nНажмите «Обработать».", reply_markup=_kb_for_request(request_id))
        return

    # Оценка по квоте: создание копии может быть ~размера оригинала (консервативно)
    orig_size = input_original.stat().st_size if input_original.exists() else 0
    if storage.would_exceed_quota(orig_size):
        await message.reply(
            "Не хватает места, чтобы сохранить обрезанную копию (лимит 30 ГБ). "
            "Удалите старые файлы через веб-интерфейс и попробуйте снова."
        )
        return

    # При изменении списка страниц удаляем старые результаты обработки (иначе несоответствие)
    (rd / "cleaned.pdf").unlink(missing_ok=True)
    (rd / "cleaned_small.pdf").unlink(missing_ok=True)

    # Создаём trimmed-копию
    try:
        old_n, new_n = _remove_pages_copy(input_original, input_trimmed, pages)
    except Exception as e:
        adapter.exception("Failed to create trimmed copy: %s", e)
        await message.reply(f"Не удалось создать обрезанную копию: {e}")
        return

    meta["pages_to_delete"] = pages
    meta["input"]["trimmed"] = {
        "path": str(input_trimmed.relative_to(storage.cfg.root_dir)),
        "size_bytes": input_trimmed.stat().st_size if input_trimmed.exists() else 0,
        "pages_deleted": pages,
        "pages_before": old_n,
        "pages_after": new_n,
    }
    meta["pages_total_effective"] = new_n
    meta["output"] = {}
    meta["status"] = "ready"
    meta["updated_at"] = int(time.time())
    storage.write_meta(user_id, request_id, meta)

    _PENDING_PAGES_INPUT.pop(user_id, None)

    await message.reply(
        f"Ок. Сохранил обрезанную копию (страниц было {old_n}, стало {new_n}).\n"
        f"Удалены страницы: {pages}\n\n"
        "Нажмите «Обработать».",
        reply_markup=_kb_for_request(request_id),
    )


async def handle_callback(
    query: CallbackQuery,
    *,
    processor,
    shrink_pdf,
    process_lock: asyncio.Lock,
    telegram_max_file_size: int,
    internal_max_file_size: int,  # не используется, оставлено для совместимости
    storage: StorageManager,
) -> None:
    await query.answer()

    # Удаляем сообщение с кнопками (best-effort)
    try:
        if query.message:
            await query.message.delete()
    except TelegramBadRequest:
        pass
    except Exception:
        pass

    data = (query.data or "").strip()
    if not data.startswith("pdfc:"):
        return

    parts = data.split(":")
    if len(parts) != 3:
        return

    action = parts[1]
    request_id = parts[2]
    user_id = query.from_user.id if query.from_user else 0

    log = logging.getLogger("pdf_cleaner.bot.handlers")
    adapter = logging.LoggerAdapter(log, {"request_id": request_id})

    rd = storage.request_dir(user_id, request_id)
    input_original = rd / "input_original.pdf"
    input_trimmed = rd / "input_trimmed.pdf"
    cleaned = rd / "cleaned.pdf"
    cleaned_small = rd / "cleaned_small.pdf"

    meta = storage.read_meta(user_id, request_id) or {}
    original_filename = meta.get("original_filename") or "document.pdf"

    if action == "pages":
        if not input_original.exists():
            await query.bot.send_message(user_id, "Исходный файл не найден. Отправьте PDF заново.")
            return

        _PENDING_PAGES_INPUT[user_id] = request_id
        meta["status"] = "awaiting_pages_input"
        meta["updated_at"] = int(time.time())
        storage.write_meta(user_id, request_id, meta)

        await query.bot.send_message(
            chat_id=query.message.chat.id if query.message else user_id,
            text=(
                "Введите страницы для удаления в формате:\n"
                "1, 2, 4-6\n\n"
                "Будут удалены: 1 2 4 5 6\n\n"
                "Отправьте 0 — чтобы очистить список удаления."
            ),
        )
        return

    if action != "proc":
        return

    if not input_original.exists():
        await query.bot.send_message(
            chat_id=query.message.chat.id if query.message else user_id,
            text="Файл не найден на сервере. Отправьте PDF заново.",
        )
        return

    # источник для обработки: trimmed если есть, иначе original
    source_pdf = input_trimmed if input_trimmed.exists() else input_original

    # квота: если уже переполнено — отказываем
    if storage.would_exceed_quota(0):
        await query.bot.send_message(
            chat_id=query.message.chat.id if query.message else user_id,
            text="Хранилище на сервере заполнено (лимит 30 ГБ). Удалите старые файлы через веб-интерфейс и попробуйте снова.",
        )
        return

    status = str(meta.get("status") or "")
    if status == "processing":
        await query.bot.send_message(
            chat_id=query.message.chat.id if query.message else user_id,
            text="Файл уже обрабатывается. Подождите немного.",
        )
        return

    try:
        async with process_lock:
            meta = storage.read_meta(user_id, request_id) or meta
            meta["status"] = "processing"
            meta["updated_at"] = int(time.time())
            storage.write_meta(user_id, request_id, meta)

            # обработка (source_pdf не мутируем)
            await asyncio.to_thread(
                processor.process_pdf,
                pdf_path=source_pdf,
                output_path=cleaned,
            )

            # shrink
            await asyncio.to_thread(shrink_pdf, cleaned, cleaned_small)

            # квота после результата: rollback только текущего запроса
            if storage.would_exceed_quota(0):
                shutil.rmtree(rd, ignore_errors=True)
                await query.bot.send_message(
                    chat_id=query.message.chat.id if query.message else user_id,
                    text="После обработки хранилище превысило лимит 30 ГБ. Результат не сохранён. Удалите старые файлы в вебке и повторите.",
                )
                return

            meta["status"] = "done"
            meta["updated_at"] = int(time.time())
            meta["output"] = {
                "cleaned": {
                    "path": str(cleaned.relative_to(storage.cfg.root_dir)),
                    "size_bytes": cleaned.stat().st_size if cleaned.exists() else 0,
                },
                "cleaned_small": {
                    "path": str(cleaned_small.relative_to(storage.cfg.root_dir)),
                    "size_bytes": cleaned_small.stat().st_size if cleaned_small.exists() else 0,
                },
            }
            storage.write_meta(user_id, request_id, meta)

    except Exception as e:
        adapter.exception("Processing failed: %s", e)
        meta = storage.read_meta(user_id, request_id) or meta
        meta["status"] = "failed_processing"
        meta["updated_at"] = int(time.time())
        meta.setdefault("errors", []).append({"stage": "processing", "error": str(e)})
        storage.write_meta(user_id, request_id, meta)
        await query.bot.send_message(
            chat_id=query.message.chat.id if query.message else user_id,
            text="Произошла ошибка при обработке PDF. Проверьте лог сервера.",
        )
        return

    # отправка результата в Telegram, с дроблением если > лимита
    if not cleaned_small.exists():
        await query.bot.send_message(
            chat_id=query.message.chat.id if query.message else user_id,
            text="Результирующий файл не найден. Проверьте лог сервера.",
        )
        return

    result_size = cleaned_small.stat().st_size
    adapter.info("Result size=%s bytes", result_size)

    chat_id = query.message.chat.id if query.message else user_id
    stem = Path(original_filename).stem

    if result_size <= telegram_max_file_size:
        await query.bot.send_document(
            chat_id=chat_id,
            document=FSInputFile(path=str(cleaned_small), filename=original_filename),
            caption="Готово! Вот ваш обработанный PDF.",
        )
        return

    tmp_dir = Path("/tmp") / f"pdf_send_parts_{request_id}"
    try:
        parts_paths = _split_pdf_to_parts_under_limit(
            cleaned_small,
            max_bytes=telegram_max_file_size,
            tmp_root=tmp_dir,
            base_filename_stem=f"{stem}_cleaned",
            logger=adapter,
        )
    except Exception as e:
        adapter.exception("Split failed: %s", e)
        await query.bot.send_message(
            chat_id=chat_id,
            text=(
                "Обработанный PDF получился больше лимита Telegram и не удалось корректно его раздробить.\n\n"
                "Файл сохранён на сервере — скачайте его через веб-интерфейс."
            ),
        )
        return

    total_parts = len(parts_paths)
    for idx, p in enumerate(parts_paths, start=1):
        cap = f"Готово! Часть {idx}/{total_parts}." if idx == 1 else f"Часть {idx}/{total_parts}."
        await query.bot.send_document(
            chat_id=chat_id,
            document=FSInputFile(path=str(p), filename=p.name),
            caption=cap,
        )

    shutil.rmtree(tmp_dir, ignore_errors=True)
