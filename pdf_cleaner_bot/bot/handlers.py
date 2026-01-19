from __future__ import annotations

import asyncio
import logging
import re
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict

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


class PendingPages(TypedDict):
    request_id: str
    chat_id: int
    prompt_message_id: int


# user_id -> {request_id, chat_id, prompt_message_id}
_PENDING_PAGES_INPUT: Dict[int, PendingPages] = {}


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


def _kb_actions(request_id: str) -> InlineKeyboardMarkup:
    """
    Только управляющие кнопки (первое сообщение после загрузки).
    """
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Обработать", callback_data=f"pdfc:proc:{request_id}"),
                InlineKeyboardButton(text="🗑 Страницы", callback_data=f"pdfc:pages:{request_id}"),
            ],
            [
                InlineKeyboardButton(text="🆘 Поддержка", url="https://t.me/vrekota"),
            ]
        ]
    )


def _kb_downloads(request_id: str) -> InlineKeyboardMarkup:
    """
    Только скачивания (сообщение после того, как бот прислал результат).
    """
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="⬇️ Исходный", callback_data=f"pdfc:dl:{request_id}:orig"),
                InlineKeyboardButton(text="⬇️ Обрезанный", callback_data=f"pdfc:dl:{request_id}:trim"),
                InlineKeyboardButton(text="⬇️ Обработанный", callback_data=f"pdfc:dl:{request_id}:proc"),
            ],
            [
                InlineKeyboardButton(text="🆘 Поддержка", url="https://t.me/vrekota"),
            ]
        ]
    )


def _parse_pages_spec(spec: str, max_page: int) -> List[int]:
    """
    "1, 2, 4-6" -> [1,2,4,5,6] (1-based)
    строгая валидация: любая страница вне 1..max_page -> ошибка
    пересечения/дубликаты -> норм (через set)
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
        dst = fitz.open()
        dst.insert_pdf(src)
        tmp = dst_pdf.with_suffix(".tmp.pdf")
        dst.save(tmp, garbage=3, deflate=True, clean=True)
        dst.close()
        src.close()
        tmp.replace(dst_pdf)
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

    limit = max(1, int(max_bytes) - 256 * 1024)  # небольшой запас

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


def _fmt_time_utc(ts: int) -> str:
    if not ts:
        return "-"
    return datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")


def _build_card_text(meta: Dict[str, Any]) -> str:
    original_filename = meta.get("original_filename") or "document.pdf"
    status = meta.get("status") or "unknown"

    created_at = int(meta.get("telegram_received_at") or meta.get("created_at") or 0)

    pages_orig = meta.get("pages_total_original")
    pages_eff = meta.get("pages_total_effective")

    inp_orig = ((meta.get("input") or {}).get("original") or {})
    size_bytes = int(inp_orig.get("size_bytes") or 0)

    pages_to_delete = meta.get("pages_to_delete") or []
    if isinstance(pages_to_delete, str):
        pages_to_delete = []

    lines = [
        "Файл сохранён.",
        "",
        f"Имя: {original_filename}",
        f"Размер: {_human_bytes(size_bytes)}",
        f"Страниц: {pages_orig if pages_orig is not None else '-'}"
        + (f" → {pages_eff}" if pages_eff is not None and pages_eff != pages_orig else ""),
        f"Время: {_fmt_time_utc(created_at)} (UTC)",
        f"Статус: {status}",
    ]

    if pages_to_delete:
        lines.append(f"Удаляем страницы: {pages_to_delete}")

    lines.append("")
    lines.append("Выберите действие:")
    return "\n".join(lines)


async def _send_pdf_to_chat(
        *,
        bot: Bot,
        chat_id: int,
        path: Path,
        filename: str,
        caption: Optional[str],
        telegram_max_file_size: int,
        request_id: str,
        logger: logging.LoggerAdapter,
) -> None:
    if not path.exists():
        await bot.send_message(chat_id, "Файл не найден на сервере.")
        return

    sz = path.stat().st_size
    if sz <= telegram_max_file_size:
        await bot.send_document(
            chat_id=chat_id,
            document=FSInputFile(path=str(path), filename=filename),
            caption=caption,
        )
        return

    # if too big -> split, else fallback to web
    tmp_dir = Path("/tmp") / f"pdf_send_parts_{request_id}_{uuid.uuid4().hex}"
    stem = Path(filename).stem
    try:
        parts_paths = _split_pdf_to_parts_under_limit(
            path,
            max_bytes=telegram_max_file_size,
            tmp_root=tmp_dir,
            base_filename_stem=stem,
            logger=logger,
        )
    except Exception as e:
        logger.exception("Split failed: %s", e)
        await bot.send_message(
            chat_id,
            "Файл слишком большой для Telegram и не удалось корректно его раздробить. Скачайте через веб-интерфейс.",
        )
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    total_parts = len(parts_paths)
    for idx, p in enumerate(parts_paths, start=1):
        cap = caption if (idx == 1 and caption) else None
        cap2 = cap or f"Часть {idx}/{total_parts}."
        await bot.send_document(
            chat_id=chat_id,
            document=FSInputFile(path=str(p), filename=p.name),
            caption=cap2,
        )

    shutil.rmtree(tmp_dir, ignore_errors=True)


async def cmd_start(message: Message) -> None:
    await message.answer(
        "Привет! Пришли PDF.\n\n"
        "Я сохраню его и покажу карточку (имя/размер/страницы/время) + кнопки:\n"
        "— Обработать\n"
        "— Задать страницы для удаления\n"
        "— Скачать исходный/обрезанный/обработанный\n\n"
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

    meta: Dict[str, Any] = {
        "request_id": request_id,
        "user_id": user_id,
        "original_filename": original_filename,
        "status": "received",
        "created_at": int(time.time()),
        "telegram_received_at": int(message.date.timestamp()) if message.date else int(time.time()),
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

    meta["status"] = "ready"
    meta["updated_at"] = int(time.time())
    meta["pages_total_original"] = pages_total
    meta["pages_total_effective"] = pages_total
    meta["input"]["original"]["size_bytes"] = input_original.stat().st_size if input_original.exists() else 0
    storage.write_meta(user_id, request_id, meta)

    # ВАЖНО: первое сообщение — только (Обработать, Страницы)
    await message.reply(_build_card_text(meta), reply_markup=_kb_actions(request_id))


async def handle_pages_text(
        message: Message,
        *,
        storage: StorageManager,
) -> None:
    user_id = message.from_user.id if message.from_user else 0
    pending = _PENDING_PAGES_INPUT.get(user_id)
    if not pending:
        return

    request_id = pending["request_id"]
    rd = storage.request_dir(user_id, request_id)
    meta = storage.read_meta(user_id, request_id) or {}

    input_original = rd / "input_original.pdf"
    input_trimmed = rd / "input_trimmed.pdf"

    # delete prompt message (best-effort)
    try:
        await message.bot.delete_message(chat_id=pending["chat_id"], message_id=pending["prompt_message_id"])
    except Exception:
        pass

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

    # If pages cleared -> remove trimmed and outputs
    if not pages:
        input_trimmed.unlink(missing_ok=True)
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
        # ДО обработки — только (Обработать, Страницы)
        await message.reply(_build_card_text(meta), reply_markup=_kb_actions(request_id))
        return

    # Conservative quota check: trimmed copy can be near original size
    orig_size = input_original.stat().st_size if input_original.exists() else 0
    if storage.would_exceed_quota(orig_size):
        await message.reply(
            "Не хватает места, чтобы сохранить обрезанную копию (лимит 30 ГБ). "
            "Удалите старые файлы через веб-интерфейс и попробуйте снова."
        )
        return

    # Remove old outputs (otherwise mismatch with new trimmed)
    (rd / "cleaned.pdf").unlink(missing_ok=True)
    (rd / "cleaned_small.pdf").unlink(missing_ok=True)

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
    # ДО обработки — только (Обработать, Страницы)
    await message.reply(_build_card_text(meta), reply_markup=_kb_actions(request_id))


async def handle_callback(
        query: CallbackQuery,
        *,
        processor,
        shrink_pdf,
        process_lock: asyncio.Lock,
        telegram_max_file_size: int,
        internal_max_file_size: int,  # оставлено для совместимости
        storage: StorageManager,
) -> None:
    await query.answer()

    data = (query.data or "").strip()
    if not data.startswith("pdfc:"):
        return

    parts = data.split(":")
    if len(parts) < 3:
        return

    action = parts[1]
    request_id = parts[2]
    kind = parts[3] if (action == "dl" and len(parts) >= 4) else ""

    user_id = query.from_user.id if query.from_user else 0
    chat_id = query.message.chat.id if query.message else user_id

    log = logging.getLogger("pdf_cleaner.bot.handlers")
    adapter = logging.LoggerAdapter(log, {"request_id": request_id})

    rd = storage.request_dir(user_id, request_id)
    input_original = rd / "input_original.pdf"
    input_trimmed = rd / "input_trimmed.pdf"
    cleaned = rd / "cleaned.pdf"
    cleaned_small = rd / "cleaned_small.pdf"

    meta = storage.read_meta(user_id, request_id) or {}
    original_filename = meta.get("original_filename") or "document.pdf"
    stem = Path(original_filename).stem

    # -------------------------
    # Downloads (DO NOT delete card message)
    # -------------------------
    if action == "dl":
        if kind == "orig":
            if not input_original.exists():
                await query.answer("Исходный файл не найден.", show_alert=True)
                return
            await _send_pdf_to_chat(
                bot=query.bot,
                chat_id=chat_id,
                path=input_original,
                filename=original_filename,
                caption="Исходный файл.",
                telegram_max_file_size=telegram_max_file_size,
                request_id=request_id,
                logger=adapter,
            )
            return

        if kind == "trim":
            if not input_trimmed.exists():
                await query.answer("Обрезанный файл ещё не создан. Сначала задайте страницы.", show_alert=True)
                return
            await _send_pdf_to_chat(
                bot=query.bot,
                chat_id=chat_id,
                path=input_trimmed,
                filename=f"{stem}_trimmed.pdf",
                caption="Обрезанная копия (ещё не обработана).",
                telegram_max_file_size=telegram_max_file_size,
                request_id=request_id,
                logger=adapter,
            )
            return

        if kind == "proc":
            if not cleaned_small.exists():
                await query.answer("Обработанный файл ещё не готов. Нажмите «Обработать».", show_alert=True)
                return
            await _send_pdf_to_chat(
                bot=query.bot,
                chat_id=chat_id,
                path=cleaned_small,
                filename=original_filename,
                caption="Обработанный файл (после сжатия).",
                telegram_max_file_size=telegram_max_file_size,
                request_id=request_id,
                logger=adapter,
            )
            return

        await query.answer("Неизвестный тип файла.", show_alert=True)
        return

    # -------------------------
    # For actions below: delete the card message (best-effort)
    # -------------------------
    if action in {"proc", "pages"}:
        try:
            if query.message:
                await query.message.delete()
        except Exception:
            pass

    # -------------------------
    # Pages prompt
    # -------------------------
    if action == "pages":
        if not input_original.exists():
            await query.bot.send_message(chat_id, "Исходный файл не найден. Отправьте PDF заново.")
            return

        _PENDING_PAGES_INPUT.pop(user_id, None)

        meta["status"] = "awaiting_pages_input"
        meta["updated_at"] = int(time.time())
        storage.write_meta(user_id, request_id, meta)

        msg = await query.bot.send_message(
            chat_id=chat_id,
            text=(
                "Введите страницы для удаления в формате:\n"
                "1, 2, 4-6\n\n"
                "Будут удалены: 1, 2, 4, 5, 6\n\n"
                "Отправьте 0 — чтобы очистить список удаления."
            ),
        )
        _PENDING_PAGES_INPUT[user_id] = {
            "request_id": request_id,
            "chat_id": chat_id,
            "prompt_message_id": msg.message_id,
        }
        return

    # -------------------------
    # Processing
    # -------------------------
    if action != "proc":
        return

    if not input_original.exists():
        await query.bot.send_message(chat_id, "Файл не найден на сервере. Отправьте PDF заново.")
        return

    if storage.would_exceed_quota(0):
        await query.bot.send_message(
            chat_id,
            "Хранилище на сервере заполнено (лимит 30 ГБ). Удалите старые файлы через веб-интерфейс и попробуйте снова.",
        )
        return

    if str(meta.get("status") or "") == "processing":
        await query.bot.send_message(chat_id, "Файл уже обрабатывается. Подождите немного.")
        return

    # source for processing: trimmed if exists, else original
    source_pdf = input_trimmed if input_trimmed.exists() else input_original

    processing_msg: Optional[Message] = None
    try:
        processing_msg = await query.bot.send_message(chat_id, "⏳ Обрабатывается...")

        async with process_lock:
            meta = storage.read_meta(user_id, request_id) or meta
            meta["status"] = "processing"
            meta["updated_at"] = int(time.time())
            storage.write_meta(user_id, request_id, meta)

            await asyncio.to_thread(
                processor.process_pdf,
                pdf_path=source_pdf,
                output_path=cleaned,
            )
            await asyncio.to_thread(shrink_pdf, cleaned, cleaned_small)

            if storage.would_exceed_quota(0):
                shutil.rmtree(rd, ignore_errors=True)
                if processing_msg:
                    try:
                        await processing_msg.delete()
                    except Exception:
                        pass
                await query.bot.send_message(
                    chat_id,
                    "После обработки хранилище превысило лимит 30 ГБ. Результат не сохранён. Удалите старые файлы в вебке и повторите.",
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

        if processing_msg:
            try:
                await processing_msg.delete()
            except Exception:
                pass

        await query.bot.send_message(chat_id, "Произошла ошибка при обработке PDF. Проверьте лог сервера.")
        # после ошибки: файл не прислан -> оставляем только (Обработать, Страницы)
        await query.bot.send_message(chat_id, _build_card_text(meta), reply_markup=_kb_actions(request_id))
        return

    # send result (or split)
    if cleaned_small.exists():
        await _send_pdf_to_chat(
            bot=query.bot,
            chat_id=chat_id,
            path=cleaned_small,
            filename=original_filename,
            caption="Готово! Вот ваш обработанный PDF.",
            telegram_max_file_size=telegram_max_file_size,
            request_id=request_id,
            logger=adapter,
        )
    else:
        await query.bot.send_message(chat_id, "Результирующий файл не найден. Проверьте лог сервера.")

    # delete "processing" message after sending
    if processing_msg:
        try:
            await processing_msg.delete()
        except Exception:
            pass

    # show updated card AFTER result: только кнопки скачивания
    meta = storage.read_meta(user_id, request_id) or meta
    await query.bot.send_message(chat_id, _build_card_text(meta), reply_markup=_kb_downloads(request_id))
