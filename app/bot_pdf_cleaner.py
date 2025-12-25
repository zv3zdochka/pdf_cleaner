# bot_pdf_cleaner.py
import asyncio
import logging
import uuid
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

import fitz  # PyMuPDF
from PIL import Image
import cv2  # may be used inside RFDetrONNX
import numpy as np  # may be used inside RFDetrONNX
import pikepdf

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message, FSInputFile
from aiogram.exceptions import TelegramBadRequest

from rfdetr import RFDetrONNX


# ==========================
# Configuration
# ==========================

BOT_TOKEN = "8323732321:AAGTZtlv0EO5tcXHLDbYRt7KrZTCoTTZ4UE"  # <-- поставь сюда токен бота

# Путь к модели. В Colab часто это что-то вроде:
# MODEL_PATH = Path("/content/drive/MyDrive/Mark_pdf_removal/weights.onnx")
MODEL_PATH = Path("weights.onnx")

WORK_DIR = Path("data")  # базовая папка для временных файлов
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Лимит Telegram на файл (отправка/получение) ~50 МБ
TELEGRAM_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MiB

# Внутренний лимит обработки (если будешь использовать вне Telegram) — до 1 ГБ
INTERNAL_MAX_FILE_SIZE = 1 * 1024 * 1024 * 1024  # 1 GiB

# Глобальный lock, чтобы не гонять несколько обработок параллельно
PROCESS_LOCK = asyncio.Lock()


class PDFRegionProcessor:
    """
    Processes a PDF: renders pages, detects regions via RF-DETR,
    and clears the content inside detected regions using PDF redaction.
    """

    def __init__(
        self,
        model_path: Path,
        input_size: Tuple[int, int] = (736, 736),
        conf_threshold: float = 0.5,
        render_zoom: float = 2.0,
        providers: Optional[List[str]] = None,
    ):
        """
        :param model_path: Path to RF-DETR ONNX model
        :param input_size: Model input size (width, height)
        :param conf_threshold: Confidence threshold for detections
        :param render_zoom: Zoom factor for rendering PDF pages to images
        :param providers: ONNX Runtime execution providers
        """
        self.render_zoom = render_zoom
        self.logger = logging.getLogger("PDFRegionProcessor")

        if providers is None:
            providers = ["CPUExecutionProvider"]  # для Colab без CUDA

        # Initialize detector once and reuse it for all requests
        self.detector = RFDetrONNX(
            model_path=model_path,
            input_size=input_size,
            conf_threshold=conf_threshold,
            providers=providers,
            logger=logging.getLogger("RFDetrONNX"),
        )

    def render_page_to_image(self, page: fitz.Page) -> Tuple[Image.Image, fitz.Matrix]:
        """
        Render a single PDF page to a PIL Image.

        :return: (PIL Image, transform matrix used during rendering)
        """
        matrix = fitz.Matrix(self.render_zoom, self.render_zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        mode = "RGB" if pix.n < 4 else "RGBA"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)

        return img, matrix

    def convert_bbox_to_pdf_coords(
        self,
        bbox: List[int],
        page: fitz.Page,
        render_matrix: fitz.Matrix,
    ) -> fitz.Rect:
        """
        Convert a bounding box from rendered image coordinates
        back to PDF page coordinates, taking page rotation into account.

        :param bbox: [x1, y1, x2, y2] in rendered image coordinates
        :param page: fitz.Page object
        :param render_matrix: Matrix used during rendering
        :return: fitz.Rect in PDF coordinates
        """
        x1, y1, x2, y2 = bbox
        x1_orig, y1_orig, x2_orig, y2_orig = x1, y1, x2, y2

        rotation = page.rotation

        # Rendered image size
        pix = page.get_pixmap(matrix=render_matrix, alpha=False)
        img_width, img_height = pix.width, pix.height

        page_rect = page.rect
        pdf_width, pdf_height = page_rect.width, page_rect.height

        self.logger.info(f"      DEBUG: rotation={rotation}°")
        self.logger.info(f"      DEBUG: img size=({img_width}x{img_height})")
        self.logger.info(f"      DEBUG: pdf size=({pdf_width:.1f}x{pdf_height:.1f})")
        self.logger.info(f"      DEBUG: bbox_img=[{x1_orig}, {y1_orig}, {x2_orig}, {y2_orig}]")

        # Rotation correction: undo rotation applied by renderer
        if rotation == 90:
            # 90° clockwise: x' = y, y' = width - x
            x1_rot, y1_rot = y1, img_width - x2
            x2_rot, y2_rot = y2, img_width - x1
            x1, y1, x2, y2 = x1_rot, y1_rot, x2_rot, y2_rot
            self.logger.info(f"      DEBUG: after rotation correction=[{x1}, {y1}, {x2}, {y2}]")
        elif rotation == 180:
            # 180°: x' = width - x, y' = height - y
            x1_rot, y1_rot = img_width - x2, img_height - y2
            x2_rot, y2_rot = img_width - x1, img_height - y1
            x1, y1, x2, y2 = x1_rot, y1_rot, x2_rot, y2_rot
            self.logger.info(f"      DEBUG: after rotation correction=[{x1}, {y1}, {x2}, {y2}]")
        elif rotation == 270:
            # 270° clockwise: x' = height - y, y' = x
            x1_rot, y1_rot = img_height - y2, x1
            x2_rot, y2_rot = img_height - y1, x2
            x1, y1, x2, y2 = x1_rot, y1_rot, x2_rot, y2_rot
            self.logger.info(f"      DEBUG: after rotation correction=[{x1}, {y1}, {x2}, {y2}]")
        else:
            self.logger.info("      DEBUG: no rotation correction needed")

        # Inverse matrix to map from image coordinates back to PDF coordinates
        inv_matrix = ~render_matrix

        # Transform bbox corners
        p1 = fitz.Point(x1, y1) * inv_matrix
        p2 = fitz.Point(x2, y2) * inv_matrix

        self.logger.info(f"      DEBUG: p1_pdf=({p1.x:.1f}, {p1.y:.1f}), p2_pdf=({p2.x:.1f}, {p2.y:.1f})")

        # Create rectangle and normalize it
        rect = fitz.Rect(p1, p2)
        rect.normalize()

        self.logger.info(
            f"      DEBUG: final_rect=[{rect.x0:.1f}, {rect.y0:.1f}, {rect.x1:.1f}, {rect.y1:.1f}]"
        )

        return rect

    def process_pdf(
        self,
        pdf_path: Path,
        output_path: Optional[Path] = None,
        temp_dir: Optional[Path] = None,
    ) -> Path:
        """
        Process a PDF: detect regions on each page and clear
        the content inside detected regions using redactions.
        No rectangles/text annotations are drawn.

        :param pdf_path: Input PDF path
        :param output_path: Output PDF path (if None, suffix '_cleaned' is added)
        :param temp_dir: Directory for temporary rendered images
        :return: Path to cleaned PDF
        """
        self.logger.info(f"Processing PDF: {pdf_path}")

        doc = fitz.open(pdf_path)

        # Determine output path
        if output_path is None:
            output_path = pdf_path.parent / f"{pdf_path.stem}_cleaned.pdf"

        # Unique temporary directory for this processing
        created_temp_dir = False
        if temp_dir is None:
            temp_dir = pdf_path.parent / f"temp_renders_{uuid.uuid4().hex}"
            created_temp_dir = True

        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                self.logger.info(f"  Page {page_num + 1}/{len(doc)} (rotation={page.rotation})")

                # 1) Render page to image
                img, render_matrix = self.render_page_to_image(page)

                # Save temporary image for detector
                temp_img_path = temp_dir / f"page_{page_num:04d}.png"
                img.save(temp_img_path)

                # 2) Run detector
                detections, _ = self.detector.predict_from_path(temp_img_path)
                self.logger.info(f"    Detected regions: {len(detections)}")

                # 3) Convert bboxes to PDF coords and add redact annotations
                redactions = 0
                for i, det in enumerate(detections):
                    bbox = det["bbox"]  # [x1, y1, x2, y2] in image coordinates
                    conf = det["confidence"]

                    pdf_rect = self.convert_bbox_to_pdf_coords(bbox, page, render_matrix)

                    # Mark region as redaction zone, no fill on top
                    page.add_redact_annot(pdf_rect, fill=None)
                    redactions += 1

                    self.logger.info(
                        f"      Region {i + 1}: conf={conf:.3f}, rect="
                        f"[{pdf_rect.x0:.1f}, {pdf_rect.y0:.1f}, {pdf_rect.x1:.1f}, {pdf_rect.y1:.1f}]"
                    )

                # 4) Apply redactions if any
                if redactions > 0:
                    page.apply_redactions(
                        text=fitz.PDF_REDACT_TEXT_REMOVE,
                        images=fitz.PDF_REDACT_IMAGE_PIXELS,
                        graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED,
                    )
                    self.logger.info(f"    Applied redactions: {redactions}")
                else:
                    self.logger.info("    Nothing to redact (no detections)")

                # Remove temporary image file
                temp_img_path.unlink(missing_ok=True)

            # Save cleaned PDF
            doc.save(output_path, garbage=3, deflate=True, clean=True)
            doc.close()

            self.logger.info(f"Cleaned PDF saved: {output_path}")
            return output_path

        finally:
            # Cleanup unique temp directory
            if created_temp_dir:
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    self.logger.warning("Failed to remove temp dir %s: %s", temp_dir, e)


def shrink_pdf(input_path: str | Path, output_path: str | Path) -> None:
    """
    Compress a PDF using pikepdf: compress streams and pack objects.
    """
    input_path = str(input_path)
    output_path = str(output_path)
    with pikepdf.Pdf.open(input_path) as pdf:
        pdf.save(
            output_path,
            compress_streams=True,
            object_stream_mode=pikepdf.ObjectStreamMode.generate,
        )


# Create a single global processor instance so the model
# is loaded only once per process.
processor = PDFRegionProcessor(
    model_path=MODEL_PATH,
    input_size=(736, 736),
    conf_threshold=0.5,
    render_zoom=2.0,
    providers=["CPUExecutionProvider"],  # для Colab без CUDA
)


# ==========================
# Telegram bot handlers
# ==========================

async def cmd_start(message: Message) -> None:
    """
    /start handler: simple greeting and short instructions.
    """
    await message.answer(
        "Привет! Пришлите мне PDF-файл (до 50 МБ — это ограничение Telegram), "
        "я автоматически найду и очищу нужные зоны, "
        "а затем пришлю вам обработанный PDF с тем же именем файла."
    )


async def handle_document(message: Message, bot: Bot) -> None:
    """
    Handle incoming documents. Only PDF files are processed.
    1) Check Telegram size limit
    2) Download file to disk (with UUID name)
    3) Run PDFRegionProcessor (under PROCESS_LOCK)
    4) Compress result
    5) Send processed PDF back to the user with original filename
    """
    document = message.document
    if not document:
        return

    original_filename = document.file_name or "document.pdf"
    file_size = document.file_size or 0

    # 1) Hard Telegram limit
    if file_size > TELEGRAM_MAX_FILE_SIZE:
        await message.reply(
            "Этот файл слишком большой для Telegram-бота. "
            "Ограничение Telegram для ботов — примерно 50 МБ на файл.\n\n"
            "Попробуйте отправить более компактный PDF "
            "или предварительно сжать его."
        )
        return

    # 2) Your internal processing limit (for safety)
    if file_size > INTERNAL_MAX_FILE_SIZE:
        await message.reply(
            "Файл слишком большой для обработки (внутренний лимит 1 ГБ)."
        )
        return

    if not original_filename.lower().endswith(".pdf"):
        await message.reply("Пожалуйста, пришлите PDF-файл.")
        return

    await message.reply(
        "Файл получен, начинаю обработку. На больших документах это может занять время..."
    )

    # Generate safe name using UUID to avoid issues with Cyrillic, spaces, etc.
    unique_id = uuid.uuid4().hex
    input_path = WORK_DIR / f"{message.from_user.id}_{unique_id}.pdf"

    logger = logging.getLogger("BotHandler")

    # Download file to local storage (streaming, without loading whole file into memory)
    try:
        tg_file = await bot.get_file(document.file_id)
    except TelegramBadRequest as e:
        logger.exception("TelegramBadRequest on get_file: %s", e)
        await message.reply(
            "Telegram отказался отдавать файл (скорее всего, он слишком большой)."
        )
        return

    await bot.download_file(tg_file.file_path, destination=input_path)
    logger.info("Downloaded file to %s (original name: %s)", input_path, original_filename)

    cleaned_path: Optional[Path] = None
    final_path: Optional[Path] = None

    try:
        # Сериализуем тяжёлую обработку, чтобы избежать гонок
        async with PROCESS_LOCK:
            # Run heavy PDF processing in a separate thread
            cleaned_path = await asyncio.to_thread(
                processor.process_pdf,
                pdf_path=input_path,
                output_path=None,
            )

            # Additional shrink step
            final_path = cleaned_path.parent / f"{cleaned_path.stem}_small.pdf"
            await asyncio.to_thread(shrink_pdf, cleaned_path, final_path)

        logger.info("Processed file saved to %s", final_path)

        # Check Telegram output size limit
        try:
            result_size = final_path.stat().st_size
        except FileNotFoundError:
            result_size = 0

        if result_size > TELEGRAM_MAX_FILE_SIZE:
            await message.reply(
                "Обработанный PDF получился больше лимита Telegram (≈50 МБ), "
                "поэтому я не могу отправить его обратно через бот.\n\n"
                "Файл сохранён на сервере, но для такого размера нужно использовать "
                "другой канал передачи (например, облако и ссылку)."
            )
            return

        # Send processed file back to user with original filename
        await message.reply_document(
            FSInputFile(path=str(final_path), filename=original_filename),
            caption="Готово! Вот ваш обработанный PDF.",
        )

    except Exception as e:
        logger.exception("Error while processing PDF: %s", e)
        await message.reply("Произошла ошибка при обработке PDF. Проверьте лог сервера.")
    finally:
        # Best-effort cleanup of temporary files to avoid filling disk with large files
        for path in (input_path, cleaned_path, final_path):
            try:
                if isinstance(path, Path):
                    path.unlink(missing_ok=True)
            except Exception as cleanup_err:
                logger.warning("Failed to delete temporary file %s: %s", path, cleanup_err)


# ==========================
# Entry point
# ==========================

async def main() -> None:
    """
    Entry point: configure logging, create bot & dispatcher,
    register handlers and start polling.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if BOT_TOKEN == "PASTE_YOUR_BOT_TOKEN_HERE":
        raise RuntimeError("Please set BOT_TOKEN in the script before running the bot.")

    bot = Bot(BOT_TOKEN)
    dp = Dispatcher()

    dp.message.register(cmd_start, CommandStart())
    dp.message.register(handle_document, F.document)

    logging.getLogger("Main").info("Bot started. Waiting for updates...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    # В Colab запускать через: !python bot_pdf_cleaner.py
    asyncio.run(main())
