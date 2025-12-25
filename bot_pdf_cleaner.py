import asyncio
import logging

from aiogram import Bot

from pdf_cleaner_bot.settings import Settings
from pdf_cleaner_bot.logging_setup import setup_logging
from pdf_cleaner_bot.bot.app import build_dispatcher

from pdf_cleaner_bot.processor.region_processor import PDFRegionProcessor
from pdf_cleaner_bot.processor.shrink import shrink_pdf


async def main() -> None:
    settings = Settings.from_env()

    setup_logging(settings.logs_dir, level=settings.log_level)

    log = logging.getLogger("pdf_cleaner.main")

    if not settings.bot_token:
        raise RuntimeError("BOT_TOKEN is not set")

    # ensure dirs exist
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)

    processor = PDFRegionProcessor(
        model_path=settings.model_path,
        input_size=(736, 736),
        conf_threshold=0.5,
        render_zoom=2.0,
        providers=["CPUExecutionProvider"],
    )

    bot = Bot(settings.bot_token)
    dp = build_dispatcher(processor=processor, shrink_pdf=shrink_pdf, settings=settings)

    log.info("Bot started. Waiting for updates...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
