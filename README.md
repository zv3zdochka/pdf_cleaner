# PDF Cleaner Bot (modularized)

Telegram-бот для очистки PDF: рендерит страницы, находит зоны моделью (RF-DETR ONNX) и удаляет содержимое внутри зон через redaction (PyMuPDF), затем дополнительно сжимает результат (pikepdf).

## Важно про токен
Вы опубликовали BOT_TOKEN в исходнике в сообщении. Это компрометация ключа.
Рекомендуется **сразу отозвать/пересоздать токен** в BotFather и хранить его только в переменных окружения.

## Структура проекта

```text
.
├── bot_pdf_cleaner.py          # entrypoint (совместимость со старым запуском)
├── rfdetr.py                   # совместимость: re-export RFDetrONNX
├── pdf_cleaner_bot/
│   ├── config.py               # конфиг (env)
│   ├── logging_setup.py
│   ├── bot/
│   │   ├── app.py              # wiring aiogram
│   │   └── handlers.py         # handlers
│   ├── processor/
│   │   ├── region_processor.py # основная обработка PDF
│   │   └── shrink.py           # pikepdf compress
│   └── ml/
│       └── rfdetr_onnx.py      # ONNX wrapper
├── weights.onnx                # веса (как и раньше)
├── requirements.txt
└── Dockerfile
```

## Запуск через Docker

Сборка:

```bash
docker build -t pdf-cleaner-bot:latest .
```

Запуск (пример):

```bash
docker run -d       --name pdf-cleaner-bot       --restart always       -e BOT_TOKEN="PASTE_YOUR_BOT_TOKEN_HERE"       -v /opt/pdf_bot_data:/app/data       -v /opt/pdf_bot_logs:/app/logs       --memory=5g       --memory-swap=5g       pdf-cleaner-bot:latest
```

## Будущее расширение (вебка)

Архитектура уже разделяет:
- *core processing* (pdf_cleaner_bot/processor) — можно вызывать из HTTP API;
- *transport layer* (pdf_cleaner_bot/bot) — Telegram;
- *config* — один источник правды для обоих интерфейсов.

Для веб-слоя логично добавить `pdf_cleaner_bot/web/` (FastAPI) и переиспользовать `PDFRegionProcessor`.
