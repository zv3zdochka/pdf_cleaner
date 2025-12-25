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

docker build -t pdf-cleaner-bot:latest .

docker run -d \
  --name pdf-cleaner-bot \
  --restart always \
  -e BOT_TOKEN="..." \
  -e STORAGE_DIR="/app/storage" \
  -e STORAGE_MAX_BYTES="32212254720" \
  -e STORAGE_MAX_AGE_DAYS="0" \
  -v pdf_storage:/app/storage \
  -v pdf_logs:/app/logs \
  -v pdf_work:/app/data \
  pdf-cleaner-bot:latest
