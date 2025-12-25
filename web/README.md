# PDF Cleaner Web UI

Небольшая веб-панель для просмотра и управления файлами, которые сохраняет бот в `STORAGE_DIR`.

## Возможности

- список запросов (входной PDF и результаты);
- просмотр деталей запроса и метаданных;
- скачивание входного и выходных файлов;
- удаление запроса «насовсем» (удаляется папка request целиком).

## Требования

- Папка `web/` должна лежать в корне репозитория рядом с `pdf_cleaner_bot/`.
- Хранилище бота должно быть на диске и монтироваться в вебку (одинаковый `STORAGE_DIR`).

## Запуск локально (внутри того же проекта)

1) Установить зависимости вебки:
```bash
pip install -r web/requirements.txt
```

2) Задать переменные окружения (пример):
```bash
export STORAGE_DIR="storage"
export STORAGE_MAX_BYTES=$((30*1024*1024*1024))
export WEB_USER="admin"
export WEB_PASSWORD="change_me"
export WEB_PORT="8080"
```

3) Запуск:
docker run -d \
  --name pdf-cleaner-web \
  --restart always \
  -e STORAGE_DIR="/app/storage" \
  -e STORAGE_MAX_BYTES="32212254720" \
  -e WEB_USER="admin" \
  -e WEB_PASSWORD="change_me" \
  -p 8080:8080 \
  -v pdf_storage:/app/storage \
  -v "$(pwd)/web:/app/web:ro" \
  -w /app \
  python:3.12-slim \
  sh -c "pip install --no-cache-dir -r /app/web/requirements.txt && uvicorn web.app:app --host 0.0.0.0 --port 8080"
