# PDF Cleaner — Telegram Bot + Web Storage UI

This repository contains a PDF cleaning service composed of:

- **Telegram Bot (aiogram v3)** — receives PDFs, optionally deletes specified pages, runs detection/redaction, compresses output, and sends the result back to the user.
- **Web UI (FastAPI)** — provides a lightweight interface to browse stored requests, download input/output PDFs, and delete requests permanently.
- **Shared storage** — both services use the same storage volume so the web UI can manage files produced by the bot.

## Key Features

### Bot
- Accepts **PDF up to ~50 MB** (Telegram bot limit).
- Shows a **file card** after upload with:
  - file name
  - file size
  - total pages
  - received time
  - two inline buttons: **Process** / **Set pages to delete**
- Page deletion input supports:
  - single pages and ranges: `1, 2, 4-6`
- Processes PDF using RF-DETR + PyMuPDF redaction.
- Produces and stores:
  - `input.pdf`
  - `cleaned.pdf`
  - `cleaned_small.pdf` (compressed)
  - `meta.json`
- If `cleaned_small.pdf` is larger than Telegram’s limit:
  - it is **split into multiple PDF parts for sending only**
  - **storage still keeps a single file** (`cleaned_small.pdf`)

### Web UI
- Lists saved requests and basic metadata.
- Download buttons for input and outputs.
- Delete requests permanently (deletes request folder).
- Shows total storage usage vs the configured quota.

### Storage & Retention
- Storage quota is enforced (default **30 GiB**).
- **No automatic time-based deletion** (manual delete via Web UI).
- If quota is exceeded, bot refuses new uploads and may rollback the current request output.

---

## Repository Layout (high level)

Typical structure:

```

.
├── docker-compose.yml
├── Dockerfile                 # bot container
├── bot_pdf_cleaner.py
├── pdf_cleaner_bot/
│   ├── bot/
│   │   ├── app.py
│   │   └── handlers.py
│   ├── storage/
│   │   └── manager.py
│   ├── logging_setup.py
│   └── pdf_utils.py
└── web/
├── Dockerfile             # web container
└── app/...

````

---

## Requirements

- Docker + Docker Compose (v2)
- A Telegram bot token
- `weights.onnx` model file is expected inside the bot container (copied during build)

---

## Quick Start (Recommended)

### 1) Create host folders for persistent data

These folders will be mounted into containers:

```bash
mkdir -p /opt/pdf_bot_storage
mkdir -p /opt/pdf_bot_logs
````

* `/opt/pdf_bot_storage` — **all stored PDFs and metadata**
* `/opt/pdf_bot_logs` — bot logs (rotated)

### 2) Create `.env` in the project root

```env
BOT_TOKEN=YOUR_TELEGRAM_BOT_TOKEN
WEB_TOKEN=optional_web_secret_token
```

* `BOT_TOKEN` is required.
* `WEB_TOKEN` is optional; if set, the web UI will require it.

### 3) Build and run everything

From the project root:

```bash
docker compose up -d --build
```

### 4) Verify services

```bash
docker compose ps
```

### 5) Open Web UI

Default (from `docker-compose.yml`):

* Web UI: `http://SERVER_IP:8081/`

If `WEB_TOKEN` is set, open once with:

```
http://SERVER_IP:8081/?token=WEB_TOKEN
```

After that, your browser session will stay authorized (cookie).

---

## Configuration

All key configuration is done via environment variables in `docker-compose.yml` (and `.env`).

Typical variables:

### Bot

* `BOT_TOKEN` — Telegram bot token (required)
* `MODEL_PATH` — model path inside container (default `/app/weights.onnx`)
* `STORAGE_DIR` — shared storage root (default `/app/storage`)
* `STORAGE_MAX_BYTES` — quota in bytes (default 30 GiB)
* `STORAGE_MAX_AGE_DAYS` — retention TTL (set to `0` to disable)
* `LOGS_DIR` — logs directory (default `/app/logs`)
* `LOG_LEVEL` — logging level (`INFO` recommended)

### Web

* `STORAGE_DIR` — must match bot storage dir mount (default `/app/storage`)
* `WEB_TOKEN` — optional token protection

---

## How to Stop / Restart

### Stop (keep data)

```bash
docker compose down
```

### Restart

```bash
docker compose restart
```

### Rebuild and restart after code changes

```bash
docker compose up -d --build
```

---

## Logs

### Follow combined logs (bot + web)

```bash
docker compose logs -f
```

### Follow only bot logs

```bash
docker compose logs -f bot
# or:
docker logs -f pdf-cleaner-bot
```

### Follow only web logs

```bash
docker compose logs -f web
# or:
docker logs -f pdf-cleaner-web
```

### Bot log files on host

Because `/opt/pdf_bot_logs` is mounted into the bot container:

```bash
ls -lh /opt/pdf_bot_logs
tail -f /opt/pdf_bot_logs/bot.log
```

Search errors:

```bash
grep -E "ERROR|Exception|Traceback" /opt/pdf_bot_logs/bot.log
```

Search by request id:

```bash
grep "request_id=YOUR_ID" /opt/pdf_bot_logs/bot.log
```

---

## Storage: Where files are kept

All saved requests live under the shared storage directory. Typical layout:

```
/opt/pdf_bot_storage/
  users/<user_id>/requests/<request_id>/
    input.pdf
    cleaned.pdf
    cleaned_small.pdf
    meta.json
    input_original.pdf        # only if you used page deletion feature
```

The Web UI reads the same storage and allows download/delete operations.

---

## How to delete ALL accumulated data (full cleanup)

### Option A (recommended): wipe only stored PDFs (keep logs)

```bash
docker compose down
rm -rf /opt/pdf_bot_storage/*
docker compose up -d
```

### Option B: wipe storage + logs completely

```bash
docker compose down
rm -rf /opt/pdf_bot_storage/*
rm -rf /opt/pdf_bot_logs/*
docker compose up -d
```

### Option C: wipe everything including docker images (rarely needed)

```bash
docker compose down --remove-orphans
docker image prune -a
docker volume prune
```

Use Option C only if you know what you’re doing.

---

## Operational Notes

### Telegram file size behavior

* Incoming files: Telegram bot limit is ~50 MB.
* If the processed output is >50 MB:

  * the bot splits it into multiple parts for sending,
  * but keeps a single `cleaned_small.pdf` in storage for the web UI.

### Page deletion behavior

* Page deletion happens **before processing**.
* You can input: `1, 2, 4-6`
* Pages are 1-based.
* If you later cancel deletion, the bot can restore from `input_original.pdf` (if it exists).
