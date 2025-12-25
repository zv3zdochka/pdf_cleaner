FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY pdf_cleaner_bot ./pdf_cleaner_bot
COPY bot_pdf_cleaner.py ./bot_pdf_cleaner.py
COPY rfdetr.py ./rfdetr.py

# Model weights (keep as before: same relative path "weights.onnx")
COPY weights.onnx ./weights.onnx

# Folders for temporary files and logs
RUN mkdir -p /app/data && mkdir -p /app/logs

RUN mkdir -p /app/data && mkdir -p /app/logs && mkdir -p /app/storage


VOLUME ["/app/storage"]

VOLUME ["/app/data"]
VOLUME ["/app/logs"]

# The bot token must be provided at runtime:
#   docker run -e BOT_TOKEN="..." ...
ENV BOT_TOKEN=""

CMD ["bash", "-c", "python -u bot_pdf_cleaner.py 2>&1 | tee /app/logs/bot.log"]
