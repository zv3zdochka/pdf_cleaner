FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV STORAGE_DIR="/app/storage" \
    WEB_HOST="0.0.0.0" \
    WEB_PORT="8000" \
    WEB_TOKEN="" \
    STORAGE_MAX_BYTES="32212254720"

EXPOSE 8000

CMD ["bash", "-c", "uvicorn app.main:app --host ${WEB_HOST} --port ${WEB_PORT}"]
