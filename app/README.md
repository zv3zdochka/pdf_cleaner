

todo:grafana

docker build -t pdf-cleaner-bot:latest .

docker run -d \
  --name pdf-cleaner-bot \
  --restart always \
  -v /opt/pdf_bot_data:/app/data \
  -v /opt/pdf_bot_logs:/app/logs \
  --memory=5g \
  --memory-swap=5g \
  pdf-cleaner-bot:latest
