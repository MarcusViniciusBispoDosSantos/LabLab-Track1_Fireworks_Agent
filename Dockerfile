FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DISABLE_DOTENV=1 \
    MODEL_ORDER_STRATEGY=hybrid \
    BATCH_SOLVE=1 \
    BATCH_CANDIDATES=1 \
    BATCH_MAX_TOKENS=9000 \
    ENSEMBLE_MODE=hard \
    ENABLE_LOCAL_FAST_PATHS=1 \
    REPAIR_ENABLED=1 \
    MAX_WORKERS=1 \
    MAX_RETRIES=4 \
    HARD_ENSEMBLE_SIZE=2 \
    REQUEST_TIMEOUT_SECONDS=40

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src

CMD ["python", "-m", "src.main"]
