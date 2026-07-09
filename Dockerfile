FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DISABLE_DOTENV=1 \
    ENABLE_LOCAL_FAST_PATHS=1 \
    RETRY_INVALID_OUTPUTS=1 \
    MAX_WORKERS=3 \
    MAX_RETRIES=5 \
    REQUEST_TIMEOUT_SECONDS=26 \
    ENABLE_MODEL_CALIBRATION=1 \
    CALIBRATION_MODEL_LIMIT=4

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src

CMD ["python", "-m", "src.main"]
