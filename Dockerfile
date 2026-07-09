FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DISABLE_DOTENV=1 \
    ENABLE_LOCAL_FAST_PATHS=0 \
    ENABLE_CATEGORY_CALIBRATION=1 \
    CALIBRATION_MODEL_LIMIT=4 \
    SELF_CHECK_MODE=all \
    MAX_WORKERS=2 \
    MAX_RETRIES=6 \
    REQUEST_TIMEOUT_SECONDS=25

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src

CMD ["python", "-m", "src.main"]
