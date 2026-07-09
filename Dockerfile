FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    INPUT_PATH=/input/tasks.json \
    OUTPUT_PATH=/output/results.json \
    MAX_WORKERS=1 \
    REQUEST_TIMEOUT_SECONDS=29 \
    MAX_RETRIES=4 \
    VERIFY_MODE=hard \
    ENABLE_LOCAL_FAST_PATHS=0 \
    USE_ENSEMBLE_FOR_CODE=0

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

RUN python -m py_compile src/*.py

CMD ["python", "-m", "src.main"]
