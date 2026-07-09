FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    INPUT_PATH=/input/tasks.json \
    OUTPUT_PATH=/output/results.json \
    MAX_WORKERS=2 \
    REQUEST_TIMEOUT_SECONDS=28 \
    MAX_RETRIES=3 \
    ACCURACY_STRATEGY=ensemble_hard \
    ENABLE_LOCAL_FAST_PATHS=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

RUN python -m py_compile src/*.py

CMD ["python", "-m", "src.main"]
