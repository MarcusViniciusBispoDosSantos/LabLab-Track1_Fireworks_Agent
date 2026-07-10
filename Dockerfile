FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DISABLE_DOTENV=1 \
    ENABLE_LOCAL_FAST_PATHS=1 \
    HARD_SECOND_TRY=0 \
    MAX_API_TASKS=999 \
    MAX_WORKERS=1 \
    MODEL_ORDER=hybrid
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
CMD ["python", "-m", "src.main"]
