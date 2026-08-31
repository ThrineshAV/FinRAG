FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    ENABLE_RERANKING=false \
    RATE_LIMIT_QUERY=20/minute \
    RATE_LIMIT_UPLOAD=5/minute \
    MAX_UPLOAD_SIZE_MB=25 \
    AUTH_REQUIRED=true

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data/evaluation.json ./data/evaluation.json

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]