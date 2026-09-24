FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    ENABLE_RERANKING=false \
    RATE_LIMIT_QUERY=20/minute \
    RATE_LIMIT_UPLOAD=5/minute \
    MAX_UPLOAD_SIZE_MB=25

RUN groupadd -r appgroup && useradd -r -g appgroup appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY src ./src

# Create directories for persistent data
RUN mkdir -p /app/vector_db /app/data

EXPOSE 8000
USER appuser

CMD ["uvicorn", "financial_rag.main:app", "--host", "0.0.0.0", "--port", "8000"]