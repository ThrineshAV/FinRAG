.PHONY: run test ingest query build clean

run:
	uvicorn src.financial_rag.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/

ingest:
	python -m src.financial_rag.application.ingestion

query:
	python -m src.financial_rag.application.retrieval

build:
	docker build -t finsight .

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
