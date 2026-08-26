============================================================
# FinSight-RAG
============================================================

Financial document Retrieval-Augmented Generation (RAG) project.
The project currently uses `src/` as its only source directory.

## Current Status

Stage 1 foundation is partially complete:

- SEC EDGAR 10-K download is available.
- PDF text extraction with PyMuPDF is available.
- Page-aware chunk creation is available.
- Embeddings use `BAAI/bge-small-en-v1.5`.
- FAISS index and aligned metadata are persisted in `vector_db/`.
- Metadata-aware retrieval supports company, fiscal year, and filing type.
- FastAPI endpoints are available in `src/api.py`.
- Existing Ollama answer generation remains available in `src/generation/llm.py`.

The project is not production-ready yet. Reranking, OpenAI generation,
evaluation, authentication, observability, deployment configuration, and
complete automated test coverage are still planned.

## Architecture

```text
User
	|
	v
FastAPI (`src/api.py`)
	|
	+--> PDF upload
	|      |
	|      v
	|   SEC ingestion / PyMuPDF
	|      |
	|      v
	|   Page-aware chunking
	|      |
	|      v
	|   Sentence-transformer embeddings
	|      |
	|      v
	|   FAISS + metadata (`vector_db/`)
	|
	+--> Financial question
				 |
				 v
			Query parser + FAISS retrieval
				 |
				 v
			Retrieved chunks and citations
```

## Source Structure

```text
src/
├── api.py                         FastAPI application and endpoints
├── rag_pipeline.py                Command-line RAG orchestration
├── ingestion/
│   └── sec_ingestion.py           SEC download and PDF extraction
├── processing/
│   ├── parser.py                  SEC HTML parsing
│   └── chunker.py                 Text and page-aware chunking
├── embeddings/
│   └── embedder.py                Embeddings and FAISS persistence
├── retrieval/
│   ├── query_parser.py            Company, year, and metric detection
│   ├── retriever.py               FAISS retrieval and metadata filtering
│   └── reranker.py                Reserved for the next retrieval stage
└── generation/
		└── llm.py                     Existing local Ollama generation
```

## Setup

Use Python 3.12 or newer and the project virtual environment.

```powershell
\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The embedding model is downloaded automatically on first use. Ollama is
required only when using the current local answer-generation path.

## Build The FAISS Index

The existing chunk files under `data/chunks/` are used by the embedder.

```powershell
$env:PYTHONPATH="."
python -m src.embeddings.embedder
```

This creates local files under `vector_db/`. They are generated artifacts and
are intentionally ignored by Git.

## Run The API

```powershell
$env:PYTHONPATH="."
uvicorn src.api:app --reload
```

Open the interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

### Available Endpoints

`GET /health`

Returns the service status.

`POST /upload`

Accepts a PDF multipart upload with these form fields:

- `file`
- `company`
- `document_type`
- `fiscal_year`
- `quarter`

The PDF is extracted, split into page-aware chunks, embedded, and stored in
FAISS.

`POST /query`

Example request:

```json
{
	"question": "What was Apple's net income in 2024?",
	"company": "Apple",
	"fiscal_year": "2024",
	"top_k": 5
}
```

The response contains an answer status, retrieved text chunks, metadata, and
page citations. Grounded LLM synthesis is planned for a later stage.

## Existing SEC Download Flow

To download the latest 10-K for a supported company:

```powershell
$env:PYTHONPATH="."
python -m src.ingestion.sec_ingestion --company apple
```

Supported companies are Apple, Microsoft, NVIDIA, Tesla, and Amazon.

## Validation

The current source modules have been compiled, the FastAPI health endpoint
has been smoke-tested, and foundation tests cover PDF extraction, page-aware
chunking, and FAISS metadata persistence. A complete end-to-end test suite is
still to be added.

## Planned Stages

1. Finish and test the Stage 1 foundation.
2. Add cross-encoder reranking: FAISS top 20 to best 5.
3. Add multi-company comparison workflows.
4. Add grounded OpenAI answer generation and citation prompts.
5. Add a 30-question evaluation benchmark and retrieval metrics.
6. Add production concerns such as structured logging, middleware, Docker,
	 health checks, integration tests, and deployment documentation.
