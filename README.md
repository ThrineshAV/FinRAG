============================================================
# FinSight-RAG
============================================================

Financial document Retrieval-Augmented Generation (RAG) project.
The project currently uses `src/` as its only source directory.

## Current Status

Stage 1 foundation is complete:

- SEC EDGAR 10-K download is available.
- PDF text extraction with PyMuPDF is available.
- Page-aware chunk creation is available.
- Embeddings use `BAAI/bge-small-en-v1.5`.
- FAISS index and aligned metadata are persisted in `vector_db/`.
- Metadata-aware retrieval supports company, fiscal year, filing type,
  document type, and quarter.
- Comparison queries can retrieve multiple supported companies together.
- FastAPI endpoints are available in `src/api.py`.
- Grounded OpenAI answer generation is available when `OPENAI_API_KEY` is set.
- A deterministic retrieval benchmark with 30 cases reports hit rate and MRR.
- API responses include request IDs and processing-time headers for tracing.
- A global exception handler returns structured JSON errors for unhandled
  exceptions.
- Existing Ollama answer generation remains available in `src/generation/llm.py`.
- Docker deployment configuration with optional reranking is available.
- Upload-to-query integration test verifies the full pipeline.

Streaming answer generation, multi-company comparison workflows,
authentication, and rate limiting are planned for future stages.

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
│   └── reranker.py                Cross-encoder and metric-aware reranking
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
required only when using the local CLI answer-generation path. Set
`OPENAI_API_KEY` to enable grounded answers from the `/query` endpoint; without
it, the endpoint returns retrieved evidence as before.

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

Returns the service status. API responses include `X-Request-ID` and
`X-Process-Time-Ms` headers.

`GET /ready`

Checks that the FAISS vector store is available. Returns `503` until the
index and metadata are ready.

`POST /upload`

Accepts a PDF multipart upload with these form fields:

- `file`
- `company`
- `document_type`
- `fiscal_year`
- `quarter`

The PDF is extracted, split into page-aware chunks, embedded, and appended to
FAISS. Existing indexed documents are preserved.

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

The response contains a grounded answer when `OPENAI_API_KEY` is configured;
otherwise it returns an evidence status, reranked text chunks, metadata, and
page citations.

## Evaluate Retrieval

The benchmark runner reads cases from `data/evaluation.json` and reports hit
rate, mean reciprocal rank, and average retrieved results:

```powershell
$env:PYTHONPATH="."
python -m src.evaluation.benchmark
```

Replace the starter cases' `relevant_chunk_ids` with verified IDs from the
indexed corpus before using the reported scores for comparison.

## Run With Docker

Build and run the API container:

```powershell
docker build -t finsight-rag .
docker run --rm -p 8000:8000 --env-file .env finsight-rag
```

The image intentionally excludes local FAISS artifacts. Mount a prepared
`vector_db` directory at `/app/vector_db` when serving indexed documents:

```powershell
docker run --rm -p 8000:8000 --env-file .env -v "${PWD}\vector_db:/app/vector_db" finsight-rag
```

Docker disables cross-encoder reranking by default to reduce memory use. Set
`-e ENABLE_RERANKING=true` when the Docker engine has enough memory for both
embedding and reranking models.

Check `/health` for process liveness and `/ready` for vector-store readiness.

## Existing SEC Download Flow

To download the latest 10-K for a supported company:

```powershell
$env:PYTHONPATH="."
python -m src.ingestion.sec_ingestion --company apple
```

Supported companies are Apple, Microsoft, NVIDIA, Tesla, and Amazon.

## Validation

The current source modules have been compiled, and tests cover PDF extraction,
page-aware chunking, FAISS metadata persistence, reranking, grounded request
formatting, API readiness, query fallback/error handling, and upload
validation.

## Planned Stages

1. Finish and test the Stage 1 foundation.
2. Curate the full 30-question evaluation benchmark.
3. Add Docker, integration tests, and deployment documentation.
