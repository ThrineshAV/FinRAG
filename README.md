============================================================
# FinSight-RAG
============================================================

[![Test](https://github.com/ThrineshAV/financial-rag/actions/workflows/test.yml/badge.svg)](https://github.com/ThrineshAV/financial-rag/actions/workflows/test.yml)
[![Docker](https://github.com/ThrineshAV/financial-rag/actions/workflows/docker.yml/badge.svg)](https://github.com/ThrineshAV/financial-rag/actions/workflows/docker.yml)
[![Security](https://github.com/ThrineshAV/financial-rag/actions/workflows/security.yml/badge.svg)](https://github.com/ThrineshAV/financial-rag/actions/workflows/security.yml)

Financial document Retrieval-Augmented Generation (RAG) project.
The project currently uses `src/` as its only source directory.

## Current Status

Stages 1–6 are complete:

- SEC EDGAR 10-K download with automatic retry logic
- PDF text extraction with PyMuPDF
- Page-aware chunk creation
- Embeddings use `BAAI/bge-small-en-v1.5`
- FAISS index and aligned metadata persisted in `vector_db/`
- Metadata-aware retrieval supports company, fiscal year, filing type,
  document type, and quarter
- Comparison queries can retrieve multiple supported companies together
- Cross-encoder reranking with financial metric boosting
- FastAPI endpoints with rate limiting and upload size validation
- Grounded OpenAI answer generation (streaming and non-streaming)
- Answer quality evaluation (faithfulness and relevance metrics)
- Comparison-aware generation prompts for multi-company queries
- A deterministic retrieval benchmark with 30 verified cases
- API responses include request IDs and processing-time headers
- Robust error handling for SEC ingestion and HTML parsing
- Docker deployment configuration with configurable limits
- API key authentication via `X-API-Key` header
- Role-based access control (reader, admin)
- Admin endpoints for API key management
- GitHub Actions CI/CD pipeline with automated testing
- Docker image building and publishing to GitHub Container Registry
- Security scanning with Trivy and Bandit
- Test coverage reporting with pytest-cov

Advanced performance optimization and multi-tenant support are planned for future stages.

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

## Configuration

The API supports the following environment variables:

### Authentication
- `AUTH_REQUIRED` — Enable API key authentication (default: `true`). Set to `false` for local development or backward compatibility.
- `ADMIN_API_KEY` — Optional bootstrap admin key for initial setup. If not set, generate one with `python -m src.auth.api_keys`.
- `API_KEYS_FILE` — Path to API keys storage file (default: `data/api_keys.json`)

**Note:** Admin endpoints (`/admin/*`) always require authentication, even when `AUTH_REQUIRED` is `false`.

### Rate Limiting
- `RATE_LIMIT_QUERY` — Rate limit for `/query` and `/query/stream` endpoints (default: `20/minute`)
- `RATE_LIMIT_UPLOAD` — Rate limit for `/upload` endpoint (default: `5/minute`)

### Upload Limits
- `MAX_UPLOAD_SIZE_MB` — Maximum PDF upload size in megabytes (default: `25`)

### Retry Behavior
SEC ingestion automatically retries failed requests up to 3 times with exponential backoff (2–10 seconds between attempts). This applies to both filing metadata requests and document downloads.

### OpenAI Generation
- `OPENAI_API_KEY` — API key for grounded answer generation
- `OPENAI_MODEL` — Model to use (default: `gpt-4o-mini`)
- `OPENAI_TEMPERATURE` — Temperature for generation (default: `0.1`)
- `OPENAI_MAX_TOKENS` — Maximum tokens for generation (optional)

### Reranking
- `ENABLE_RERANKING` — Enable cross-encoder reranking (default: `true` locally, `false` in Docker)
- `RERANKER_MODEL` — Cross-encoder model name (default: `BAAI/bge-reranker-base`)

### Caching
- `CACHE_ENABLED` — Enable caching system (default: `true`)
- `CACHE_TTL_SECONDS` — Time-to-live for cached items in seconds (default: `3600`)
- `REDIS_URL` — Redis connection URL (default: `redis://localhost:6379/0`)
- `CACHE_EMBEDDING_TTL_SECONDS` — TTL for query embedding cache (default: `3600`)

The caching system uses Redis when available and falls back to an in-memory cache when Redis is not accessible. This allows the application to run in development environments without external dependencies while providing performance benefits in production with Redis.

Two levels of caching are implemented:
1. **Query Result Caching**: Complete responses from `/query` and `/query/stream` endpoints are cached based on the question, filters, and OpenAI configuration. Cache hits return the previous response with an `X-Cache: HIT` header.
2. **Query Embedding Caching**: Embeddings generated for search queries are cached to avoid recomputation of identical queries. This reduces the computational load on the embedding model.

Cache statistics are tracked internally and can be extended with monitoring endpoints in future versions.

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

## Authentication

The API uses API key authentication via the `X-API-Key` header. Authentication is enabled by default.

### Generate an Admin Key

```powershell
python -m src.auth.api_keys
```

This prints a new admin key. Copy it — it's shown only once.

Alternatively, set `ADMIN_API_KEY` in your environment:

```powershell
$env:ADMIN_API_KEY="fsr_your_key_here"
python -m src.auth.api_keys
```

### Create Additional Keys

Use the admin key to create reader or admin keys via the API:

```bash
curl -X POST http://127.0.0.1:8000/admin/keys \
  -H "X-API-Key: YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "data-science-team", "role": "reader"}'
```

### Roles

- **`reader`** — Can query documents via `/query` and `/query/stream`
- **`admin`** — Can query, upload documents, and manage API keys

### Disable Authentication (Local Development)

```powershell
$env:AUTH_REQUIRED="false"
uvicorn src.api:app --reload
```

**Note:** Admin endpoints always require authentication, even when `AUTH_REQUIRED=false`.

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

`POST /query/stream`

Streams answers as Server-Sent Events (SSE). Requires `OPENAI_API_KEY`. Same request format as `/query`.

### Admin Endpoints (Require Admin Key)

`POST /admin/keys`

Create a new API key. Request body:

```json
{
  "name": "team-name",
  "role": "reader"
}
```

Returns the raw key (shown once) and key metadata.

`GET /admin/keys`

List all API keys (no secret material).

`DELETE /admin/keys/{key_id}`

Revoke an API key by its ID.

## Evaluate Retrieval

The benchmark runner reads cases from `data/evaluation.json` and reports hit
rate, mean reciprocal rank, and average retrieved results:

```powershell
$env:PYTHONPATH="."
python -m src.evaluation.benchmark
```

Replace the starter cases' `relevant_chunk_ids` with verified IDs from the
indexed corpus before using the reported scores for comparison.

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment:

### Automated Workflows

- **Test** — Runs on every push and PR. Executes all 83 tests with coverage reporting.
- **Docker** — Builds and pushes Docker images to GitHub Container Registry on tags and main branch.
- **Security** — Scans dependencies and Docker images for vulnerabilities using Trivy and Bandit.
- **Deploy** — Template workflow for cloud deployment (AWS/GCP/Azure).

### Running Tests Locally

```powershell
$env:PYTHONPATH="."
pytest --cov=src --cov-report=term-missing
```

### Building Docker Images

```powershell
docker build -t finsight-rag:local .
```

### Deployment

The deploy workflow provides templates for:
- AWS ECS
- Google Cloud Run
- Azure Container Instances
- SSH-based VPS deployment

Configure GitHub Environments and secrets in your repository settings before deploying.

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
