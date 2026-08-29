# FinSight-RAG Project Progress

## 1. Project Purpose

FinSight-RAG is a financial document question-answering system. It is designed
for questions about annual reports, SEC filings, and other financial documents.
The system retrieves relevant document passages first, then uses those passages
to support an answer.

Example question:

> What was Apple's net income in fiscal year 2024?

The important RAG principle is that the answer should be based on retrieved
financial evidence instead of the language model's general memory.

## 2. Current Architecture

The project uses `src/` as its only source directory.

```text
SEC filing or uploaded PDF
          |
          v
src/ingestion/sec_ingestion.py
          |
          v
src/processing/chunker.py
          |
          v
src/embeddings/embedder.py
          |
          v
FAISS index + metadata in vector_db/
          |
          v
src/retrieval/query_parser.py
          |
          v
src/retrieval/retriever.py
          |
          v
Retrieved text, scores, metadata, and citations
          |
          v
src/api.py or src/rag_pipeline.py
```

## 3. What Has Been Implemented

### 3.1 SEC ingestion

File: `src/ingestion/sec_ingestion.py`

The existing SEC ingestion module:

- Stores company information for Apple, Microsoft, NVIDIA, Tesla, and Amazon.
- Requests filing information from SEC EDGAR.
- Finds the latest 10-K filing.
- Downloads the filing HTML and associated metadata.
- Extracts text from uploaded PDFs with PyMuPDF.
- Preserves one-based PDF page numbers.
- Rejects missing, invalid, or text-free PDFs.

The SEC download path and PDF upload path are different inputs, but both are
part of the same ingestion area. No duplicate ingestion module was added.

### 3.2 Document processing and chunking

File: `src/processing/chunker.py`

The existing chunker uses LangChain's recursive text splitter for the original
processed SEC documents. It was extended with `create_page_chunks()` for PDFs.

Each PDF chunk contains:

- `chunk_id`
- `chunk_index`
- `text`
- `company`
- `document_type`
- `fiscal_year`
- `quarter`
- `page_number`

The page number is preserved so later answers can cite the page containing the
retrieved evidence.

### 3.3 Embeddings and FAISS

File: `src/embeddings/embedder.py`

The embedding module was changed from the original local Qdrant implementation
to FAISS for the current architecture.

It now:

- Loads `BAAI/bge-small-en-v1.5` lazily.
- Loads structured chunks from `data/chunks/`.
- Generates normalized embeddings.
- Builds a cosine-similarity-compatible FAISS `IndexFlatIP` index.
- Stores the index at `vector_db/financial_documents.faiss`.
- Stores aligned chunk metadata at `vector_db/financial_documents.json`.
- Checks that the vector count and metadata count match when loading.

The files in `vector_db/` are generated local artifacts and are ignored by
Git through `.gitignore`.

### 3.4 Query parsing

File: `src/retrieval/query_parser.py`

The parser identifies common query fields:

- Company name or ticker
- Fiscal year
- Filing type
- Financial metric, such as revenue, net income, or gross profit

For example, a question mentioning `Apple`, `2024`, and `net income` is
converted into structured values that retrieval can use.

### 3.5 Retrieval

File: `src/retrieval/retriever.py`

The retriever now:

1. Validates the question and `top_k` value.
2. Parses the question.
3. Embeds the question.
4. Searches the FAISS index with a configurable candidate count
   (default: `top_k * 4`, clamped to `[top_k, index.ntotal]`).
5. Applies metadata filters such as ticker, fiscal year, filing type, company,
   document type, and quarter when values are available.
6. Reranks filtered candidates (when enabled) and returns text, scores,
   chunk identifiers, reranking scores, confidence, and metadata.

The query parser preserves the first detected company for backward
compatibility and also exposes all detected company names and tickers. FAISS
filtering accepts the ticker list, allowing comparison questions to retrieve
records for multiple supported companies.

FAISS itself does not provide payload filtering like Qdrant. Therefore, the
current Stage 1 implementation searches candidates and applies metadata
filters to the returned records.

### 3.6 FastAPI API

File: `src/api.py`

The API currently exposes:

#### `GET /health`

Returns:

```json
{"status": "ok"}
```

#### `POST /upload`

Accepts a PDF multipart upload and these fields:

- `file`
- `company`
- `document_type`
- `fiscal_year`
- `quarter`

The endpoint runs PDF extraction, page-aware chunking, embedding, and FAISS
persistence. Uploaded chunks append to the existing FAISS store, preserving
previously indexed documents.

#### `POST /query`

Accepts a validated request such as:

```json
{
  "question": "What was Apple's net income in 2024?",
  "company": "Apple",
  "fiscal_year": "2024",
  "top_k": 5
}
```

Returns:

- A retrieval status message (or grounded LLM answer when OpenAI is configured)
- Retrieved text chunks
- Metadata for each chunk
- Citation records with chunk ID, page number, source, score, and
  optional reranking fields (`rerank_score`, `cross_encoder_score`,
  `metric_score`, `confidence`)

The current response intentionally reports retrieved evidence. Full grounded
LLM answer synthesis is planned for a later stage.

### 3.7 Command-line pipeline

File: `src/rag_pipeline.py`

The command-line pipeline connects retrieval to the existing local Ollama
answer generator. It:

1. Receives a financial question.
2. Retrieves relevant FAISS chunks.
3. Builds a context string.
4. Sends the context to the Ollama generator.
5. Returns an answer and source information.

The API currently focuses on retrieval and citations. The command-line path
still uses the existing Ollama generation module.

### 3.8 Reranking

File: `src/retrieval/reranker.py`

Retrieval now reranks metadata-filtered FAISS candidates with `BAAI/bge-reranker-base`
(upgraded from `cross-encoder/ms-marco-MiniLM-L-6-v2` in Stage 2). The model name
is configurable via the `RERANKER_MODEL` environment variable. Financial metric
matches receive an additional relevance boost. The cross-encoder loads lazily, and
reranking uses the flattened dictionary records produced by the FAISS store.

Each reranked result now carries:

- `rerank_score` — combined cross-encoder + metric boost score
- `cross_encoder_score` — raw cross-encoder output
- `metric_score` — financial metric relevance (0.0 or 1.0)
- `confidence` — min-max normalized score within the result set (0.0–1.0)

### 3.9 Tests

File: `tests/test_stage1_foundation.py`, `tests/test_reranking.py`,
`tests/test_edge_cases.py`, `tests/test_api_integration.py`

The current tests verify:

- PDF extraction preserves page numbers.
- PDF chunks preserve required metadata.
- FAISS index and metadata can be written and loaded together.
- FAISS records can be reranked with cross-encoder and metric scores.
- Queries can detect multiple companies for comparison retrieval.
- Grounded OpenAI request formatting is covered with a mocked provider.
- Retrieval hit rate and mean reciprocal rank are covered with deterministic
   test data.
- **(Stage 2)** Reranker reorders documents by semantic relevance.
- **(Stage 2)** Metric boost promotes matching documents when cross-encoder
   scores are equal.
- **(Stage 2)** Confidence scores are normalized to 0.0–1.0.
- **(Stage 2)** Single-result confidence is 1.0.
- **(Stage 2)** Reranker respects `top_k`.
- **(Stage 2)** Default model is `BAAI/bge-reranker-base` and is configurable.

The latest test run passed:

```text
45 passed

### 3.10 Evaluation

Files: `src/evaluation/metrics.py`, `src/evaluation/benchmark.py`,
`data/evaluation.json`

The project now includes a benchmark runner that calculates hit rate, mean
reciprocal rank, and average retrieved results. The included JSON cases are a
starter template and require verified `relevant_chunk_ids` before production
evaluation.

### 3.11 API observability

File: `src/api.py`

The API middleware now preserves or generates an `X-Request-ID`, adds an
`X-Process-Time-Ms` response header, and emits structured request log fields
for method, path, status, request ID, and duration.

The `/ready` endpoint separately verifies that the FAISS vector store can be
loaded, returning `503` when query dependencies are unavailable.

API integration tests cover health and readiness responses, query fallback and
missing-index errors, citation formatting, and invalid upload rejection.

### 3.12 Containerization

Files: `Dockerfile`, `.dockerignore`

The API can run in a Python 3.12 container. Large local artifacts and secrets
are excluded from the image; a prepared `vector_db` directory must be mounted
at runtime for query readiness. Cross-encoder reranking is disabled by default
in Docker to avoid out-of-memory kills; it remains enabled by default locally.
```

### 3.13 Stage 2 — Retrieval quality improvement

Files: `src/retrieval/reranker.py`, `src/retrieval/retriever.py`,
`src/api.py`, `tests/test_reranking.py`

Stage 2 upgraded retrieval quality across four areas:

1. **Reranker model upgrade** — Switched from `cross-encoder/ms-marco-MiniLM-L-6-v2`
   to `BAAI/bge-reranker-base`, which provides stronger relevance judgments for
   domain-specific financial text. The model name is now configurable via the
   `RERANKER_MODEL` environment variable.

2. **Configurable candidate count** — `retrieve_documents()` accepts an optional
   `candidate_count` parameter. The default oversampling was reduced from
   `top_k * 10` to `top_k * 4`, which is sufficient headroom for metadata
   filtering without wasting search time on large indices. The count is clamped
   to `[top_k, index.ntotal]`.

3. **Confidence signals** — Each reranked result now includes a `confidence`
   field (0.0–1.0) calculated via min-max normalization of the rerank scores
   within the result set. The API `Citation` model surfaces `rerank_score`,
   `cross_encoder_score`, `metric_score`, and `confidence` as optional fields.

4. **Reranking tests** — Seven new tests in `tests/test_reranking.py` prove
   that the reranker changes ordering when semantic relevance differs, that
   metric boosts promote matching documents, that confidence is correctly
   normalized, and that the model name is configurable.

## 4. Git Milestones

### `2e7d031` - Initial FinSight-RAG pipeline

Created the initial financial RAG pipeline with SEC ingestion, HTML parsing,
chunking, embeddings, Qdrant storage, query parsing, reranking, and Ollama
generation components.

### `387f33b` - Metadata-aware retrieval

Added metadata filtering for company-related retrieval fields such as ticker,
fiscal year, and filing type.

### `a2f8ef3` - Metric-aware reranking

Added metric relevance scoring and reranking support for financial terms.

### `aef47f4` - FAISS retrieval API and project documentation

Moved the active vector storage path to FAISS, added the FastAPI API, added
PDF/page-aware support, updated `.gitignore`, and documented the project.

### `7433d2c` - Stage 1 foundation coverage

Added foundation tests and updated the README validation section.

Current branch status at the time of writing:

```text
main -> origin/main
```

## 5. Current Dependencies

The main dependencies used by the current implementation are:

- `fastapi` for the HTTP API
- `uvicorn` for serving FastAPI
- `python-multipart` for PDF form uploads
- `PyMuPDF` for PDF text extraction
- `sentence-transformers` for embeddings
- `faiss-cpu` for local vector search
- `langchain-text-splitters` for existing SEC text chunking
- `beautifulsoup4` and `lxml` for SEC HTML parsing
- `requests` for SEC and Ollama HTTP calls
- `pydantic` for API validation
- `pytest` for tests

## 6. How To Run The Current Code

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Build the FAISS index from existing chunks:

```powershell
$env:PYTHONPATH="."
python -m src.embeddings.embedder
```

Start the API:

```powershell
$env:PYTHONPATH="."
uvicorn src.api:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Run tests:

```powershell
$env:PYTHONPATH="."
pytest -q
```

Download a supported SEC 10-K:

```powershell
$env:PYTHONPATH="."
python -m src.ingestion.sec_ingestion --company apple
```

## 7. What Is Not Complete Yet

The project is a functional foundation, not a finished production service.
The main gaps are:

- Streaming-ready answer generation.
- Multi-company comparison workflows.
- Benchmark evaluation cases need verified `relevant_chunk_ids` from the
  indexed corpus (30 questions are defined in `data/evaluation.json`).
- Authentication, rate limiting, and request limits.
- Complete unit and integration test coverage.

The following items have been completed since the initial progress log:

- Full upload-to-query integration test (added).
- Persistent FAISS update behavior (append, not rebuild).
- Complete API filter support for every upload metadata field
  (`document_type`, `quarter` wired through).
- Cross-encoder reranking connected to the FAISS retriever.
- Grounded OpenAI answer generation.
- 30-case benchmark dataset defined in `data/evaluation.json`.
- Structured exception handler added to the API.
- Docker and deployment configuration.

## 8. Next Planned Stage

The next planned feature is Stage 3: answer generation and evaluation.

Stage 2 (retrieval quality improvement) is complete. The reranker has been
upgraded to `BAAI/bge-reranker-base`, candidate counts are configurable,
confidence signals are surfaced in the API, and reranking behavior is covered
by 7 dedicated tests.

Stage 3 should focus on:

- Streaming-ready grounded answer generation.
- Evaluation harness improvements with verified `relevant_chunk_ids`.
- Answer quality metrics (faithfulness, relevance).
- Multi-company comparison workflows.

## 9. Design Decisions To Remember

- `src/` is the only application source directory. Do not create duplicate
  `app/` or `rag/` implementations.
- `sec_ingestion.py` owns both SEC download support and PDF extraction support.
- Page numbers belong to chunk metadata, not only to the original document.
- FAISS stores vectors; the JSON sidecar stores the aligned searchable metadata.
- Generated vector files belong in `vector_db/` and should not be committed.
- Stage 1 should remain independently understandable before adding reranking,
  LLM generation, evaluation, or deployment complexity.
