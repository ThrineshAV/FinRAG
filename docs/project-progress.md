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
4. Searches the FAISS index.
5. Oversamples candidates before filtering.
6. Applies metadata filters such as ticker, fiscal year, filing type, company,
   document type, and quarter when values are available.
7. Returns text, scores, chunk identifiers, and metadata.

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

- A retrieval status message
- Retrieved text chunks
- Metadata for each chunk
- Citation records with chunk ID, page number, source, and score

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

Retrieval now reranks metadata-filtered FAISS candidates with the
`cross-encoder/ms-marco-MiniLM-L-6-v2` model. Financial metric matches receive
an additional relevance boost. The cross-encoder loads lazily, and reranking
uses the flattened dictionary records produced by the FAISS store.

### 3.9 Tests

File: `tests/test_stage1_foundation.py`

The current tests verify:

- PDF extraction preserves page numbers.
- PDF chunks preserve required metadata.
- FAISS index and metadata can be written and loaded together.
- FAISS records can be reranked with cross-encoder and metric scores.
- Queries can detect multiple companies for comparison retrieval.
- Grounded OpenAI request formatting is covered with a mocked provider.
- Retrieval hit rate and mean reciprocal rank are covered with deterministic
   test data.

The latest test run passed:

```text
6 passed

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

- Full upload-to-query integration test.
- Persistent FAISS update behavior for multiple uploads instead of rebuilding
  the complete local index each time.
- Complete API filter support for every upload metadata field.
- Cross-encoder reranking connected to the FAISS retriever.
- Replacement of the placeholder retrieval response with grounded OpenAI
  answer generation.
- Streaming-ready answer generation.
- Multi-company comparison workflows.
- Benchmark dataset and retrieval/faithfulness evaluation.
- Structured logging and exception middleware.
- Authentication, rate limiting, and request limits.
- Docker and deployment configuration.
- Complete unit and integration test coverage.

## 8. Next Planned Stage

The next planned feature is Stage 2: retrieval quality improvement.

Target flow:

```text
Question
   |
   v
Embedding
   |
   v
FAISS top 20 candidates
   |
   v
BAAI/bge-reranker-base
   |
   v
Best 5 chunks
   |
   v
Grounded answer generation
```

Stage 2 should add a FAISS-compatible reranker interface, configurable
candidate and final-result counts, reranking scores, confidence information,
and tests proving that the reranker changes ordering when semantic relevance
improves.

## 9. Design Decisions To Remember

- `src/` is the only application source directory. Do not create duplicate
  `app/` or `rag/` implementations.
- `sec_ingestion.py` owns both SEC download support and PDF extraction support.
- Page numbers belong to chunk metadata, not only to the original document.
- FAISS stores vectors; the JSON sidecar stores the aligned searchable metadata.
- Generated vector files belong in `vector_db/` and should not be committed.
- Stage 1 should remain independently understandable before adding reranking,
  LLM generation, evaluation, or deployment complexity.


