"""FastAPI interface for the FinSight-RAG pipeline."""

from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import time
from uuid import uuid4
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.auth.api_keys import create_api_key, list_api_keys, revoke_api_key
from src.auth.dependencies import require_admin, require_api_key, require_upload
from src.auth.models import APIKeyCreate, APIKeyInfo, APIKeyRecord, APIKeyResponse
from src.cache.manager import build_cache_key, get_cache_manager
from src.embeddings.embedder import generate_embeddings, store_embeddings
from src.embeddings.embedder import load_vector_store
from src.ingestion.sec_ingestion import extract_pdf_pages
from src.processing.chunker import create_page_chunks
from src.generation.llm import (
    generate_answer_grounded,
    generate_answer_grounded_stream,
    is_grounded_generation_available,
)
from src.retrieval.retriever import retrieve_documents

logger = logging.getLogger(__name__)

# Rate limiting configuration
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="FinSight-RAG",
    version="1.0.0",
)

# ✅ LOAD .env FILE - READS ENVIRONMENT VARIABLES
load_dotenv()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ✅ ADD CORS MIDDLEWARE - ALLOWS FRONTEND TO COMMUNICATE WITH BACKEND
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development/testing)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers (including X-API-Key)
)

# Custom OpenAPI with security scheme
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="FinSight-RAG",
        version="1.0.0",
        routes=app.routes,
    )

    # Add security scheme for API Key
    openapi_schema["components"]["securitySchemes"] = {
        "APIKeyHeader": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API Key for authentication. Pass via X-API-Key header.",
        }
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

ingestion_lock = asyncio.Lock()


# Configuration
RATE_LIMIT_QUERY = os.getenv("RATE_LIMIT_QUERY", "20/minute")
RATE_LIMIT_UPLOAD = os.getenv("RATE_LIMIT_UPLOAD", "5/minute")
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25")) * 1024 * 1024  # Convert MB to bytes
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
_query_cache_stats = {
    "query_hits": 0,
    "query_misses": 0,
    "stream_hits": 0,
    "stream_misses": 0,
}



@app.middleware("http")
async def request_logging_middleware(request, call_next):
    """Attach a request ID and log request duration for every response."""
    request_id = request.headers.get("X-Request-ID", str(uuid4()))
    started_at = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - started_at) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    logger.info(
        "api_request method=%s path=%s status=%s request_id=%s duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        request_id,
        duration_ms,
    )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Return a structured JSON error for any unhandled exception."""
    request_id = request.headers.get("X-Request-ID", "unknown")
    logger.exception(
        "unhandled_exception request_id=%s path=%s error=%s",
        request_id,
        request.url.path,
        exc,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id},
    )


class QueryRequest(BaseModel):
    """Validated financial question and optional metadata filters."""

    question: str = Field(min_length=3)
    company: str | None = None
    fiscal_year: str | None = None
    filing_type: str | None = None
    document_type: str | None = None
    quarter: str | None = None
    top_k: int = Field(default=5, ge=1, le=50)


class Citation(BaseModel):
    """Citation for one retrieved source chunk."""

    chunk_id: str
    source: str
    page_number: int | str
    score: float
    rerank_score: float | None = None
    cross_encoder_score: float | None = None
    metric_score: float | None = None
    confidence: float | None = None


class QueryResponse(BaseModel):
    """Structured retrieval response."""

    answer: str
    retrieved_chunks: list[str]
    metadata: list[dict[str, Any]]
    citations: list[Citation]


class CacheStats(BaseModel):
    """Cache hit/miss counters for query endpoints."""

    backend: str
    query_hits: int
    query_misses: int
    stream_hits: int
    stream_misses: int
    total_hits: int
    total_misses: int
    hit_rate: float


def _query_cache_key(namespace: str, query_request: QueryRequest, *, openai_enabled: bool) -> str:
    """Build a cache key from the question, filters, and answer mode."""
    payload = {
        "request": query_request.model_dump(mode="json"),
        "openai_enabled": openai_enabled,
        "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    }
    return build_cache_key(namespace, payload)




def _cached_query_response(value: Any) -> QueryResponse | None:
    """Validate a cached query payload without allowing bad data to break requests."""
    if not isinstance(value, dict):
        return None
    try:
        return QueryResponse.model_validate(value)
    except (TypeError, ValueError):
        logger.warning("Ignoring malformed cached query response")
        return None


def _cache_lookup(cache_key: str, namespace: str) -> QueryResponse | None:
    """Look up a cached query response and update the appropriate counters."""
    cached = _cached_query_response(get_cache_manager().get(cache_key))
    stat_prefix = "stream" if namespace == "query_stream" else "query"
    if cached is None:
        _query_cache_stats[f"{stat_prefix}_misses"] += 1
    else:
        _query_cache_stats[f"{stat_prefix}_hits"] += 1
    return cached


class UploadResponse(BaseModel):
    """Result of indexing an uploaded PDF."""

    filename: str
    chunks_indexed: int
    metadata: dict[str, str]


@app.get("/health")
async def health() -> dict[str, str]:
    """Return service liveness."""
    return {"status": "ok"}


@app.get("/ready")
async def readiness() -> dict[str, str]:
    """Report whether the vector store is ready to serve queries."""
    try:
        load_vector_store()
    except (FileNotFoundError, ValueError, OSError) as exc:
        logger.warning("Vector store is not ready: %s", exc)
        raise HTTPException(status_code=503, detail="Vector store is unavailable") from exc
    return {"status": "ready"}


@app.post("/upload")
@limiter.limit(RATE_LIMIT_UPLOAD)
async def upload_pdf(
    request: Request,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(..., description="Multiple PDF files to upload"),
    company: str = Form(...),
    document_type: str = Form(...),
    fiscal_year: str = Form(...),
    quarter: str = Form(...),
    mode: str = Form("auto", description="Processing mode: 'auto', 'sync', or 'async'"),
    _key: APIKeyRecord | None = Depends(require_upload),
):
    """Smart Upload: Extract, chunk, embed, and index financial PDFs.

    Intelligently handles both single and multiple files with sync/async options.

    Request body (multipart/form-data):
    - files: Upload one or more PDF files
    - company: Company name
    - document_type: Document type (10-K, 10-Q, etc.)
    - fiscal_year: Fiscal year (2024, etc.)
    - quarter: Quarter (Q1, Q2, FY, etc.)
    - mode: Processing mode (optional, default: "auto")
      * "auto": 1 file = sync (201), 2+ files = async (202)
      * "sync": Always synchronous, wait for all files (201)
      * "async": Always asynchronous, queue for background (202)

    Returns (Sync Mode - 201):
        List of upload results with filename and chunks indexed for each file
        [
          {"filename": "doc.pdf", "chunks_indexed": 45, "metadata": {...}},
          ...
        ]

    Returns (Async Mode - 202):
        Job status with job_id for tracking
        {"status": "accepted", "job_id": "uuid", "file_count": 3, "mode": "async"}

    HTTP Status:
    - 201 Created (sync mode: all files processed)
    - 202 Accepted (async mode: files queued)

    Example cURL (Auto mode):
        curl -X POST http://127.0.0.1:8000/upload \\
          -F "files=@file1.pdf" \\
          -F "files=@file2.pdf" \\
          -F "company=Apple" \\
          -F "document_type=10-K" \\
          -F "fiscal_year=2024" \\
          -F "quarter=Q1" \\
          -H "X-API-Key: your_key"

    Example cURL (Force async):
        curl -X POST http://127.0.0.1:8000/upload \\
          -F "files=@large_file.pdf" \\
          -F "mode=async" \\
          -F "company=Apple" \\
          -F "document_type=10-K" \\
          -F "fiscal_year=2024" \\
          -F "quarter=Q1" \\
          -H "X-API-Key: your_key"
    """
    _ = (request, _key)

    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="At least one PDF file is required")

    # Validate all files are PDFs
    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' must be a PDF. Only PDF files are supported."
            )

    # Determine processing mode
    if mode == "auto":
        use_async = len(files) > 1
    elif mode == "async":
        use_async = True
    elif mode == "sync":
        use_async = False
    else:
        raise HTTPException(status_code=400, detail="mode must be 'auto', 'sync', or 'async'")

    metadata = {
        "company": company,
        "document_type": document_type,
        "fiscal_year": fiscal_year,
        "quarter": quarter,
    }

    # ASYNC MODE: Queue for background processing
    if use_async:
        job_id = str(uuid4())

        async def _process_async():
            for file in files:
                async with ingestion_lock:
                    await _process_single_file(file, metadata)

        background_tasks.add_task(_process_async)
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "status": "accepted",
                "job_id": job_id,
                "file_count": len(files),
                "mode": "async",
                "message": f"Queued {len(files)} file(s) for background processing. Use job_id to track status."
            }
        )

    # SYNC MODE: Process immediately and return results
    results: list[UploadResponse] = []

    for file in files:
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as temporary_file:
                temporary_file.write(await file.read())
                temporary_path = Path(temporary_file.name)

            pages = extract_pdf_pages(temporary_path)
            chunks = create_page_chunks(pages, {**metadata, "document_id": Path(file.filename).stem})
            embeddings = generate_embeddings(chunks)
            store_embeddings(chunks, embeddings)

            results.append(UploadResponse(
                filename=file.filename,
                chunks_indexed=len(chunks),
                metadata={key: value for key, value in metadata.items() if key != "document_id"},
            ))
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("PDF indexing failed for %s: %s", file.filename, exc)
            raise HTTPException(
                status_code=422,
                detail=f"Failed to index '{file.filename}': {str(exc)}"
            ) from exc
        finally:
            if temporary_path and temporary_path.exists():
                temporary_path.unlink(missing_ok=True)

    # Clear cache after sync processing
    cache = get_cache_manager()
    cache.clear("finsight:query:")
    cache.clear("finsight:query_stream:")

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content=[r.model_dump() for r in results]
    )


async def _process_single_file(file: UploadFile, metadata: dict[str, Any]) -> None:
    """Process a single file, intended to be called inside ingestion_lock."""
    try:
        content = await file.read()
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temporary_file:
            temporary_file.write(content)
            temporary_path = Path(temporary_file.name)

        try:
            pages = extract_pdf_pages(temporary_path)
            chunks = create_page_chunks(pages, {**metadata, "document_id": Path(file.filename).stem})
            embeddings = generate_embeddings(chunks)
            store_embeddings(chunks, embeddings)
            cache = get_cache_manager()
            cache.clear("finsight:query:")
            cache.clear("finsight:query_stream:")
        finally:
            temporary_path.unlink(missing_ok=True)
    except Exception as exc:
        logger.error("Background ingestion failed for file %s: %s", file.filename, exc)


@app.post("/query")
@limiter.limit(RATE_LIMIT_QUERY)
async def query_documents(
    request: Request,
    query_request: QueryRequest,
    stream: str = "true",
    _key: APIKeyRecord | None = Depends(require_api_key),
):
    """Smart Query: Retrieve and optionally stream answer.

    Intelligently handles both direct query responses and streaming answers.

    Query Parameters:
    - stream: Response mode (optional, default: "true")
        * "true": Stream answer token-by-token (SSE format) - Real-time 🎯
        * "false": Direct JSON response - Traditional
        * "auto": Uses streaming by default (same as "true")

    Returns (stream=false):
        {
          "answer": "Apple's revenue was $391.04 billion...",
          "retrieved_chunks": [...],
          "citations": [...],
          "metadata": [...]
        }

    Returns (stream=true):
        Server-Sent Events stream of tokens:
        data: {"token": "Apple's"}
        data: {"token": " revenue"}
        ...
        data: {"done": true, "citations": [...]}

    HTTP Status: 200 OK
    """
    _ = (request, _key)

    # Normalize stream parameter
    if stream.lower() in ["true", "auto", "1", "yes"]:
        use_stream = True
    elif stream.lower() in ["false", "0", "no"]:
        use_stream = False
    else:
        raise HTTPException(status_code=400, detail="stream must be 'true', 'false', or 'auto'")

    # Check if grounded generation is available
    try:
        grounded_available = is_grounded_generation_available()
    except:
        grounded_available = False

    # Retrieve documents
    try:
        results, _ = retrieve_documents(
            query_request.question,
            top_k=query_request.top_k,
            filters=_query_filters(query_request),
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not results:
        raise HTTPException(status_code=404, detail="No relevant information was found")

    # DIRECT RESPONSE MODE (stream=false)
    if not use_stream:
        query_response = _build_query_response(query_request, results, grounded_enabled=grounded_available)
        return JSONResponse(
            status_code=200,
            content=query_response.model_dump(mode="json")
        )

    # STREAMING MODE (stream=true)
    citations = _build_citations(results)
    context = _build_context(results)

    def event_generator():
        answer_tokens = []
        try:
            for token in generate_answer_grounded_stream(query_request.question, context):
                answer_tokens.append(token)
                yield f"data: {_json.dumps({'token': token})}\n\n"
        except Exception as exc:
            logger.error("Streaming generation failed: %s", exc)
            yield f"data: {_json.dumps({'error': str(exc)})}\n\n"
            return

        answer = "".join(answer_tokens)
        full_response = QueryResponse(
            answer=answer,
            retrieved_chunks=[result.get("text", "") for result in results],
            metadata=results,
            citations=citations,
        )

        yield f"data: {_json.dumps({'done': True, 'citations': [c.model_dump() for c in citations]})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _query_filters(query_request: QueryRequest) -> dict[str, str | None]:
    """Return metadata filters for retrieval from a query request."""
    return {
        "company": query_request.company,
        "fiscal_year": query_request.fiscal_year,
        "filing_type": query_request.filing_type,
        "document_type": query_request.document_type,
        "quarter": query_request.quarter,
    }


def _build_query_response(
    query_request: QueryRequest,
    results: list[dict[str, Any]],
    *,
    grounded_enabled: bool,
) -> QueryResponse:
    """Build the structured query response from retrieval results."""
    answer = "No relevant information was found."
    if results:
        answer = "Relevant financial passages were retrieved."
        if grounded_enabled:
            context = _build_context(results)
            try:
                answer = generate_answer_grounded(query_request.question, context)
            except (ValueError, KeyError, IndexError, Exception) as exc:
                logger.error("Grounded answer generation failed: %s", exc)
                raise HTTPException(status_code=502, detail="Answer generation failed") from exc

    return QueryResponse(
        answer=answer,
        retrieved_chunks=[result.get("text", "") for result in results],
        metadata=results,
        citations=_build_citations(results),
    )


def _build_citations(results: list[dict[str, Any]]) -> list[Citation]:
    """Build citation records from retrieval results."""
    return [
        Citation(
            chunk_id=str(result.get("chunk_id", result.get("chunk_index", "unknown"))),
            source=str(
                result.get(
                    "source",
                    result.get("primary_document", result.get("document_id", "unknown")),
                )
            ),
            page_number=result.get("page_number", "unknown"),
            score=round(float(result.get("score", 0.0)), 4),
            rerank_score=round(float(result["rerank_score"]), 4) if "rerank_score" in result else None,
            cross_encoder_score=round(float(result["cross_encoder_score"]), 4) if "cross_encoder_score" in result else None,
            metric_score=round(float(result["metric_score"]), 4) if "metric_score" in result else None,
            confidence=round(float(result["confidence"]), 4) if "confidence" in result else None,
        )
        for result in results
    ]


def _build_context(results: list[dict[str, Any]]) -> str:
    """Format retrieval results into a context string for generation."""
    return "\n\n".join(
        f"Source: {result.get('source', result.get('document_id', 'unknown'))}; "
        f"Page: {result.get('page_number', 'unknown')}\n{result.get('text', '')}"
        for result in results
    )




# ---------------------------------------------------------------------------
# Admin Endpoints
# ---------------------------------------------------------------------------

@app.post("/admin/keys", response_model=APIKeyResponse)
async def create_key(
    request: Request,
    key_request: APIKeyCreate,
    _admin: APIKeyRecord = Depends(require_admin)
) -> APIKeyResponse:
    """Create a new API key. Requires admin privileges."""
    logger.info("Admin %s created a new API key", _admin.key_id)
    return create_api_key(key_request)


@app.get("/admin/keys", response_model=list[APIKeyInfo])
async def list_keys(
    request: Request,
    _admin: APIKeyRecord = Depends(require_admin)
) -> list[APIKeyInfo]:
    """List all API keys. Requires admin privileges."""
    logger.info("Admin %s listed API keys", _admin.key_id)
    return list_api_keys()


@app.delete("/admin/keys/{key_id}")
async def delete_key(
    request: Request,
    key_id: str,
    _admin: APIKeyRecord = Depends(require_admin)
) -> dict[str, str]:
    """Revoke an API key. Requires admin privileges."""
    if revoke_api_key(key_id):
        logger.info("Admin %s revoked API key %s", _admin.key_id, key_id)
        return {"detail": f"Key {key_id} revoked successfully"}
    raise HTTPException(status_code=404, detail="API key not found")
