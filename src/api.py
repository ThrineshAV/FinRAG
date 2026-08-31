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

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAIError
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
    generate_openai_answer,
    generate_openai_answer_stream,
    is_openai_configured,
)
from src.retrieval.retriever import retrieve_documents

logger = logging.getLogger(__name__)

# Rate limiting configuration
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="FinSight-RAG", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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


@app.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit(RATE_LIMIT_UPLOAD)
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
    company: str = Form(...),
    document_type: str = Form(...),
    fiscal_year: str = Form(...),
    quarter: str = Form(...),
    _key: APIKeyRecord | None = Depends(require_upload),
) -> UploadResponse:
    """Extract, chunk, embed, and index an uploaded financial PDF."""
    _ = _key
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Check file size via Content-Length header
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_SIZE:
        max_mb = MAX_UPLOAD_SIZE / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds maximum allowed size of {max_mb:.0f} MB"
        )

    temporary_path: Path | None = None
    metadata = {
        "company": company,
        "document_type": document_type,
        "fiscal_year": fiscal_year,
        "quarter": quarter,
        "document_id": Path(file.filename).stem,
    }

    try:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temporary_file:
            temporary_file.write(await file.read())
            temporary_path = Path(temporary_file.name)

        pages = extract_pdf_pages(temporary_path)
        chunks = create_page_chunks(pages, metadata)
        embeddings = generate_embeddings(chunks)
        store_embeddings(chunks, embeddings)
        cache = get_cache_manager()
        cache.clear("finsight:query:")
        cache.clear("finsight:query_stream:")
        return UploadResponse(
            filename=file.filename,
            chunks_indexed=len(chunks),
            metadata={key: value for key, value in metadata.items() if key != "document_id"},
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("PDF indexing failed: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

@app.post("/upload/batch", status_code=status.HTTP_202_ACCEPTED)
async def upload_batch(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    company: str = Form(...),
    document_type: str = Form(...),
    fiscal_year: str = Form(...),
    quarter: str = Form(...),
    _key: APIKeyRecord | None = Depends(require_upload),
) -> dict[str, str]:
    """Queue multiple PDFs for extraction, chunking, and indexing in the background."""
    job_id = str(uuid4())
    metadata = {
        "company": company,
        "document_type": document_type,
        "fiscal_year": fiscal_year,
        "quarter": quarter,
    }

    async def _process_batch_coro():
        for file in files:
            async with ingestion_lock:
                await _process_single_file(file, metadata)

    background_tasks.add_task(_process_batch_coro)
    return {"status": "accepted", "job_id": job_id}


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


@app.post("/query", response_model=QueryResponse)
@limiter.limit(RATE_LIMIT_QUERY)
async def query_documents(
    request: Request,
    response: Response,
    query_request: QueryRequest,
    _key: APIKeyRecord | None = Depends(require_api_key),
) -> QueryResponse:
    """Retrieve relevant chunks with metadata and page citations, using cache when available."""
    _ = (request, _key)
    openai_enabled = is_openai_configured()
    cache_key = _query_cache_key("query", query_request, openai_enabled=openai_enabled)
    cached_response = _cache_lookup(cache_key, "query")
    if cached_response is not None:
        response.headers["X-Cache"] = "HIT"
        return cached_response

    response.headers["X-Cache"] = "MISS"
    try:
        results, _ = retrieve_documents(
            query_request.question,
            top_k=query_request.top_k,
            filters=_query_filters(query_request),
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    query_response = _build_query_response(query_request, results, openai_enabled=openai_enabled)
    get_cache_manager().set(cache_key, query_response.model_dump(mode="json"), ttl=CACHE_TTL_SECONDS)
    return query_response


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
    openai_enabled: bool,
) -> QueryResponse:
    """Build the structured query response from retrieval results."""
    answer = "No relevant information was found."
    if results:
        answer = "Relevant financial passages were retrieved."
        if openai_enabled:
            context = _build_context(results)
            try:
                answer = generate_openai_answer(query_request.question, context)
            except (ValueError, OpenAIError, KeyError, IndexError) as exc:
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


@app.post("/query/stream")
@limiter.limit(RATE_LIMIT_QUERY)
async def query_documents_stream(
    request: Request,
    response: Response,
    query_request: QueryRequest,
    _key: APIKeyRecord | None = Depends(require_api_key),
) -> StreamingResponse:
    """Stream an answer as Server-Sent Events (SSE).

    Retrieval + reranking happen synchronously before the stream starts.
    Only the LLM generation phase streams token-by-token.
    """
    _ = (request, _key)
    if not is_openai_configured():
        raise HTTPException(
            status_code=400,
            detail="Streaming requires OPENAI_API_KEY to be configured",
        )

    cache_key = _query_cache_key("query_stream", query_request, openai_enabled=True)
    cached_response = _cache_lookup(cache_key, "query_stream")
    if cached_response is not None:
        # For stream cache, we store the full finished response
        response.headers["X-Cache"] = "HIT"
        return StreamingResponse(
            iter([f"data: {_json.dumps({'cached': True, 'answer': cached_response.answer, 'citations': [c.model_dump() for c in cached_response.citations]})}\n\n"]),
            media_type="text/event-stream"
        )

    response.headers["X-Cache"] = "MISS"
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

    context = _build_context(results)
    citations = _build_citations(results)

    # To cache the final result, we collect the stream manually
    def event_generator():
        answer_tokens = []
        try:
            for token in generate_openai_answer_stream(query_request.question, context):
                answer_tokens.append(token)
                yield f"data: {_json.dumps({'token': token})}\n\n"
        except (ValueError, OpenAIError) as exc:
            logger.error("Streaming generation failed: %s", exc)
            yield f"data: {_json.dumps({'error': str(exc)})}\n\n"
            return

        answer = "".join(answer_tokens)

        # Save to cache
        full_response = QueryResponse(
            answer=answer,
            retrieved_chunks=[result.get("text", "") for result in results],
            metadata=results,
            citations=citations,
        )
        get_cache_manager().set(cache_key, full_response.model_dump(mode="json"), ttl=CACHE_TTL_SECONDS)

        yield f"data: {_json.dumps({'done': True, 'citations': [c.model_dump() for c in citations]})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Admin Endpoints
# ---------------------------------------------------------------------------

@app.post("/admin/keys", response_model=APIKeyResponse)
async def create_key(
    key_request: APIKeyCreate,
    _admin: APIKeyRecord = Depends(require_admin)
) -> APIKeyResponse:
    """Create a new API key. Requires admin privileges."""
    logger.info("Admin %s created a new API key", _admin.key_id)
    return create_api_key(key_request)


@app.get("/admin/keys", response_model=list[APIKeyInfo])
async def list_keys(
    _admin: APIKeyRecord = Depends(require_admin)
) -> list[APIKeyInfo]:
    """List all API keys. Requires admin privileges."""
    logger.info("Admin %s listed API keys", _admin.key_id)
    return list_api_keys()


@app.delete("/admin/keys/{key_id}")
async def delete_key(
    key_id: str,
    _admin: APIKeyRecord = Depends(require_admin)
) -> dict[str, str]:
    """Revoke an API key. Requires admin privileges."""
    if revoke_api_key(key_id):
        logger.info("Admin %s revoked API key %s", _admin.key_id, key_id)
        return {"detail": f"Key {key_id} revoked successfully"}
    raise HTTPException(status_code=404, detail="API key not found")
