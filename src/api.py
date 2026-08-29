"""FastAPI interface for the FinSight-RAG pipeline."""

from __future__ import annotations

import json as _json
import logging
import os
import time
from uuid import uuid4
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import requests
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAIError
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

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

# Configuration
RATE_LIMIT_QUERY = os.getenv("RATE_LIMIT_QUERY", "20/minute")
RATE_LIMIT_UPLOAD = os.getenv("RATE_LIMIT_UPLOAD", "5/minute")
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25")) * 1024 * 1024  # Convert MB to bytes


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
) -> UploadResponse:
    """Extract, chunk, embed, and index an uploaded financial PDF."""
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
        return UploadResponse(
            filename=file.filename,
            chunks_indexed=len(chunks),
            metadata={key: value for key, value in metadata.items() if key != "document_id"},
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("PDF indexing failed: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@app.post("/query", response_model=QueryResponse)
@limiter.limit(RATE_LIMIT_QUERY)
async def query_documents(request: Request, query_request: QueryRequest) -> QueryResponse:
    """Retrieve relevant chunks with metadata and page citations."""
    try:
        results, _ = retrieve_documents(
            query_request.question,
            top_k=query_request.top_k,
            filters={
                "company": query_request.company,
                "fiscal_year": query_request.fiscal_year,
                "filing_type": query_request.filing_type,
                "document_type": query_request.document_type,
                "quarter": query_request.quarter,
            },
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    answer = "No relevant information was found."
    if results:
        answer = "Relevant financial passages were retrieved."
        if is_openai_configured():
            context = "\n\n".join(
                f"Source: {result.get('source', result.get('document_id', 'unknown'))}; "
                f"Page: {result.get('page_number', 'unknown')}\n{result.get('text', '')}"
                for result in results
            )
            try:
                answer = generate_openai_answer(query_request.question, context)
            except (ValueError, OpenAIError, KeyError, IndexError) as exc:
                logger.error("Grounded answer generation failed: %s", exc)
                raise HTTPException(status_code=502, detail="Answer generation failed") from exc

    citations = _build_citations(results)
    return QueryResponse(
        answer=answer,
        retrieved_chunks=[result.get("text", "") for result in results],
        metadata=results,
        citations=citations,
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
async def query_documents_stream(request: Request, query_request: QueryRequest) -> StreamingResponse:
    """Stream an answer as Server-Sent Events (SSE).

    Retrieval + reranking happen synchronously before the stream starts.
    Only the LLM generation phase streams token-by-token.

    Event format:
    - ``data: {"token": "..."}`` for each answer token
    - ``data: {"done": true, "citations": [...]}`` as the final event
    """
    if not is_openai_configured():
        raise HTTPException(
            status_code=400,
            detail="Streaming requires OPENAI_API_KEY to be configured",
        )

    try:
        results, _ = retrieve_documents(
            query_request.question,
            top_k=query_request.top_k,
            filters={
                "company": query_request.company,
                "fiscal_year": query_request.fiscal_year,
                "filing_type": query_request.filing_type,
                "document_type": query_request.document_type,
                "quarter": query_request.quarter,
            },
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not results:
        raise HTTPException(status_code=404, detail="No relevant information was found")

    context = _build_context(results)
    citations = _build_citations(results)

    def event_generator():
        try:
            for token in generate_openai_answer_stream(query_request.question, context):
                yield f"data: {_json.dumps({'token': token})}\n\n"
        except (ValueError, OpenAIError) as exc:
            logger.error("Streaming generation failed: %s", exc)
            yield f"data: {_json.dumps({'error': str(exc)})}\n\n"
            return
        yield f"data: {_json.dumps({'done': True, 'citations': [c.model_dump() for c in citations]})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
