"""FastAPI interface for the FinSight-RAG pipeline."""

from __future__ import annotations

import logging
import time
from uuid import uuid4
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from src.embeddings.embedder import generate_embeddings, store_embeddings
from src.embeddings.embedder import load_vector_store
from src.ingestion.sec_ingestion import extract_pdf_pages
from src.processing.chunker import create_page_chunks
from src.generation.llm import generate_openai_answer, is_openai_configured
from src.retrieval.retriever import retrieve_documents

logger = logging.getLogger(__name__)

app = FastAPI(title="FinSight-RAG", version="1.0.0")


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


class QueryRequest(BaseModel):
    """Validated financial question and optional metadata filters."""

    question: str = Field(min_length=3)
    company: str | None = None
    fiscal_year: str | None = None
    filing_type: str | None = None
    top_k: int = Field(default=5, ge=1, le=50)


class Citation(BaseModel):
    """Citation for one retrieved source chunk."""

    chunk_id: str
    source: str
    page_number: int | str
    score: float


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
async def upload_pdf(
    file: UploadFile = File(...),
    company: str = Form(...),
    document_type: str = Form(...),
    fiscal_year: str = Form(...),
    quarter: str = Form(...),
) -> UploadResponse:
    """Extract, chunk, embed, and index an uploaded financial PDF."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

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
async def query_documents(request: QueryRequest) -> QueryResponse:
    """Retrieve relevant chunks with metadata and page citations."""
    try:
        results, _ = retrieve_documents(
            request.question,
            top_k=request.top_k,
            filters={
                "company": request.company,
                "fiscal_year": request.fiscal_year,
                "filing_type": request.filing_type,
                "document_type": None,
                "quarter": None,
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
                answer = generate_openai_answer(request.question, context)
            except (ValueError, requests.RequestException, KeyError, IndexError) as exc:
                logger.error("Grounded answer generation failed: %s", exc)
                raise HTTPException(status_code=502, detail="Answer generation failed") from exc

    citations = [
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
        )
        for result in results
    ]
    return QueryResponse(
        answer=answer,
        retrieved_chunks=[result.get("text", "") for result in results],
        metadata=results,
        citations=citations,
    )
