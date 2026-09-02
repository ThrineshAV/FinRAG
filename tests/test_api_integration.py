import os
from pathlib import Path
from unittest.mock import patch

import fitz
import numpy as np
from fastapi.testclient import TestClient

from src import api
from src.embeddings import embedder

# Disable authentication for API integration tests
os.environ["AUTH_REQUIRED"] = "false"

client = TestClient(api.app)


def test_health_endpoint_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_endpoint_returns_ready_when_store_loads(monkeypatch) -> None:
    monkeypatch.setattr(api, "load_vector_store", lambda: (object(), []))

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_query_returns_retrieved_evidence_without_grounding(monkeypatch) -> None:
    monkeypatch.setattr(
        api,
        "retrieve_documents",
        lambda question, top_k, filters: (
            [{
                "chunk_id": "apple-p4-c0",
                "text": "Net income was $10 million.",
                "source": "apple-2025",
                "page_number": 4,
                "score": 0.91,
            }],
            {},
        ),
    )
    monkeypatch.setattr(api, "is_grounded_generation_available", lambda: False)

    response = client.post("/query", json={"question": "What was net income?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Relevant financial passages were retrieved."
    assert body["citations"] == [{
        "chunk_id": "apple-p4-c0",
        "source": "apple-2025",
        "page_number": 4,
        "score": 0.91,
        "rerank_score": None,
        "cross_encoder_score": None,
        "metric_score": None,
        "confidence": None,
    }]


def test_query_returns_service_unavailable_when_index_is_missing(monkeypatch) -> None:
    def missing_index(question, top_k, filters):
        raise FileNotFoundError("missing index")

    monkeypatch.setattr(api, "retrieve_documents", missing_index)

    response = client.post("/query", json={"question": "What was revenue?"})

    assert response.status_code == 503
    assert response.json() == {"detail": "missing index"}


def test_upload_rejects_non_pdf() -> None:
    response = client.post(
        "/upload",
        files={"file": ("notes.txt", b"not a PDF", "text/plain")},
        data={
            "company": "Apple",
            "document_type": "annual_report",
            "fiscal_year": "2025",
            "quarter": "FY",
        },
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Only PDF files are supported"}


def _create_test_pdf(path: Path, text: str) -> None:
    """Create a single-page PDF with the given text."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


class FakeEmbeddingModel:
    """Return fixed-dimension normalized vectors without downloading a model."""

    def encode(self, texts, **kwargs):
        rng = np.random.RandomState(42)
        vectors = rng.randn(len(texts), 384).astype("float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms


def test_upload_then_query_returns_uploaded_content(tmp_path, monkeypatch) -> None:
    """End-to-end: upload a PDF, then query and find its content."""
    # Isolate FAISS storage to tmp_path
    monkeypatch.setattr(embedder, "VECTOR_DB_DIR", tmp_path)
    monkeypatch.setattr(embedder, "INDEX_PATH", tmp_path / "index.faiss")
    monkeypatch.setattr(embedder, "METADATA_PATH", tmp_path / "metadata.json")

    # Use a fake embedding model so tests run without downloading
    fake_model = FakeEmbeddingModel()
    monkeypatch.setattr(embedder, "_embedding_model", fake_model)

    # Disable grounded generation so the query returns evidence-only
    monkeypatch.setattr(api, "is_grounded_generation_available", lambda: False)

    # Disable reranking to avoid downloading the cross-encoder
    monkeypatch.setenv("ENABLE_RERANKING", "false")

    # Create a test PDF
    pdf_path = tmp_path / "tesla_report.pdf"
    _create_test_pdf(pdf_path, "Tesla reported revenue of $96 billion in fiscal year 2025.")

    # Upload
    with open(pdf_path, "rb") as f:
        upload_response = client.post(
            "/upload",
            files={"file": ("tesla_report.pdf", f, "application/pdf")},
            data={
                "company": "Tesla",
                "document_type": "annual_report",
                "fiscal_year": "2025",
                "quarter": "FY",
            },
        )

    assert upload_response.status_code == 201
    assert upload_response.json()["chunks_indexed"] >= 1

    # Query
    query_response = client.post(
        "/query",
        json={
            "question": "What was Tesla's revenue in 2025?",
            "company": "Tesla",
            "fiscal_year": "2025",
        },
    )

    assert query_response.status_code == 200
    body = query_response.json()
    assert len(body["retrieved_chunks"]) >= 1
    assert any("Tesla" in chunk or "revenue" in chunk for chunk in body["retrieved_chunks"])