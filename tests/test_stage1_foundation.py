from __future__ import annotations

from pathlib import Path

import fitz
import numpy as np

from src.embeddings import embedder
from src.ingestion.sec_ingestion import extract_pdf_pages
from src.processing.chunker import create_page_chunks
from src.retrieval.reranker import rerank_documents
from src.retrieval.query_parser import parse_query
from src.retrieval.retriever import _matches_filter
from src.generation import llm
from src.evaluation.metrics import evaluate_retrieval
from src.api import app
from fastapi.testclient import TestClient


def create_test_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Apple net income was reported on this page.")
    document.save(path)
    document.close()


def test_pdf_extraction_preserves_page_number(tmp_path: Path) -> None:
    pdf_path = tmp_path / "annual_report.pdf"
    create_test_pdf(pdf_path)

    pages = extract_pdf_pages(pdf_path)

    assert pages == [
        {"page_number": 1, "text": "Apple net income was reported on this page.\n"}
    ]


def test_page_chunks_preserve_required_metadata() -> None:
    chunks = create_page_chunks(
        [{"page_number": 4, "text": "Apple net income was reported."}],
        {
            "document_id": "apple-2024",
            "company": "Apple",
            "document_type": "annual_report",
            "fiscal_year": "2024",
            "quarter": "FY",
        },
    )

    assert chunks[0]["chunk_id"] == "apple-2024-p4-c0"
    assert chunks[0]["metadata"]["page_number"] == 4
    assert chunks[0]["metadata"]["company"] == "Apple"


def test_faiss_store_round_trip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(embedder, "VECTOR_DB_DIR", tmp_path)
    monkeypatch.setattr(embedder, "INDEX_PATH", tmp_path / "index.faiss")
    monkeypatch.setattr(embedder, "METADATA_PATH", tmp_path / "metadata.json")

    chunks = [{
        "chunk_id": "apple-p1-c0",
        "chunk_index": 0,
        "text": "Apple net income",
        "metadata": {"company": "Apple", "page_number": 1},
    }]
    embedder.store_embeddings(chunks, np.array([[1.0, 0.0]], dtype="float32"))

    index, metadata = embedder.load_vector_store()

    assert index.ntotal == 1
    assert metadata[0]["chunk_id"] == "apple-p1-c0"
    assert metadata[0]["company"] == "Apple"

    additional_chunk = [{
        "chunk_id": "microsoft-p1-c0",
        "chunk_index": 0,
        "text": "Microsoft revenue",
        "metadata": {"company": "Microsoft", "page_number": 1},
    }]
    embedder.store_embeddings(additional_chunk, np.array([[0.0, 1.0]], dtype="float32"))
    appended_index, appended_metadata = embedder.load_vector_store()

    assert appended_index.ntotal == 2
    assert [item["chunk_id"] for item in appended_metadata] == [
        "apple-p1-c0",
        "microsoft-p1-c0",
    ]


def test_reranker_supports_faiss_records() -> None:
    class FakeModel:
        def predict(self, pairs):
            return [0.2, 0.8]

    documents = [
        {"chunk_id": "first", "text": "Revenue was reported."},
        {"chunk_id": "second", "text": "Net income was reported."},
    ]

    results = rerank_documents(
        "What was net income?",
        documents,
        top_k=1,
        metric="net income",
        model=FakeModel(),
    )

    assert results[0]["chunk_id"] == "second"
    assert results[0]["metric_score"] == 1.0


def test_query_parser_detects_multiple_companies() -> None:
    parsed = parse_query("Compare Apple's and Microsoft's revenue in 2025")

    assert parsed["tickers"] == ["AAPL", "MSFT"]
    assert parsed["ticker"] == "AAPL"


def test_company_filter_accepts_common_legal_name_variant() -> None:
    assert _matches_filter({"company": "Apple Inc."}, "company", "Apple")


def test_gemini_generation_uses_grounded_context(monkeypatch) -> None:
    from unittest.mock import MagicMock

    captured: dict = {}

    class FakeResponse:
        text = "Apple reported $10."

    def fake_generate(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeResponse()

    model_instance = MagicMock()
    model_instance.generate_content.side_effect = fake_generate

    def fake_get_client():
        return model_instance

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(llm, "_get_gemini_client", fake_get_client)

    answer = llm.generate_answer_grounded(
        "What was Apple's net income?",
        "Source: apple-2024; Page: 4\nNet income was $10.",
    )

    assert answer == "Apple reported $10."

    # The context (sources) must be passed through to the Gemini SDK
    # so the model only sees grounded sources.
    parts_blob = str(captured["args"])
    assert "Source: apple-2024" in parts_blob
    assert "Net income was $10." in parts_blob
    assert "What was Apple's net income?" in parts_blob


def test_retrieval_metrics_calculate_hit_rate_and_mrr() -> None:
    responses = {
        "first question": [{"chunk_id": "wrong"}, {"chunk_id": "target-1"}],
        "second question": [{"chunk_id": "target-2"}],
    }

    metrics = evaluate_retrieval(
        [
            {"question": "first question", "relevant_chunk_ids": ["target-1"]},
            {"question": "second question", "relevant_chunk_ids": ["target-2"]},
        ],
        lambda question, top_k: responses[question][:top_k],
        top_k=3,
    )

    assert metrics == {
        "cases": 2,
        "hit_rate": 1.0,
        "mrr": 0.75,
        "average_retrieved": 1.5,
    }


def test_api_adds_request_observability_headers() -> None:
    response = TestClient(app).get("/health", headers={"X-Request-ID": "test-request"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-request"
    assert float(response.headers["X-Process-Time-Ms"]) >= 0


def test_api_readiness_reports_unavailable_store(monkeypatch) -> None:
    from src import api

    def unavailable_store():
        raise FileNotFoundError("missing index")

    monkeypatch.setattr(api, "load_vector_store", unavailable_store)
    response = TestClient(api.app).get("/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "Vector store is unavailable"}
