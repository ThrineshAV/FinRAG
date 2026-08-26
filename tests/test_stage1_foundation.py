from __future__ import annotations

from pathlib import Path

import fitz
import numpy as np

from src.embeddings import embedder
from src.ingestion.sec_ingestion import extract_pdf_pages
from src.processing.chunker import create_page_chunks


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
