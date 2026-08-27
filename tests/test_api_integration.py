from fastapi.testclient import TestClient

from src import api


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


def test_query_returns_retrieved_evidence_without_openai(monkeypatch) -> None:
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
    monkeypatch.setattr(api, "is_openai_configured", lambda: False)

    response = client.post("/query", json={"question": "What was net income?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Relevant financial passages were retrieved."
    assert body["citations"] == [{
        "chunk_id": "apple-p4-c0",
        "source": "apple-2025",
        "page_number": 4,
        "score": 0.91,
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