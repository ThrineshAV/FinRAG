"""Tests proving that reranking changes result ordering and adds confidence."""

from __future__ import annotations

from unittest.mock import patch

from src.retrieval.reranker import rerank_documents, RERANKER_MODEL


class FakeModel:
    """Return predetermined cross-encoder scores."""

    def __init__(self, scores: list[float]):
        self._scores = scores

    def predict(self, pairs):
        return self._scores


def test_reranker_reorders_by_semantic_relevance() -> None:
    """The reranker should promote a semantically closer document over
    the one that happened to rank first from FAISS."""
    documents = [
        {"chunk_id": "faiss-top", "text": "General corporate overview."},
        {"chunk_id": "faiss-mid", "text": "Revenue breakdown by segment."},
        {"chunk_id": "faiss-low", "text": "Net income was $10 billion."},
    ]
    # Cross-encoder says the third document is the best match.
    model = FakeModel([0.1, 0.3, 0.9])

    results = rerank_documents(
        "What was net income?",
        documents,
        top_k=3,
        model=model,
    )

    assert results[0]["chunk_id"] == "faiss-low"
    assert results[-1]["chunk_id"] == "faiss-top"


def test_metric_boost_promotes_matching_document() -> None:
    """When cross-encoder scores are equal, the metric boost should
    promote the document that contains the requested financial metric."""
    documents = [
        {"chunk_id": "no-metric", "text": "General business discussion."},
        {"chunk_id": "has-metric", "text": "Revenue was $50 billion."},
    ]
    # Equal cross-encoder scores.
    model = FakeModel([0.5, 0.5])

    results = rerank_documents(
        "What was revenue?",
        documents,
        top_k=2,
        metric="revenue",
        model=model,
    )

    assert results[0]["chunk_id"] == "has-metric"


def test_confidence_is_normalized_0_to_1() -> None:
    """Every result should have a confidence in [0.0, 1.0], with the
    top result at 1.0 and the bottom at 0.0."""
    documents = [
        {"chunk_id": "a", "text": "First."},
        {"chunk_id": "b", "text": "Second."},
        {"chunk_id": "c", "text": "Third."},
    ]
    model = FakeModel([0.2, 0.8, 0.5])

    results = rerank_documents(
        "query",
        documents,
        top_k=3,
        model=model,
    )

    confidences = [r["confidence"] for r in results]
    assert all(0.0 <= c <= 1.0 for c in confidences)
    assert confidences[0] == 1.0  # highest scorer
    assert confidences[-1] == 0.0  # lowest scorer


def test_confidence_single_result() -> None:
    """A single result should have confidence 1.0."""
    documents = [{"chunk_id": "only", "text": "Only document."}]
    model = FakeModel([0.7])

    results = rerank_documents("query", documents, top_k=1, model=model)

    assert len(results) == 1
    assert results[0]["confidence"] == 1.0


def test_reranker_respects_top_k() -> None:
    """Requesting top_k=2 from 5 documents should return exactly 2."""
    documents = [
        {"chunk_id": f"doc-{i}", "text": f"Document {i}."}
        for i in range(5)
    ]
    model = FakeModel([0.1, 0.9, 0.5, 0.3, 0.7])

    results = rerank_documents("query", documents, top_k=2, model=model)

    assert len(results) == 2


def test_bge_reranker_model_name_default() -> None:
    """The default reranker model should be BAAI/bge-reranker-base."""
    assert RERANKER_MODEL == "BAAI/bge-reranker-base"


def test_bge_reranker_model_name_configurable(monkeypatch) -> None:
    """The RERANKER_MODEL env var should override the default model name."""
    monkeypatch.setenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    # Re-import to pick up the env var
    import importlib
    from src.retrieval import reranker
    importlib.reload(reranker)

    assert reranker.RERANKER_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Restore original
    monkeypatch.delenv("RERANKER_MODEL")
    importlib.reload(reranker)
