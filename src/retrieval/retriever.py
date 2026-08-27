"""Metadata-aware semantic retrieval over the local FAISS index."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from src.embeddings.embedder import get_embedding_model, load_vector_store
from src.retrieval.query_parser import parse_query
from src.retrieval.reranker import rerank_documents


TOP_K = 5


def _matches_filter(record: dict[str, Any], key: str, value: Any) -> bool:
    """Match metadata filters without rejecting common company name variants."""
    record_value = record.get(key)
    if isinstance(value, list):
        return any(_matches_filter(record, key, item) for item in value)
    if key != "company" or not isinstance(record_value, str) or not isinstance(value, str):
        return record_value == value

    normalize = lambda text: text.lower().replace(",", "").replace(".", "").strip()
    normalized_record = normalize(record_value)
    normalized_value = normalize(value)
    legal_suffixes = (" inc", " corporation", " corp", " com", " company")
    for suffix in legal_suffixes:
        normalized_record = normalized_record.removesuffix(suffix).strip()
        normalized_value = normalized_value.removesuffix(suffix).strip()
    return normalized_record == normalized_value


def retrieve_documents(
    query: str,
    top_k: int = TOP_K,
    filters: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Retrieve relevant chunks and return them with parsed query data."""
    if not query.strip():
        raise ValueError("Query cannot be empty")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    parsed_query = parse_query(query)
    index, metadata = load_vector_store()
    query_vector = get_embedding_model().encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    )
    candidate_count = min(max(top_k * 10, top_k), index.ntotal)
    scores, indices = index.search(np.asarray(query_vector, dtype="float32"), candidate_count)

    filters = {
        "ticker": parsed_query.get("tickers") or parsed_query.get("ticker"),
        "fiscal_year": parsed_query.get("fiscal_year"),
        "filing_type": parsed_query.get("filing_type"),
        "document_type": None,
        "quarter": None,
        **(filters or {}),
    }
    results: list[dict[str, Any]] = []
    for score, index_position in zip(scores[0], indices[0]):
        if index_position < 0:
            continue
        record = metadata[index_position]
        if all(
            not value
            or _matches_filter(record, key, value)
            for key, value in filters.items()
        ):
            results.append({"score": float(score), **record})

    if os.getenv("ENABLE_RERANKING", "true").lower() in {"1", "true", "yes"}:
        results = rerank_documents(
            query,
            results,
            top_k=top_k,
            metric=parsed_query.get("metric"),
        )
    else:
        results = results[:top_k]

    return results, parsed_query


def display_results(query: str, results: list[dict[str, Any]]) -> None:
    """Print retrieved chunks for command-line inspection."""
    print(f"\nQuestion: {query}")
    for rank, result in enumerate(results, start=1):
        print(
            f"\n{rank}. score={result['score']:.4f} "
            f"page={result.get('page_number', 'unknown')}"
        )
        print(result.get("text", "")[:1000])


if __name__ == "__main__":
    question = "What was NVIDIA's net income in fiscal year 2026?"
    results, _ = retrieve_documents(question)
    display_results(question, results)
