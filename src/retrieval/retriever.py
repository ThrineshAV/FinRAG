"""Metadata-aware semantic retrieval over the local FAISS index."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.embeddings.embedder import get_embedding_model, load_vector_store
from src.retrieval.query_parser import parse_query


TOP_K = 5


def retrieve_documents(
    query: str,
    top_k: int = TOP_K,
    filters: dict[str, str | None] | None = None,
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
        "ticker": parsed_query.get("ticker"),
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
        if all(not value or record.get(key) == value for key, value in filters.items()):
            results.append({"score": float(score), **record})
        if len(results) == top_k:
            break

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
