"""Deterministic retrieval evaluation metrics."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypedDict


class EvaluationCase(TypedDict):
    """One question and the chunk IDs considered relevant."""

    question: str
    relevant_chunk_ids: list[str]


def evaluate_retrieval(
    cases: Iterable[EvaluationCase],
    retrieve: Callable[[str, int], list[dict[str, Any]]],
    top_k: int = 5,
) -> dict[str, float | int]:
    """Evaluate retrieval using hit rate and mean reciprocal rank."""
    if top_k < 1:
        raise ValueError("top_k must be positive")

    case_list = list(cases)
    if not case_list:
        raise ValueError("At least one evaluation case is required")

    hits = 0
    reciprocal_rank_total = 0.0
    retrieved_total = 0

    for case in case_list:
        relevant_ids = set(case["relevant_chunk_ids"])
        results = retrieve(case["question"], top_k)
        retrieved_total += len(results)
        for rank, result in enumerate(results, start=1):
            if result.get("chunk_id") in relevant_ids:
                hits += 1
                reciprocal_rank_total += 1.0 / rank
                break

    case_count = len(case_list)
    return {
        "cases": case_count,
        "hit_rate": hits / case_count,
        "mrr": reciprocal_rank_total / case_count,
        "average_retrieved": retrieved_total / case_count,
    }


def retrieve_for_evaluation(query: str, top_k: int) -> list[dict[str, Any]]:
    """Adapt the production retriever to the benchmark callback contract."""
    from src.retrieval.retriever import retrieve_documents

    results, _ = retrieve_documents(query, top_k=top_k)
    return results