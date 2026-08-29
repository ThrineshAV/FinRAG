"""Deterministic retrieval and answer quality evaluation metrics."""

from __future__ import annotations

import re
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


# ============================================================
# Answer quality metrics
# ============================================================

def _tokenize(text: str) -> list[str]:
    """Simple word-level tokenizer for overlap calculations."""
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


def evaluate_answer_quality(
    question: str,
    answer: str,
    context: str,
) -> dict[str, float]:
    """Score a generated answer for faithfulness and relevance.

    Returns ``{"faithfulness": float, "relevance": float}``, each 0.0–1.0.

    **Faithfulness** — extractive overlap: what fraction of substantive tokens
    in the answer also appear in the source context.  A fully grounded answer
    should score close to 1.0.

    **Relevance** — does the answer address the financial metric requested in
    the question?  Uses ``query_parser.parse_query()`` to identify the expected
    metric, then checks whether it appears in the answer.
    """
    # -- Faithfulness (extractive overlap) --------------------------------
    answer_tokens = _tokenize(answer)
    context_tokens = set(_tokenize(context))

    if answer_tokens:
        overlap = sum(1 for token in answer_tokens if token in context_tokens)
        faithfulness = overlap / len(answer_tokens)
    else:
        faithfulness = 0.0

    # -- Relevance (metric presence) --------------------------------------
    from src.retrieval.query_parser import parse_query

    parsed = parse_query(question)
    metric = parsed.get("metric", "")
    if metric:
        relevance = 1.0 if metric.lower() in answer.lower() else 0.0
    else:
        # No specific metric requested — cannot penalize, assume relevant.
        relevance = 1.0

    return {
        "faithfulness": round(faithfulness, 4),
        "relevance": round(relevance, 4),
    }