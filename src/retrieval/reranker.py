from __future__ import annotations

import re
from typing import Any

from sentence_transformers import CrossEncoder


# ============================================================
# Configuration
# ============================================================

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Weight used for the financial metric relevance
METRIC_BOOST = 1.5


# ============================================================
# Load reranker
# ============================================================

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """Load the cross-encoder only when reranking is requested."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


# ============================================================
# Metric relevance
# ============================================================

def calculate_metric_relevance(
    text,
    metric
):
    """
    Calculate how strongly a document matches
    the financial metric requested by the user.
    """

    if not metric:
        return 0.0

    if not text:
        return 0.0

    text_lower = text.lower()
    metric_lower = metric.lower()

    # Direct metric match
    if metric_lower in text_lower:
        return 1.0

    # Handle equivalent financial terminology
    metric_aliases = {

        "revenue": [
            "revenue",
            "net sales",
            "total net sales"
        ],

        "net income": [
            "net income",
            "net earnings"
        ],

        "gross profit": [
            "gross profit"
        ],

        "gross margin": [
            "gross margin"
        ],

        "operating income": [
            "operating income"
        ],

        "operating expenses": [
            "operating expenses",
            "operating expense"
        ],

        "income before tax": [
            "income before income tax",
            "income before tax",
            "pretax income"
        ],

        "earnings per share": [
            "earnings per share",
            "eps"
        ],

        "cash flow": [
            "cash flow",
            "cash flows"
        ]
    }

    aliases = metric_aliases.get(
        metric_lower,
        []
    )

    for alias in aliases:

        if re.search(
            rf"\b{re.escape(alias)}\b",
            text_lower
        ):

            return 1.0

    return 0.0


# ============================================================
# Rerank documents
# ============================================================

def rerank_documents(
    query: str,
    documents: list[dict[str, Any]],
    top_k: int = 5,
    metric: str | None = None,
    model: Any | None = None,
) -> list[dict[str, Any]]:
    """
    Rerank retrieved documents using a CrossEncoder
    combined with financial metric relevance.

    Documents are flattened FAISS metadata records with a ``text`` field.
    """

    if not documents:
        return []
    if top_k < 1:
        raise ValueError("top_k must be positive")

    pairs = [(query, document.get("text", "")) for document in documents]

    # --------------------------------------------------------
    # Generate CrossEncoder scores
    # --------------------------------------------------------

    cross_encoder_scores = (model or get_reranker()).predict(pairs)

    # --------------------------------------------------------
    # Calculate final ranking scores
    # --------------------------------------------------------

    ranked_documents = []

    for document, cross_score in zip(documents, cross_encoder_scores):
        text = document.get("text", "")

        metric_score = calculate_metric_relevance(
            text,
            metric
        )

        final_score = (
            float(cross_score)
            +
            METRIC_BOOST * metric_score
        )

        ranked_documents.append(
            {
                **document,
                "score": final_score,
                "rerank_score": final_score,
                "cross_encoder_score": float(cross_score),
                "metric_score": metric_score,
            }
        )

    # --------------------------------------------------------
    # Highest final score first
    # --------------------------------------------------------

    ranked_documents.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    # --------------------------------------------------------
    # Return top K
    # --------------------------------------------------------

    return ranked_documents[:top_k]


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    print(
        "\nReranker module loaded successfully."
    )