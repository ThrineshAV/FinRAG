from sentence_transformers import CrossEncoder
import re


# ============================================================
# Configuration
# ============================================================

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Weight used for the financial metric relevance
METRIC_BOOST = 1.5


# ============================================================
# Load reranker
# ============================================================

print("Loading reranker model...")

reranker = CrossEncoder(
    RERANKER_MODEL
)

print("Reranker model loaded.")


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
    query,
    documents,
    top_k=5,
    metric=None
):
    """
    Rerank retrieved documents using a CrossEncoder
    combined with financial metric relevance.

    documents should be Qdrant ScoredPoint objects.
    """

    if not documents:
        return []

    pairs = []

    for document in documents:

        payload = document.payload or {}

        text = payload.get(
            "text",
            ""
        )

        pairs.append(
            (
                query,
                text
            )
        )

    # --------------------------------------------------------
    # Generate CrossEncoder scores
    # --------------------------------------------------------

    cross_encoder_scores = reranker.predict(
        pairs
    )

    # --------------------------------------------------------
    # Calculate final ranking scores
    # --------------------------------------------------------

    ranked_documents = []

    for document, cross_score in zip(
        documents,
        cross_encoder_scores
    ):

        payload = document.payload or {}

        text = payload.get(
            "text",
            ""
        )

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
                "rerank_score": final_score,

                "cross_encoder_score":
                    float(cross_score),

                "metric_score":
                    metric_score,

                "document":
                    document
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