from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from query_parser import parse_query
from reranker import rerank_documents


# ============================================================
# Configuration
# ============================================================

QDRANT_PATH = "data/qdrant"
COLLECTION_NAME = "financial_documents"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Retrieve 20 candidates before reranking
TOP_K = 20

# Keep only the best 5 after reranking
RERANK_TOP_K = 5


# ============================================================
# Initialize
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")

qdrant_client = QdrantClient(
    path=QDRANT_PATH
)


# ============================================================
# Build metadata filter
# ============================================================

def build_metadata_filter(
    ticker=None,
    fiscal_year=None,
    filing_type=None
):
    """
    Build an optional Qdrant metadata filter.
    """

    conditions = []

    if ticker:
        conditions.append(
            FieldCondition(
                key="ticker",
                match=MatchValue(
                    value=ticker
                )
            )
        )

    if fiscal_year:
        conditions.append(
            FieldCondition(
                key="fiscal_year",
                match=MatchValue(
                    value=fiscal_year
                )
            )
        )

    if filing_type:
        conditions.append(
            FieldCondition(
                key="filing_type",
                match=MatchValue(
                    value=filing_type
                )
            )
        )

    if not conditions:
        return None

    return Filter(
        must=conditions
    )


# ============================================================
# Retrieve documents
# ============================================================

def retrieve_documents(
    query: str,
    top_k: int = TOP_K
):
    """
    Parse the user query and retrieve relevant
    financial documents from Qdrant.

    Returns:
        search_results
        parsed_query
    """

    # --------------------------------------------------------
    # Query understanding
    # --------------------------------------------------------

    parsed_query = parse_query(
        query
    )

    ticker = parsed_query["ticker"]
    fiscal_year = parsed_query["fiscal_year"]
    filing_type = parsed_query["filing_type"]
    metric = parsed_query["metric"]

    # --------------------------------------------------------
    # Display parsed information
    # --------------------------------------------------------

    print("\nQuery Understanding:")

    print(
        f"Company: "
        f"{parsed_query['company']}"
    )

    print(
        f"Ticker: "
        f"{ticker}"
    )

    print(
        f"Fiscal Year: "
        f"{fiscal_year}"
    )

    print(
        f"Filing Type: "
        f"{filing_type}"
    )

    print(
        f"Metric: "
        f"{metric}"
    )

    # --------------------------------------------------------
    # Create query embedding
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # --------------------------------------------------------
    # Build metadata filter
    # --------------------------------------------------------

    metadata_filter = build_metadata_filter(
        ticker=ticker,
        fiscal_year=fiscal_year,
        filing_type=filing_type
    )

    # --------------------------------------------------------
    # Search Qdrant
    # --------------------------------------------------------

    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        query_filter=metadata_filter,
        limit=top_k,
        with_payload=True
    ).points

    return search_results, parsed_query


# ============================================================
# Display reranked results
# ============================================================

def display_reranked_results(
    query,
    results
):

    print("\n")
    print("=" * 80)
    print("RERANKED RESULTS")
    print("=" * 80)

    print(
        f"\nQuestion:\n{query}"
    )

    print("-" * 80)

    if not results:

        print(
            "No relevant documents found."
        )

        return

    for rank, item in enumerate(
        results,
        start=1
    ):

        document = item["document"]

        rerank_score = item[
            "rerank_score"
        ]

        cross_encoder_score = item.get(
            "cross_encoder_score",
            0.0
        )

        metric_score = item.get(
            "metric_score",
            0.0
        )

        payload = (
            document.payload or {}
        )

        print(
            f"\nRERANKED RESULT {rank}"
        )

        print(
            f"Final Rerank Score: "
            f"{rerank_score:.4f}"
        )

        print(
            f"CrossEncoder Score: "
            f"{cross_encoder_score:.4f}"
        )

        print(
            f"Metric Relevance: "
            f"{metric_score:.4f}"
        )

        print(
            f"Original Score: "
            f"{document.score:.4f}"
        )

        print(
            f"Company: "
            f"{payload.get('company', 'Unknown')}"
        )

        print(
            f"Ticker: "
            f"{payload.get('ticker', 'Unknown')}"
        )

        print(
            f"Filing: "
            f"{payload.get('filing_type', 'Unknown')}"
        )

        print(
            f"Fiscal Year: "
            f"{payload.get('fiscal_year', 'Unknown')}"
        )

        print(
            f"Filing Date: "
            f"{payload.get('filing_date', 'Unknown')}"
        )

        print(
            f"Chunk: "
            f"{payload.get('chunk_index', 'Unknown')}"
        )

        print("\nText:")

        print(
            payload.get(
                "text",
                ""
            )[:1000]
        )

        print("-" * 80)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    question = (
        "What was NVIDIA's net income "
        "in fiscal year 2026?"
    )

    # --------------------------------------------------------
    # Step 1: Retrieve 20 candidates
    # --------------------------------------------------------

    results, parsed_query = retrieve_documents(
        query=question,
        top_k=TOP_K
    )

    print("\n")
    print("=" * 80)
    print(
        f"Retrieved {len(results)} "
        f"candidate documents."
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Get requested metric
    # --------------------------------------------------------

    metric = parsed_query["metric"]

    # --------------------------------------------------------
    # Step 2: Rerank the 20 candidates
    # --------------------------------------------------------

    reranked_results = rerank_documents(
        query=question,
        documents=results,
        top_k=RERANK_TOP_K,
        metric=metric
    )

    # --------------------------------------------------------
    # Step 3: Display top 5 after reranking
    # --------------------------------------------------------

    display_reranked_results(
        question,
        reranked_results
    )

    print(
        "\nReranking completed successfully."
    )