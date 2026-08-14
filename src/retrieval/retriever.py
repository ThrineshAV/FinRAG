from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


# ============================================================
# Configuration
# ============================================================

QDRANT_PATH = "data/qdrant"

COLLECTION_NAME = "financial_documents"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

TOP_K = 5


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
# Retrieve relevant documents
# ============================================================

def retrieve_documents(
    query: str,
    top_k: int = TOP_K,
    ticker=None,
    fiscal_year=None,
    filing_type=None
):
    """
    Retrieve relevant chunks using semantic search
    with optional metadata filtering.
    """

    # --------------------------------------------------------
    # Create query embedding
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # --------------------------------------------------------
    # Build optional filter
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

    return search_results


# ============================================================
# Display results
# ============================================================

def display_results(
    query,
    results
):

    print("\n")
    print("=" * 80)
    print("FINANCIAL RAG RETRIEVAL")
    print("=" * 80)

    print(
        f"\nQuestion:\n{query}"
    )

    print(
        "\nRetrieved Documents:"
    )

    print("-" * 80)

    if not results:

        print(
            "No relevant documents found."
        )

        return

    for rank, result in enumerate(
        results,
        start=1
    ):

        payload = (
            result.payload or {}
        )

        print(
            f"\nRESULT {rank}"
        )

        print(
            f"Score: {result.score:.4f}"
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
            )
        )

        print("-" * 80)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    question = (
        "What was Apple's total net sales "
        "in fiscal year 2025?"
    )

    results = retrieve_documents(
        query=question,

        top_k=5,

        ticker="AAPL",

        fiscal_year="2025",

        filing_type="10-K"
    )

    display_results(
        question,
        results
    )

    print(
        "\nRetrieval completed successfully."
    )