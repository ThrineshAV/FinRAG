from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


# ============================================================
# Configuration
# ============================================================

QDRANT_PATH = "data/qdrant"

COLLECTION_NAME = "financial_documents"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

TOP_K = 5


# ============================================================
# Initialize models and database
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
# Retrieve relevant chunks
# ============================================================

def retrieve_documents(
    query: str,
    top_k: int = TOP_K
):
    """
    Convert the user query into an embedding
    and retrieve the most relevant chunks from Qdrant.
    """

    # Create query embedding
    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # Search Qdrant
    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        with_payload=True
    ).points

    return search_results


# ============================================================
# Display retrieval results
# ============================================================

def display_results(
    query: str,
    results
):
    """
    Display retrieved chunks in a readable format.
    """

    print("\n")
    print("=" * 80)
    print("FINANCIAL RAG RETRIEVAL")
    print("=" * 80)

    print(f"\nQuestion:")
    print(query)

    print("\nRetrieved Documents:")
    print("-" * 80)

    if not results:
        print("No relevant documents found.")
        return

    for rank, result in enumerate(
        results,
        start=1
    ):

        payload = result.payload or {}

        score = result.score

        chunk_index = payload.get(
            "chunk_index",
            "N/A"
        )

        source = payload.get(
            "source",
            "Unknown"
        )

        text = payload.get(
            "text",
            ""
        )

        print(f"\nRESULT {rank}")
        print(f"Score: {score:.4f}")
        print(f"Source: {source}")
        print(f"Chunk: {chunk_index}")

        print("\nText:")
        print(text)

        print("-" * 80)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("\nStarting retrieval system...")

    question = (
        "What was Apple's total net sales "
        "in fiscal year 2025?"
    )

    results = retrieve_documents(
        question
    )

    display_results(
        question,
        results
    )

    print("\nRetrieval completed successfully.")