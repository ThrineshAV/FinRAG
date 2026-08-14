import json
import uuid
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)


# ============================================================
# Configuration
# ============================================================

CHUNKS_DIR = Path("data/chunks")

QDRANT_PATH = "data/qdrant"

COLLECTION_NAME = "financial_documents"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

VECTOR_SIZE = 384


# ============================================================
# Load embedding model
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded successfully.")


# ============================================================
# Initialize Qdrant
# ============================================================

qdrant_client = QdrantClient(
    path=QDRANT_PATH
)


# ============================================================
# Create collection
# ============================================================

def create_collection():

    collections = (
        qdrant_client
        .get_collections()
        .collections
    )

    existing_names = [
        collection.name
        for collection in collections
    ]

    if COLLECTION_NAME not in existing_names:

        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,

            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )

        print(
            f"Created collection: "
            f"{COLLECTION_NAME}"
        )

    else:

        print(
            f"Collection already exists: "
            f"{COLLECTION_NAME}"
        )


# ============================================================
# Load structured chunks
# ============================================================

def load_chunks():

    chunk_files = list(
        CHUNKS_DIR.glob("*_chunks.json")
    )

    if not chunk_files:

        raise FileNotFoundError(
            "No structured chunk files found "
            "in data/chunks/"
        )

    all_chunks = []

    for chunk_file in chunk_files:

        print(
            f"Loading: {chunk_file.name}"
        )

        with open(
            chunk_file,
            "r",
            encoding="utf-8"
        ) as file:

            chunks = json.load(file)

        all_chunks.extend(
            chunks
        )

    return all_chunks


# ============================================================
# Generate embeddings
# ============================================================

def generate_embeddings(
    chunks
):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        f"Generating embeddings for "
        f"{len(texts)} chunks..."
    )

    embeddings = embedding_model.encode(
        texts,

        batch_size=32,

        show_progress_bar=True,

        normalize_embeddings=True
    )

    return embeddings


# ============================================================
# Store embeddings
# ============================================================

def store_embeddings(
    chunks,
    embeddings
):

    points = []

    for index, (
        chunk,
        embedding
    ) in enumerate(
        zip(chunks, embeddings)
    ):

        metadata = chunk.get(
            "metadata",
            {}
        )

        payload = {

            # Main chunk information
            "text": chunk["text"],

            "chunk_index":
                chunk["chunk_index"],

            # Financial metadata
            "company":
                metadata.get("company"),

            "ticker":
                metadata.get("ticker"),

            "cik":
                metadata.get("cik"),

            "filing_type":
                metadata.get("filing_type"),

            "filing_date":
                metadata.get("filing_date"),

            "report_date":
                metadata.get("report_date"),

            "fiscal_year":
                metadata.get("fiscal_year"),

            "accession_number":
                metadata.get("accession_number"),

            "primary_document":
                metadata.get("primary_document"),

            "source_url":
                metadata.get("source_url"),
        }

        point = PointStruct(

            id=str(
                uuid.uuid4()
            ),

            vector=embedding.tolist(),

            payload=payload
        )

        points.append(
            point
        )

    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,

        points=points
    )

    print(
        f"Stored {len(points)} vectors "
        f"in Qdrant."
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("FinSight-RAG Embedding Pipeline")
    print("=" * 60)

    create_collection()

    chunks = load_chunks()

    print(
        f"Loaded {len(chunks)} chunks."
    )

    embeddings = generate_embeddings(
        chunks
    )

    print(
        f"Generated {len(embeddings)} embeddings."
    )

    store_embeddings(
        chunks,
        embeddings
    )

    print(
        "\nEmbedding pipeline completed successfully."
    )

    print("=" * 60)