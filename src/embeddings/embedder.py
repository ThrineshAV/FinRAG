from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid


# ============================================================
# Configuration
# ============================================================

CHUNKS_DIR = Path("data/chunks")
QDRANT_PATH = "data/qdrant"

COLLECTION_NAME = "financial_documents"

# Free, local embedding model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# bge-small-en-v1.5 produces 384-dimensional vectors
VECTOR_SIZE = 384


# ============================================================
# Initialize embedding model
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
# Create Qdrant collection
# ============================================================

def create_collection():

    existing_collections = [
        collection.name
        for collection in qdrant_client
        .get_collections()
        .collections
    ]

    if COLLECTION_NAME not in existing_collections:

        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )

        print(
            f"Created collection: {COLLECTION_NAME}"
        )

    else:

        print(
            f"Collection already exists: "
            f"{COLLECTION_NAME}"
        )


# ============================================================
# Load chunks
# ============================================================

def load_chunks():

    chunk_files = list(
        CHUNKS_DIR.glob("*_chunks.txt")
    )

    if not chunk_files:
        raise FileNotFoundError(
            "No chunk files found in data/chunks/"
        )

    chunks_file = chunk_files[0]

    print(
        f"Loading chunks from: "
        f"{chunks_file.name}"
    )

    with open(
        chunks_file,
        "r",
        encoding="utf-8"
    ) as file:

        content = file.read()

    raw_chunks = content.split(
        "=" * 80
    )

    chunks = []

    for chunk in raw_chunks:

        chunk = chunk.strip()

        if not chunk:
            continue

        if chunk.startswith("CHUNK"):
            continue

        chunks.append(chunk)

    return chunks, chunks_file


# ============================================================
# Generate embeddings
# ============================================================

def generate_embeddings(chunks):

    print(
        f"Generating embeddings for "
        f"{len(chunks)} chunks..."
    )

    embeddings = embedding_model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return embeddings


# ============================================================
# Store embeddings in Qdrant
# ============================================================

def store_embeddings(
    chunks,
    embeddings,
    source_file
):

    points = []

    for index, (chunk, embedding) in enumerate(
        zip(chunks, embeddings)
    ):

        point = PointStruct(
            id=str(uuid.uuid4()),

            vector=embedding.tolist(),

            payload={
                "text": chunk,
                "chunk_index": index,
                "source": source_file.stem
            }
        )

        points.append(point)

    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print(
        f"Stored {len(points)} vectors in Qdrant."
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

    chunks, source_file = load_chunks()

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
        embeddings,
        source_file
    )

    print("\nEmbedding pipeline completed successfully.")
    print("=" * 60)