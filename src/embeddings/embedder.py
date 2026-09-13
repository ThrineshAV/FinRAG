"""Generate and persist embeddings for financial document chunks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
INDEX_PATH = VECTOR_DB_DIR / "financial_documents.faiss"
METADATA_PATH = VECTOR_DB_DIR / "financial_documents.json"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once, on first use."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def load_chunks() -> list[dict[str, Any]]:
    """Load all structured chunks from the project data directory."""
    chunk_files = sorted(CHUNKS_DIR.glob("*_chunks.json"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in {CHUNKS_DIR}")
    chunks: list[dict[str, Any]] = []
    for chunk_file in chunk_files:
        with chunk_file.open("r", encoding="utf-8") as file:
            chunks.extend(json.load(file))
    return chunks


def generate_embeddings(chunks: list[dict[str, Any]]) -> np.ndarray:
    """Generate normalized vectors for chunk text."""
    if not chunks:
        raise ValueError("Cannot embed an empty chunk collection")
    embeddings = get_embedding_model().encode(
        [chunk["text"] for chunk in chunks],
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings, dtype="float32")


def store_embeddings(chunks: list[dict[str, Any]], embeddings: np.ndarray) -> None:
    """Append chunks to, or create, a cosine-similarity FAISS store."""
    vectors = np.asarray(embeddings, dtype="float32")
    if len(chunks) != len(vectors):
        raise ValueError("Each chunk must have one embedding")
    if vectors.ndim != 2 or not len(vectors):
        raise ValueError("Embeddings must be a non-empty matrix")

    # Compute SHA256 hashes for deduplication
    chunk_hashes = [hashlib.sha256(chunk["text"].encode()).hexdigest() for chunk in chunks]

    # Load existing hashes if metadata exists
    existing_hashes = set()
    if METADATA_PATH.exists():
        try:
            existing_data = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            existing_hashes = {item.get("chunk_hash") for item in existing_data if "chunk_hash" in item}
        except (json.JSONDecodeError, IOError):
            # If metadata is corrupt, treat as empty to avoid blocking
            existing_hashes = set()

    # Filter to new chunks only (track which original indices were kept)
    new_chunks = []
    new_hashes = []
    kept_indices = []
    for i, (chunk, chunk_hash) in enumerate(zip(chunks, chunk_hashes)):
        if chunk_hash not in existing_hashes:
            new_chunks.append(chunk)
            new_hashes.append(chunk_hash)
            kept_indices.append(i)
            existing_hashes.add(chunk_hash)

    if not new_chunks:
        return  # Nothing to add

    # Filter vectors to only include kept chunks
    new_vectors = vectors[kept_indices]

    # Build metadata with hash
    metadata = [
        {
            "chunk_id": chunk.get("chunk_id", str(chunk["chunk_index"])),
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
            "chunk_hash": chunk_hash,
            **chunk.get("metadata", {}),
        }
        for chunk, chunk_hash in zip(new_chunks, new_hashes)
    ]

    if INDEX_PATH.exists() and METADATA_PATH.exists():
        index, existing_metadata = load_vector_store()
        if index.d != new_vectors.shape[1]:
            raise ValueError("Embedding dimensions do not match the existing index")
        index.add(new_vectors)
        metadata = existing_metadata + metadata
    else:
        index = faiss.IndexFlatIP(new_vectors.shape[1])
        index.add(new_vectors)

    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_vector_store() -> tuple[faiss.Index, list[dict[str, Any]]]:
    """Load the FAISS index and its aligned metadata."""
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("FAISS index is unavailable. Run embedder.py first.")
    index = faiss.read_index(str(INDEX_PATH))
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if index.ntotal != len(metadata):
        raise ValueError("FAISS index and metadata counts do not match")
    return index, metadata


if __name__ == "__main__":
    chunks = load_chunks()
    store_embeddings(chunks, generate_embeddings(chunks))
    print(f"Stored {len(chunks)} vectors in {INDEX_PATH}")
