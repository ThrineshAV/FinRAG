"""Generate and persist embeddings for financial document chunks."""

from __future__ import annotations

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
    """Persist a cosine-similarity FAISS index and aligned metadata."""
    vectors = np.asarray(embeddings, dtype="float32")
    if len(chunks) != len(vectors):
        raise ValueError("Each chunk must have one embedding")
    if vectors.ndim != 2 or not len(vectors):
        raise ValueError("Embeddings must be a non-empty matrix")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    metadata = [
        {
            "chunk_id": chunk.get("chunk_id", str(chunk["chunk_index"])),
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
            **chunk.get("metadata", {}),
        }
        for chunk in chunks
    ]

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
