"""Verify that evaluation dataset chunk IDs exist in the FAISS index."""

from __future__ import annotations

import json
from pathlib import Path

from src.embeddings.embedder import load_vector_store


def verify_evaluation_dataset() -> dict[str, int | float]:
    """Load the FAISS index and verify evaluation chunk IDs exist.

    Returns a summary dict with:
    - total_cases: number of evaluation cases
    - cases_with_valid_ids: cases where at least one chunk ID exists
    - total_chunk_ids: total chunk IDs referenced
    - valid_chunk_ids: chunk IDs found in the index
    - coverage: fraction of cases with valid IDs
    """

    # Load the FAISS index and metadata
    try:
        index, chunks = load_vector_store()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: Could not load vector store: {exc}")
        print("Build the index first with: python -m src.embeddings.embedder")
        return {
            "total_cases": 0,
            "cases_with_valid_ids": 0,
            "total_chunk_ids": 0,
            "valid_chunk_ids": 0,
            "coverage": 0.0,
        }

    # Build a set of all chunk IDs in the index
    indexed_chunk_ids = {chunk.get("chunk_id") for chunk in chunks}

    # Load evaluation dataset
    eval_path = Path(__file__).resolve().parents[2] / "data" / "evaluation.json"
    with eval_path.open("r", encoding="utf-8") as f:
        cases = json.load(f)

    total_cases = len(cases)
    cases_with_valid_ids = 0
    total_chunk_ids = 0
    valid_chunk_ids = 0

    print(f"Loaded {len(indexed_chunk_ids):,} chunk IDs from FAISS index")
    print(f"Evaluating {total_cases} test cases...\n")

    for i, case in enumerate(cases, start=1):
        question = case["question"]
        relevant_ids = case["relevant_chunk_ids"]
        total_chunk_ids += len(relevant_ids)

        found = [cid for cid in relevant_ids if cid in indexed_chunk_ids]
        valid_chunk_ids += len(found)

        if found:
            cases_with_valid_ids += 1
            status = f"✓ {len(found)}/{len(relevant_ids)} chunks found"
        else:
            status = f"✗ 0/{len(relevant_ids)} chunks found"

        print(f"{i:2d}. {status:25s} | {question}")

    coverage = cases_with_valid_ids / total_cases if total_cases > 0 else 0.0

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total cases:              {total_cases}")
    print(f"Cases with valid IDs:     {cases_with_valid_ids}")
    print(f"Total chunk IDs:          {total_chunk_ids}")
    print(f"Valid chunk IDs:          {valid_chunk_ids}")
    print(f"Coverage:                 {coverage:.1%}")

    if coverage < 0.8:
        print("\n⚠️  WARNING: Less than 80% of cases have valid chunk IDs.")
        print("Consider rebuilding the FAISS index or updating evaluation.json.")
    else:
        print("\n✓ Evaluation dataset is ready for benchmarking.")

    return {
        "total_cases": total_cases,
        "cases_with_valid_ids": cases_with_valid_ids,
        "total_chunk_ids": total_chunk_ids,
        "valid_chunk_ids": valid_chunk_ids,
        "coverage": round(coverage, 4),
    }


if __name__ == "__main__":
    verify_evaluation_dataset()
