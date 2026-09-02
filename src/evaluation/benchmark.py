"""Run the local retrieval benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.metrics import (
    EvaluationCase,
    evaluate_answer_quality,
    evaluate_retrieval,
    retrieve_for_evaluation,
)


BENCHMARK_PATH = Path(__file__).resolve().parents[2] / "data" / "evaluation.json"


def load_cases(path: Path = BENCHMARK_PATH) -> list[EvaluationCase]:
    """Load benchmark cases from a JSON file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _run_with_generation(cases: list[EvaluationCase]) -> dict:
    """Run retrieval + generation benchmark with answer quality scoring."""
    from src.generation.llm import generate_answer_grounded
    from src.retrieval.retriever import retrieve_documents

    retrieval_metrics = evaluate_retrieval(cases, retrieve_for_evaluation)

    faithfulness_total = 0.0
    relevance_total = 0.0

    for case in cases:
        results, _ = retrieve_documents(case["question"], top_k=5)
        context = "\n\n".join(
            f"Source: {r.get('source', 'unknown')}; "
            f"Page: {r.get('page_number', 'unknown')}\n{r.get('text', '')}"
            for r in results
        )
        answer = generate_answer_grounded(case["question"], context)
        quality = evaluate_answer_quality(case["question"], answer, context)
        faithfulness_total += quality["faithfulness"]
        relevance_total += quality["relevance"]

    case_count = len(cases)
    retrieval_metrics["avg_faithfulness"] = round(faithfulness_total / case_count, 4)
    retrieval_metrics["avg_relevance"] = round(relevance_total / case_count, 4)
    return retrieval_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the retrieval benchmark")
    parser.add_argument(
        "--include-generation",
        action="store_true",
        help="Also run generation and report answer quality metrics (requires GEMINI_API_KEY)",
    )
    args = parser.parse_args()

    cases = load_cases()

    if args.include_generation:
        results = _run_with_generation(cases)
    else:
        results = evaluate_retrieval(cases, retrieve_for_evaluation)

    print(json.dumps(results, indent=2))
