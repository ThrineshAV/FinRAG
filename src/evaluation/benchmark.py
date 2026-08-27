"""Run the local retrieval benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.metrics import EvaluationCase, evaluate_retrieval, retrieve_for_evaluation


BENCHMARK_PATH = Path(__file__).resolve().parents[2] / "data" / "evaluation.json"


def load_cases(path: Path = BENCHMARK_PATH) -> list[EvaluationCase]:
    """Load benchmark cases from a JSON file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    print(json.dumps(evaluate_retrieval(load_cases(), retrieve_for_evaluation), indent=2))