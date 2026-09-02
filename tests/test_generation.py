"""Tests for Stage 3: answer generation, streaming, and quality metrics."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from src import api
from src.evaluation.metrics import evaluate_answer_quality
from src.generation import llm

# Disable authentication for generation tests
os.environ["AUTH_REQUIRED"] = "false"


# ============================================================
# 1. Gemini SDK sends a grounded prompt
# ============================================================

def test_gemini_sdk_sends_grounded_prompt(monkeypatch) -> None:
    """generate_answer_grounded() should use the Gemini SDK and embed the
    context into the user message so the model only sees grounded sources."""

    class FakePart:
        def __init__(self, text):
            self.text = text

    class FakeResponse:
        text = "Revenue was $50 billion."

    model_instance = MagicMock()
    model_instance.generate_content.return_value = FakeResponse()

    def fake_get_client():
        return model_instance

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(llm, "_get_gemini_client", fake_get_client)

    answer = llm.generate_answer_grounded(
        "What was Apple's revenue?",
        "Source: apple-2024; Page: 5\nRevenue was $50 billion.",
    )

    assert answer == "Revenue was $50 billion."
    # Verify Gemini was called with content parts
    call_args = model_instance.generate_content.call_args
    parts = call_args[0][0]  # first positional arg is the content list
    assert len(parts) == 1
    assert "financial research assistant" in parts[0]["parts"][0].lower()
    assert "Revenue was $50 billion." in parts[0]["parts"][1]


# ============================================================
# 2. Streaming yields token deltas
# ============================================================

def test_gemini_stream_yields_token_deltas(monkeypatch) -> None:
    """generate_answer_grounded_stream() should yield individual tokens."""

    class FakeChunk:
        def __init__(self, text):
            self.text = text

    chunks = [FakeChunk("Hello"), FakeChunk(" world"), FakeChunk("!")]

    model_instance = MagicMock()
    model_instance.generate_content.return_value = iter(chunks)

    def fake_get_client():
        return model_instance

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(llm, "_get_gemini_client", fake_get_client)

    collected = list(llm.generate_answer_grounded_stream(
        "What was revenue?", "Revenue was $10."
    ))

    assert collected == ["Hello", " world", "!"]
    assert "".join(collected) == "Hello world!"


# ============================================================
# 3. Streaming SSE endpoint
# ============================================================

def test_stream_endpoint_returns_sse_events(monkeypatch) -> None:
    """POST /query/stream should return SSE data lines with tokens and
    a final done event carrying citations."""
    monkeypatch.setattr(
        api,
        "retrieve_documents",
        lambda question, top_k, filters: (
            [{
                "chunk_id": "apple-p1-c0",
                "text": "Revenue was $50 billion.",
                "source": "apple-2024",
                "page_number": 1,
                "score": 0.95,
            }],
            {},
        ),
    )
    monkeypatch.setattr(api, "is_grounded_generation_available", lambda: True)

    def fake_stream(question, context):
        yield "Revenue"
        yield " was"
        yield " $50B."

    monkeypatch.setattr(api, "generate_answer_grounded_stream", fake_stream)

    client = TestClient(api.app)
    response = client.post(
        "/query/stream",
        json={"question": "What was Apple's revenue?"},
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    lines = [line for line in response.text.split("\n") if line.startswith("data:")]
    assert len(lines) >= 4  # 3 token events + 1 done event

    # Check token events
    import json
    first_event = json.loads(lines[0].removeprefix("data: "))
    assert "token" in first_event
    assert first_event["token"] == "Revenue"

    # Check final done event
    last_event = json.loads(lines[-1].removeprefix("data: "))
    assert last_event["done"] is True
    assert len(last_event["citations"]) == 1
    assert last_event["citations"][0]["chunk_id"] == "apple-p1-c0"


# ============================================================
# 4. Comparison prompt detection
# ============================================================

def test_comparison_prompt_detects_multiple_companies() -> None:
    """_build_system_prompt should switch to the comparison prompt when
    the context contains sources from more than one company."""
    context = (
        "Source: apple-2024; Page: 1\nApple revenue was $400B.\n\n"
        "Source: nvidia-2024; Page: 3\nNVIDIA revenue was $60B."
    )

    prompt = llm._build_system_prompt(context)

    assert "comparison" in prompt.lower()
    assert "side by side" in prompt.lower()


def test_single_company_uses_standard_prompt() -> None:
    """When all sources come from a single company, the standard prompt
    should be used (no comparison language)."""
    context = (
        "Source: apple-2024; Page: 1\nApple revenue was $400B.\n\n"
        "Source: apple-2024; Page: 5\nApple net income was $100B."
    )

    prompt = llm._build_system_prompt(context)

    assert "comparison" not in prompt.lower()


# ============================================================
# 5. Faithfulness detects hallucination
# ============================================================

def test_faithfulness_score_detects_hallucination() -> None:
    """An answer containing information not in the context should score
    lower on faithfulness than a fully grounded answer."""
    context = "Apple reported revenue of $400 billion in fiscal year 2024."

    grounded_answer = "Apple reported revenue of $400 billion."
    hallucinated_answer = "Apple reported revenue of $400 billion and net income of $200 billion."

    grounded_score = evaluate_answer_quality(
        "What was Apple's revenue?", grounded_answer, context
    )
    hallucinated_score = evaluate_answer_quality(
        "What was Apple's revenue?", hallucinated_answer, context
    )

    assert grounded_score["faithfulness"] > hallucinated_score["faithfulness"]


# ============================================================
# 6. Relevance checks metric presence
# ============================================================

def test_relevance_score_checks_metric_presence() -> None:
    """When the question asks about 'revenue', an answer that mentions
    'revenue' should score relevance = 1.0, and one that omits it
    should score 0.0."""
    context = "Apple reported revenue of $400 billion."

    relevant_answer = "Apple's revenue was $400 billion in fiscal year 2024."
    irrelevant_answer = "Apple is a technology company based in Cupertino."

    relevant_score = evaluate_answer_quality(
        "What was Apple's revenue?", relevant_answer, context
    )
    irrelevant_score = evaluate_answer_quality(
        "What was Apple's revenue?", irrelevant_answer, context
    )

    assert relevant_score["relevance"] == 1.0
    assert irrelevant_score["relevance"] == 0.0
