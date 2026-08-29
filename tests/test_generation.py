"""Tests for Stage 3: answer generation, streaming, and quality metrics."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from src import api
from src.evaluation.metrics import evaluate_answer_quality
from src.generation import llm


# ============================================================
# 1. OpenAI SDK sends a grounded prompt
# ============================================================

def test_openai_sdk_sends_grounded_prompt(monkeypatch) -> None:
    """generate_openai_answer() should use the OpenAI SDK and embed the
    context into the user message so the model only sees grounded sources."""
    captured: dict = {}

    class FakeMessage:
        content = "Revenue was $50 billion."

    class FakeChoice:
        message = FakeMessage()

    class FakeCompletion:
        choices = [FakeChoice()]

    def fake_create(**kwargs):
        captured.update(kwargs)
        return FakeCompletion()

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = fake_create

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(llm, "_get_openai_client", lambda: fake_client)

    answer = llm.generate_openai_answer(
        "What was Apple's revenue?",
        "Source: apple-2024; Page: 5\nRevenue was $50 billion.",
    )

    assert answer == "Revenue was $50 billion."
    assert captured["messages"][0]["role"] == "system"
    assert "financial research assistant" in captured["messages"][0]["content"].lower()
    assert captured["messages"][1]["content"].startswith("SOURCES:")
    assert "Revenue was $50 billion." in captured["messages"][1]["content"]
    assert captured["model"] == "gpt-4o-mini"


# ============================================================
# 2. Streaming yields token deltas
# ============================================================

def test_openai_stream_yields_token_deltas(monkeypatch) -> None:
    """generate_openai_answer_stream() should yield individual tokens."""

    class FakeDelta:
        def __init__(self, content):
            self.content = content

    class FakeStreamChoice:
        def __init__(self, content):
            self.delta = FakeDelta(content)

    class FakeStreamChunk:
        def __init__(self, content):
            self.choices = [FakeStreamChoice(content)]

    chunks = [FakeStreamChunk("Hello"), FakeStreamChunk(" world"), FakeStreamChunk("!")]

    def fake_create(**kwargs):
        assert kwargs.get("stream") is True
        return iter(chunks)

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = fake_create

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(llm, "_get_openai_client", lambda: fake_client)

    collected = list(llm.generate_openai_answer_stream(
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
    monkeypatch.setattr(api, "is_openai_configured", lambda: True)

    def fake_stream(question, context):
        yield "Revenue"
        yield " was"
        yield " $50B."

    monkeypatch.setattr(api, "generate_openai_answer_stream", fake_stream)

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
