"""LLM generation for the FinSight-RAG pipeline.

Provides three generation paths:
- ``generate_answer()`` — local Ollama (used by the CLI pipeline).
- ``generate_openai_answer()`` — OpenAI SDK, non-streaming.
- ``generate_openai_answer_stream()`` — OpenAI SDK, yields token deltas.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from typing import Any

import requests
from openai import OpenAI


# ============================================================
# Configuration
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_NAME = "llama3.2:3b"
OPENAI_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are a financial research assistant. Answer only "
    "from the supplied sources. If the sources do not "
    "support the answer, say so. Preserve units and cite "
    "sources as [Source: <document>, page <number>]."
)

COMPARISON_SYSTEM_PROMPT = (
    "You are a financial research assistant. The sources below cover "
    "multiple companies. Structure your answer as a comparison: present "
    "each company's figures side by side, note differences, and cite "
    "sources as [Source: <document>, page <number>]. If a source does "
    "not contain enough information for a company, say so rather than "
    "guessing."
)


def _openai_model() -> str:
    return os.getenv("OPENAI_MODEL", OPENAI_MODEL)


def _openai_temperature() -> float:
    return float(os.getenv("OPENAI_TEMPERATURE", "0.1"))


def _openai_max_tokens() -> int | None:
    value = os.getenv("OPENAI_MAX_TOKENS")
    return int(value) if value else None


def _build_system_prompt(context: str) -> str:
    """Select a comparison-aware system prompt when multiple companies appear."""
    companies_seen: set[str] = set()
    for line in context.splitlines():
        lower = line.lower()
        if lower.startswith("source:"):
            # Lines formatted as "Source: apple-2024; Page: 3"
            # Extract the document identifier before the semicolon.
            identifier = line.split(":")[1].split(";")[0].strip().lower()
            # The document ID typically starts with the company name.
            companies_seen.add(identifier.split("-")[0])
    if len(companies_seen) >= 2:
        return COMPARISON_SYSTEM_PROMPT
    return SYSTEM_PROMPT


def _get_openai_client() -> OpenAI:
    """Build a configured OpenAI client from the environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured")
    return OpenAI(api_key=api_key)


# ============================================================
# Ollama generation  (CLI pipeline — unchanged)
# ============================================================

def generate_answer(
    question: str,
    context: str
) -> str:
    """
    Generate an answer using retrieved financial context.
    """

    prompt = f"""
You are a financial research assistant.

Answer the user's question using ONLY the information
provided in the context below.

Rules:
1. Do not invent financial information.
2. If the context does not contain enough information,
   say that the information is not available in the
   provided documents.
3. Give a concise and factual answer.
4. Preserve financial units such as million or billion.
5. When numbers are provided, do not change them.
6. Mention the source information when available.

CONTEXT:
-------------------------
{context}
-------------------------

USER QUESTION:
{question}

ANSWER:
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    result = response.json()

    return result["response"].strip()


def is_openai_configured() -> bool:
    """Return whether the optional grounded OpenAI path can be used."""
    return bool(os.getenv("OPENAI_API_KEY"))


# ============================================================
# OpenAI SDK generation  (non-streaming)
# ============================================================

def generate_openai_answer(question: str, context: str) -> str:
    """Generate a citation-aware answer using only retrieved context."""
    client = _get_openai_client()

    kwargs: dict[str, Any] = {
        "model": _openai_model(),
        "temperature": _openai_temperature(),
        "messages": [
            {"role": "system", "content": _build_system_prompt(context)},
            {"role": "user", "content": f"SOURCES:\n{context}\n\nQUESTION:\n{question}"},
        ],
    }
    max_tokens = _openai_max_tokens()
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    completion = client.chat.completions.create(**kwargs)
    return completion.choices[0].message.content.strip()


# ============================================================
# OpenAI SDK generation  (streaming — yields token deltas)
# ============================================================

def generate_openai_answer_stream(
    question: str, context: str,
) -> Generator[str, None, None]:
    """Yield answer tokens as they arrive from the OpenAI API."""
    client = _get_openai_client()

    kwargs: dict[str, Any] = {
        "model": _openai_model(),
        "temperature": _openai_temperature(),
        "stream": True,
        "messages": [
            {"role": "system", "content": _build_system_prompt(context)},
            {"role": "user", "content": f"SOURCES:\n{context}\n\nQUESTION:\n{question}"},
        ],
    }
    max_tokens = _openai_max_tokens()
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    stream = client.chat.completions.create(**kwargs)
    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            yield delta.content


# ============================================================
# Test generation independently
# ============================================================

if __name__ == "__main__":

    test_context = """
Apple Inc. reported total net sales of
$416,161 million in fiscal year 2025.
Total net sales were $391,035 million in 2024.
"""

    test_question = (
        "What was Apple's total net sales "
        "in fiscal year 2025?"
    )

    print("\nGenerating answer...\n")

    answer = generate_answer(
        test_question,
        test_context
    )

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer)
