"""LLM generation for the FinSight-RAG pipeline using Google Gemini API.

Provides generation paths:
- ``generate_answer()`` — local Ollama (used by the CLI pipeline).
- ``generate_answer_grounded()`` — Gemini SDK, non-streaming.
- ``generate_answer_grounded_stream()`` — Gemini SDK, yields token deltas.
"""

from __future__ import annotations

import os
from collections.abc import Generator

import requests
import google.generativeai as genai


# ============================================================
# Configuration
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"

GEMINI_MODEL = "gemini-3.6-flash"

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


def _gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", GEMINI_MODEL)


def _gemini_temperature() -> float:
    return float(os.getenv("GEMINI_TEMPERATURE", "0.1"))


def _gemini_max_tokens() -> int | None:
    value = os.getenv("GEMINI_MAX_TOKENS")
    return int(value) if value else None


def _build_system_prompt(context: str) -> str:
    """Select a comparison-aware system prompt when multiple companies appear."""
    companies_seen: set[str] = set()
    for line in context.splitlines():
        lower = line.lower()
        if lower.startswith("source:"):
            identifier = line.split(":")[1].split(";")[0].strip().lower()
            companies_seen.add(identifier.split("-")[0])
    if len(companies_seen) >= 2:
        return COMPARISON_SYSTEM_PROMPT
    return SYSTEM_PROMPT


def _get_gemini_client() -> genai.GenerativeModel:
    """Build a configured Gemini client from the environment."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name=_gemini_model(),
        generation_config=genai.types.GenerationConfig(
            temperature=_gemini_temperature(),
            max_output_tokens=_gemini_max_tokens(),
        ),
    )


def is_grounded_generation_available() -> bool:
    """Return whether grounded generation (Gemini) is available."""
    api_key = os.getenv("GEMINI_API_KEY")
    return bool(api_key)


# ============================================================
# Ollama generation  (CLI pipeline — unchanged)
# ============================================================

def generate_answer(
    question: str,
    context: str
) -> str:
    """
    Generate an answer using retrieved financial context via local Ollama.
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


# ============================================================
# Gemini SDK generation  (non-streaming)
# ============================================================

def generate_answer_grounded(question: str, context: str) -> str:
    """Generate a citation-aware answer using Gemini API."""
    model = _get_gemini_client()

    system_prompt = _build_system_prompt(context)
    user_message = f"SOURCES:\n{context}\n\nQUESTION:\n{question}"

    response = model.generate_content(
        [
            {"role": "user", "parts": [system_prompt, user_message]}
        ]
    )

    return response.text.strip()


# ============================================================
# Gemini SDK generation  (streaming — yields token deltas)
# ============================================================

def generate_answer_grounded_stream(
    question: str, context: str,
) -> Generator[str, None, None]:
    """Yield answer tokens as they arrive from the Gemini API."""
    model = _get_gemini_client()

    system_prompt = _build_system_prompt(context)
    user_message = f"SOURCES:\n{context}\n\nQUESTION:\n{question}"

    stream = model.generate_content(
        [
            {"role": "user", "parts": [system_prompt, user_message]}
        ],
        stream=True,
    )

    for chunk in stream:
        if chunk.text:
            yield chunk.text


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

    print("\nGenerating answer with Gemini...\n")

    answer = generate_answer_grounded(
        test_question,
        test_context
    )

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer)
