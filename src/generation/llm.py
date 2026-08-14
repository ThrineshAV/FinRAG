import requests


# ============================================================
# Configuration
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_NAME = "llama3.2:3b"


# ============================================================
# Generate answer
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