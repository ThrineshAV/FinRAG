from src.retrieval.retriever import retrieve_documents
from src.generation.llm import generate_answer


# ============================================================
# Build context from retrieved documents
# ============================================================

def build_context(results):
    """
    Convert retrieved FAISS results into a single
    context string for the LLM.
    """

    context_parts = []

    for rank, result in enumerate(
        results,
        start=1
    ):

        text = result.get(
            "text",
            ""
        )

        source = result.get(
            "source",
            "Unknown"
        )

        page_number = result.get(
            "page_number",
            "Unknown"
        )

        context_parts.append(
            f"""
SOURCE {rank}
Document: {source}
Page: {page_number}

{text}
"""
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# Run complete RAG pipeline
# ============================================================

def ask_financial_question(
    question: str
):
    """
    Complete RAG workflow:

    Question
        ↓
    Retrieval
        ↓
    Context
        ↓
    LLM
        ↓
    Answer
    """

    print("\n")
    print("=" * 80)
    print("FINSIGHT-RAG")
    print("=" * 80)

    print(f"\nQuestion:\n{question}")

    # --------------------------------------------------------
    # Step 1: Retrieve relevant documents
    # --------------------------------------------------------

    print("\nRetrieving relevant financial information...")

    results, _ = retrieve_documents(
        question,
        top_k=5
    )

    if not results:
        return {
            "answer": "No relevant information was found.",
            "sources": []
        }

    print(
        f"Retrieved {len(results)} relevant chunks."
    )

    # --------------------------------------------------------
    # Step 2: Build context
    # --------------------------------------------------------

    context = build_context(
        results
    )

    # --------------------------------------------------------
    # Step 3: Generate answer
    # --------------------------------------------------------

    print("\nGenerating answer with local LLM...")

    answer = generate_answer(
        question,
        context
    )

    # --------------------------------------------------------
    # Step 4: Extract sources
    # --------------------------------------------------------

    sources = []

    for result in results:

        source = result.get(
            "source",
            "Unknown"
        )

        page_number = result.get(
            "page_number",
            "Unknown"
        )

        score = result.get("score", 0.0)

        sources.append({
            "source": source,
            "page": page_number,
            "score": round(
                score,
                4
            )
        })

    return {
        "answer": answer,
        "sources": sources
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    question = (
        "What was Apple's total net sales "
        "in fiscal year 2025?"
    )

    result = ask_financial_question(
        question
    )

    print("\n")
    print("=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)

    print(
        f"\n{result['answer']}"
    )

    print("\n")
    print("=" * 80)
    print("SOURCES")
    print("=" * 80)

    for source in result["sources"]:

        print(
            f"\nDocument: {source['source']}"
        )

        print(
            f"Page: {source['page']}"
        )

        print(
            f"Similarity Score: "
            f"{source['score']}"
        )

    print("\nRAG pipeline completed successfully.")