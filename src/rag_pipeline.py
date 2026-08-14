from retrieval.retriever import retrieve_documents
from generation.llm import generate_answer


# ============================================================
# Build context from retrieved documents
# ============================================================

def build_context(results):
    """
    Convert retrieved Qdrant results into a single
    context string for the LLM.
    """

    context_parts = []

    for rank, result in enumerate(
        results,
        start=1
    ):

        payload = result.payload or {}

        text = payload.get(
            "text",
            ""
        )

        source = payload.get(
            "source",
            "Unknown"
        )

        chunk_index = payload.get(
            "chunk_index",
            "Unknown"
        )

        context_parts.append(
            f"""
SOURCE {rank}
Document: {source}
Chunk: {chunk_index}

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

    results = retrieve_documents(
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

        payload = result.payload or {}

        source = payload.get(
            "source",
            "Unknown"
        )

        chunk_index = payload.get(
            "chunk_index",
            "Unknown"
        )

        score = result.score

        sources.append({
            "source": source,
            "chunk": chunk_index,
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
            f"Chunk: {source['chunk']}"
        )

        print(
            f"Similarity Score: "
            f"{source['score']}"
        )

    print("\nRAG pipeline completed successfully.")