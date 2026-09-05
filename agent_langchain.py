"""LangChain agent consuming /query backend with JWT Bearer auth.
Resume line: "Built RAG pipeline with JWT auth; integrated via LangChain agent tool."
"""
import os
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import Tool

BASE = os.getenv("API_URL", "http://localhost:8000")
TOKEN = os.getenv("JWT_TOKEN")           # get via /auth/login first
GEMINI_KEY = os.getenv("GEMINI_API_KEY") # your Gemini API key

def ask_finance(question: str, company: str = "Apple", fiscal_year: str = "2024") -> str:
    """Query the FinSight RAG backend for financial evidence."""
    r = requests.post(
        f"{BASE}/query",
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        json={"question": question, "company": company, "fiscal_year": fiscal_year},
        timeout=30,
    )
    if r.ok:
        return r.json()["answer"]
    else:
        return f"Error {r.status_code}: {r.text}"

# Define the FinSight RAG tool for LangChain
finsight_tool = Tool(
    name="FinSightRAG",
    func=ask_finance,
    description=(
        "Query financial RAG for evidence. Use when asked about company financials, "
        "revenue, net income, earnings, cash flow, balance sheet, or any financial metric. "
        "Input should be the full question string."
    ),
)

# Real Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GEMINI_KEY,
    temperature=0.1,
)

if __name__ == "__main__":
    if not TOKEN:
        print("No JWT_TOKEN found. Set the JWT_TOKEN environment variable.")
        print("Get one via: POST /auth/login to the backend service.")
        exit(1)
    if not GEMINI_KEY:
        print("No GEMINI_API_KEY found. Set the GEMINI_API_KEY environment variable.")
        exit(1)

    print(f"FinSight-RAG LangChain Agent (Gemini: gemini-2.0-flash)")
    print(f"Backend: {BASE}")
    print("Type 'quit' to exit\n")

    from langchain.agents import AgentType, initialize_agent
    from langchain.memory import ConversationBufferMemory

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = initialize_agent(
        tools=[finsight_tool],
        llm=llm,
        memory=memory,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
    )

    while True:
        try:
            question = input("Ask a question: ").strip()
            if not question:
                continue
            if question.lower() == "quit":
                print("Goodbye!")
                break
            print("\n--- Answer ---")
            answer = agent.run(question)
            print(answer)
            print("--- End ---\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
