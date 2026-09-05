"""Slack bot integration for FinSight-RAG.
Calls /query backend with JWT Bearer auth, returns answer with citations.
Run: $env:SLACK_BOT_TOKEN="xoxb-..."; $env:JWT_TOKEN="..."; python slack_bot.py
"""
import os
import re
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

API_URL = os.getenv("API_URL", "http://localhost:8000")
JWT_TOKEN = os.getenv("JWT_TOKEN")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")  # xoxb-...
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")


def _call_rag(question: str, company: str = "Apple", fiscal_year: str = "2024") -> dict:
    """Call the /query endpoint with JWT Bearer auth."""
    r = requests.post(
        f"{API_URL}/query",
        headers={
            "Authorization": f"Bearer {JWT_TOKEN}",
            "Content-Type": "application/json",
        },
        json={
            "question": question,
            "company": company,
            "fiscal_year": fiscal_year,
        },
        timeout=30,
    )
    if not r.ok:
        return {"error": f"Backend error {r.status_code}: {r.text}"}
    return r.json()


def _parse_question(text: str) -> tuple[str, str, str]:
    """Extract company, fiscal_year, and question from slack text.
    Example: 'What was Apple revenue in 2024?' -> (Apple, 2024, What was Apple revenue in 2024?)
    """
    # Extract year
    year_match = re.search(r"\b(20\d{2})\b", text)
    fiscal_year = year_match.group(1) if year_match else "2024"

    # Extract company (Apple, Microsoft, NVIDIA, Tesla, Amazon)
    companies = ["Apple", "Microsoft", "NVIDIA", "Tesla", "Amazon"]
    company = "Apple"
    for c in companies:
        if c.lower() in text.lower():
            company = c
            break

    return company, fiscal_year, text


def _format_response(rag_result: dict) -> str:
    """Format the RAG result for Slack."""
    if "error" in rag_result:
        return f":warning: {rag_result['error']}"

    answer = rag_result.get("answer", "No answer returned.")
    chunks = rag_result.get("retrieved_chunks", [])

    response = f"*Answer:*\n{answer}\n\n"
    if chunks:
        response += f"*Citations:* {len(chunks)} evidence chunks retrieved\n"
        for i, chunk in enumerate(chunks[:3], 1):
            preview = chunk[:200].replace("\n", " ").strip()
            response += f"  {i}. {preview}...\n"
    return response


@app.route("/slack/events", methods=["POST"])
def slack_events():
    """Slack Events API endpoint."""
    data = request.json

    # Slack URL verification
    if data.get("type") == "url_verification":
        return jsonify({"challenge": data.get("challenge")})

    # Handle message events
    event = data.get("event", {})
    if event.get("type") == "message" and "text" in event:
        text = event["text"]
        company, fiscal_year, question = _parse_question(text)
        rag_result = _call_rag(question, company, fiscal_year)
        reply = _format_response(rag_result)

        # Post reply back to Slack
        channel = event.get("channel")
        if channel and SLACK_BOT_TOKEN:
            requests.post(
                "https://slack.com/api/chat.postMessage",
                headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
                json={"channel": channel, "text": reply},
            )

        return jsonify({"status": "ok"})

    return jsonify({"status": "ignored"})


if __name__ == "__main__":
    if not JWT_TOKEN:
        print("ERROR: JWT_TOKEN not set. Get one via POST /auth/login")
        exit(1)
    print(f"FinSight-RAG Slack Bot starting on http://localhost:3000")
    print(f"Backend: {API_URL}")
    app.run(host="0.0.0.0", port=3000, debug=False)
