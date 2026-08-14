import re


# ============================================================
# Company dictionary
# ============================================================

COMPANIES = {
    "apple": {
        "ticker": "AAPL",
        "name": "Apple Inc."
    },

    "microsoft": {
        "ticker": "MSFT",
        "name": "Microsoft Corporation"
    },

    "nvidia": {
        "ticker": "NVDA",
        "name": "NVIDIA Corporation"
    },

    "tesla": {
        "ticker": "TSLA",
        "name": "Tesla, Inc."
    },

    "amazon": {
        "ticker": "AMZN",
        "name": "Amazon.com, Inc."
    }
}


# ============================================================
# Financial metric dictionary
# ============================================================

METRICS = {
    "net income": [
        "net income",
        "net earnings",
        "profit"
    ],

    "revenue": [
        "revenue",
        "revenues",
        "net sales",
        "total net sales"
    ],

    "gross profit": [
        "gross profit"
    ],

    "gross margin": [
        "gross margin",
        "gross margin percentage"
    ],

    "operating income": [
        "operating income"
    ],

    "operating expenses": [
        "operating expenses",
        "operating expense"
    ],

    "income before tax": [
        "income before income tax",
        "income before tax",
        "pretax income"
    ],

    "earnings per share": [
        "earnings per share",
        "eps"
    ],

    "cash flow": [
        "cash flow",
        "cash flows"
    ]
}


# ============================================================
# Detect company
# ============================================================

def detect_company(query: str):

    query_lower = query.lower()

    for company, information in COMPANIES.items():

        if company in query_lower:

            return {
                "company": information["name"],
                "ticker": information["ticker"]
            }

    # Also support ticker symbols
    for company, information in COMPANIES.items():

        ticker = information["ticker"].lower()

        if re.search(
            rf"\b{ticker}\b",
            query_lower
        ):

            return {
                "company": information["name"],
                "ticker": information["ticker"]
            }

    return {
        "company": None,
        "ticker": None
    }


# ============================================================
# Detect fiscal year
# ============================================================

def detect_fiscal_year(query: str):

    # Matches:
    # 2025
    # 2026
    # FY2025
    # FY 2026
    # fiscal year 2025

    patterns = [
        r"\bfy\s*(20\d{2})\b",
        r"\bfiscal\s+year\s+(20\d{2})\b",
        r"\b(20\d{2})\b"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            query.lower()
        )

        if match:

            return match.group(1)

    return None


# ============================================================
# Detect filing type
# ============================================================

def detect_filing_type(query: str):

    query_lower = query.lower()

    if "10-k" in query_lower or "10k" in query_lower:

        return "10-K"

    if "annual report" in query_lower:

        return "10-K"

    return "10-K"


# ============================================================
# Detect financial metric
# ============================================================

def detect_metric(query: str):

    query_lower = query.lower()

    # Check longer/more specific phrases first
    # so "net sales" is detected before "sales", etc.

    metric_patterns = []

    for metric, phrases in METRICS.items():

        for phrase in phrases:

            metric_patterns.append(
                (
                    phrase,
                    metric
                )
            )

    # Longest phrase first
    metric_patterns.sort(
        key=lambda x: len(x[0]),
        reverse=True
    )

    for phrase, metric in metric_patterns:

        if re.search(
            rf"\b{re.escape(phrase)}\b",
            query_lower
        ):

            return metric

    return None


# ============================================================
# Parse complete query
# ============================================================

def parse_query(query: str):

    company_info = detect_company(
        query
    )

    fiscal_year = detect_fiscal_year(
        query
    )

    filing_type = detect_filing_type(
        query
    )

    metric = detect_metric(
        query
    )

    return {
        "original_query": query,

        "company":
            company_info["company"],

        "ticker":
            company_info["ticker"],

        "fiscal_year":
            fiscal_year,

        "filing_type":
            filing_type,

        "metric":
            metric
    }


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    test_questions = [

        "What was Apple's revenue in fiscal year 2025?",

        "What was Microsoft's revenue in FY 2026?",

        "How much revenue did NVIDIA generate in 2026?",

        "What was Tesla's net income in 2026?",

        "What was AMZN revenue in fiscal year 2026?",

        "What was Apple's gross profit in 2025?",

        "What was Microsoft's operating income in 2026?",

        "What was NVIDIA's gross margin in 2026?",

        "What was Tesla's earnings per share in 2026?"
    ]

    print("\n")
    print("=" * 70)
    print("FinSight-RAG Query Understanding")
    print("=" * 70)

    for question in test_questions:

        result = parse_query(
            question
        )

        print("\nQuestion:")
        print(question)

        print("\nParsed:")
        print(result)

        print("-" * 70)