"""Edge-case tests for chunker, parser, and query_parser modules."""

from __future__ import annotations

from pathlib import Path

from src.processing.chunker import create_chunks, create_page_chunks
from src.processing.parser import parse_html_file, load_metadata
from src.retrieval.query_parser import (
    detect_companies,
    detect_company,
    detect_fiscal_year,
    detect_filing_type,
    detect_metric,
    parse_query,
)


# ============================================================
# Chunker tests
# ============================================================


def test_create_page_chunks_returns_empty_for_empty_pages() -> None:
    chunks = create_page_chunks([], {"document_id": "empty"})

    assert chunks == []


def test_create_page_chunks_handles_multiple_pages() -> None:
    pages = [
        {"page_number": 1, "text": "First page content."},
        {"page_number": 2, "text": "Second page content."},
    ]
    chunks = create_page_chunks(pages, {"document_id": "multi"})

    assert len(chunks) == 2
    assert chunks[0]["chunk_id"] == "multi-p1-c0"
    assert chunks[1]["chunk_id"] == "multi-p2-c0"
    assert chunks[0]["chunk_index"] == 0
    assert chunks[1]["chunk_index"] == 1


def test_create_chunks_preserves_metadata_fields() -> None:
    metadata = {
        "company": "Apple Inc.",
        "ticker": "AAPL",
        "fiscal_year": "2025",
        "filing_type": "10-K",
    }
    chunks = create_chunks("Apple reported net income.", metadata)

    assert len(chunks) == 1
    assert chunks[0]["metadata"]["company"] == "Apple Inc."
    assert chunks[0]["metadata"]["ticker"] == "AAPL"
    assert chunks[0]["metadata"]["fiscal_year"] == "2025"


def test_create_chunks_returns_empty_for_empty_text() -> None:
    chunks = create_chunks("", {})

    assert chunks == []


# ============================================================
# Parser tests
# ============================================================


def test_parse_html_file_strips_scripts_and_styles(tmp_path: Path) -> None:
    html = (
        "<html><head><style>body{color:red}</style></head>"
        "<body><script>alert('x')</script>"
        "<p>Clean text here.</p></body></html>"
    )
    html_file = tmp_path / "test.html"
    html_file.write_text(html, encoding="utf-8")

    text = parse_html_file(html_file)

    assert "alert" not in text
    assert "color:red" not in text
    assert "Clean text here." in text


def test_parse_html_file_normalizes_whitespace(tmp_path: Path) -> None:
    html = "<html><body><p>  lots   of   spaces  </p></body></html>"
    html_file = tmp_path / "spaces.html"
    html_file.write_text(html, encoding="utf-8")

    text = parse_html_file(html_file)

    assert "lots of spaces" in text


def test_load_metadata_returns_empty_dict_when_file_missing(tmp_path: Path) -> None:
    html_file = tmp_path / "no_metadata.html"
    html_file.write_text("<html></html>", encoding="utf-8")

    metadata = load_metadata(html_file)

    assert metadata == {}


def test_load_metadata_reads_json_sidecar(tmp_path: Path) -> None:
    html_file = tmp_path / "report.html"
    html_file.write_text("<html></html>", encoding="utf-8")
    json_file = tmp_path / "report.json"
    json_file.write_text('{"company": "Tesla"}', encoding="utf-8")

    metadata = load_metadata(html_file)

    assert metadata == {"company": "Tesla"}


# ============================================================
# Query parser tests
# ============================================================


def test_detect_company_by_ticker() -> None:
    result = detect_company("What was AAPL revenue?")

    assert result["ticker"] == "AAPL"
    assert result["company"] == "Apple Inc."


def test_detect_companies_returns_empty_for_unknown() -> None:
    result = detect_companies("What was Google's revenue?")

    assert result == []


def test_detect_company_returns_none_for_unknown() -> None:
    result = detect_company("What was Google's revenue?")

    assert result == {"company": None, "ticker": None}


def test_detect_fiscal_year_fy_prefix() -> None:
    assert detect_fiscal_year("FY2025 results") == "2025"
    assert detect_fiscal_year("FY 2026 revenue") == "2026"


def test_detect_fiscal_year_full_phrase() -> None:
    assert detect_fiscal_year("fiscal year 2025 report") == "2025"


def test_detect_fiscal_year_bare_year() -> None:
    assert detect_fiscal_year("revenue in 2025") == "2025"


def test_detect_fiscal_year_no_match() -> None:
    assert detect_fiscal_year("what is revenue?") is None


def test_detect_filing_type_defaults_to_10k() -> None:
    assert detect_filing_type("show me the report") == "10-K"


def test_detect_filing_type_explicit_10k() -> None:
    assert detect_filing_type("10-K filing for Apple") == "10-K"
    assert detect_filing_type("10K filing for Apple") == "10-K"


def test_detect_metric_revenue() -> None:
    assert detect_metric("What was total revenue?") == "revenue"


def test_detect_metric_net_income() -> None:
    assert detect_metric("What was net income?") == "net income"


def test_detect_metric_no_match() -> None:
    assert detect_metric("Tell me about the company") is None


def test_detect_metric_prefers_longer_phrase() -> None:
    # "net sales" should match "revenue" (via alias), not just partial
    assert detect_metric("What were total net sales?") == "revenue"


def test_parse_query_full_extraction() -> None:
    result = parse_query("What was Apple's net income in FY2025?")

    assert result["company"] == "Apple Inc."
    assert result["ticker"] == "AAPL"
    assert result["fiscal_year"] == "2025"
    assert result["metric"] == "net income"
    assert result["filing_type"] == "10-K"
