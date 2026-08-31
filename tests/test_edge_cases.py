"""Edge-case tests for chunker, parser, and query_parser modules."""

from __future__ import annotations

import os
from pathlib import Path

# Disable authentication for edge case tests
os.environ["AUTH_REQUIRED"] = "false"

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


# ============================================================
# SEC ingestion retry tests
# ============================================================


def test_sec_get_company_filings_retries_on_timeout(monkeypatch) -> None:
    """get_company_filings should retry on network timeout."""
    from unittest.mock import Mock
    import requests
    from tenacity import RetryError
    from src.ingestion import sec_ingestion

    attempt_count = 0

    def mock_get(*args, **kwargs):
        nonlocal attempt_count
        attempt_count += 1
        raise requests.exceptions.Timeout("Connection timeout")

    monkeypatch.setattr(requests, "get", mock_get)

    try:
        sec_ingestion.get_company_filings("0000320193")
        assert False, "Should have raised after retries"
    except RetryError:
        # tenacity should have retried 3 times
        assert attempt_count == 3


def test_sec_download_filing_retries_on_connection_error(monkeypatch) -> None:
    """download_filing should retry on connection errors."""
    from unittest.mock import Mock
    import requests
    from tenacity import RetryError
    from src.ingestion import sec_ingestion

    attempt_count = 0

    def mock_get(*args, **kwargs):
        nonlocal attempt_count
        attempt_count += 1
        if "submissions" in args[0]:
            # Return valid filings data for the first call
            mock_response = Mock()
            mock_response.json.return_value = {
                "filings": {
                    "recent": {
                        "form": ["10-K"],
                        "accessionNumber": ["0001234567-25-000001"],
                        "filingDate": ["2025-01-15"],
                        "primaryDocument": ["aapl-10k.htm"],
                        "reportDate": ["2024-12-31"],
                    }
                }
            }
            mock_response.raise_for_status = Mock()
            return mock_response
        else:
            # Fail the filing download
            raise requests.exceptions.ConnectionError("Connection refused")

    monkeypatch.setattr(requests, "get", mock_get)

    try:
        sec_ingestion.download_filing("apple", sec_ingestion.COMPANIES["apple"])
        assert False, "Should have raised after retries"
    except RetryError:
        # download_filing calls get_company_filings (1 success) then tries to download 3 times
        # Each download_filing retry also calls get_company_filings again
        # So: 1 (initial get_company_filings) + 3 (failed downloads) + 2 (retries of get_company_filings) = 6
        assert attempt_count == 6


# ============================================================
# API rate limit tests
# ============================================================


def test_api_rate_limit_enforced_on_query_endpoint() -> None:
    """POST /query should enforce rate limits."""
    from fastapi.testclient import TestClient
    from src import api
    from unittest.mock import MagicMock

    # Mock retrieval to avoid index dependency
    monkeypatch_mock = MagicMock(return_value=([], {}))
    api.retrieve_documents = monkeypatch_mock

    client = TestClient(api.app)

    # Make requests up to the limit (assuming 20/minute default)
    # We'll make 21 requests rapidly
    responses = []
    for i in range(21):
        response = client.post("/query", json={"question": f"test query {i}"})
        responses.append(response.status_code)

    # At least one should be 429 (rate limited)
    assert 429 in responses, "Expected at least one 429 Too Many Requests response"


def test_api_upload_size_limit_enforced() -> None:
    """POST /upload should reject files exceeding MAX_UPLOAD_SIZE."""
    from fastapi.testclient import TestClient
    from src import api
    from io import BytesIO

    client = TestClient(api.app)

    # Create a mock file larger than 25 MB
    large_content = b"0" * (26 * 1024 * 1024)  # 26 MB
    large_file = BytesIO(large_content)

    response = client.post(
        "/upload",
        files={"file": ("large.pdf", large_file, "application/pdf")},
        data={
            "company": "Apple",
            "document_type": "10-K",
            "fiscal_year": "2025",
            "quarter": "Q4",
        },
    )

    assert response.status_code == 413
    assert "exceeds maximum" in response.json()["detail"].lower()


# ============================================================
# HTML parsing edge case tests
# ============================================================


def test_parse_html_file_handles_malformed_html(tmp_path: Path) -> None:
    """parse_html_file should return empty string for malformed HTML."""
    from src.processing.parser import parse_html_file

    malformed_html = "<html><body><p>Unclosed paragraph<div>Unclosed div"
    html_file = tmp_path / "malformed.html"
    html_file.write_text(malformed_html, encoding="utf-8")

    # Should not crash, should return some text or empty
    result = parse_html_file(html_file)
    assert isinstance(result, str)


def test_parse_html_file_handles_empty_document(tmp_path: Path) -> None:
    """parse_html_file should handle empty HTML documents."""
    from src.processing.parser import parse_html_file

    empty_html = "<html><head></head><body></body></html>"
    html_file = tmp_path / "empty.html"
    html_file.write_text(empty_html, encoding="utf-8")

    result = parse_html_file(html_file)
    assert result == ""


def test_parse_html_file_handles_invalid_utf8(tmp_path: Path) -> None:
    """parse_html_file should handle files with invalid UTF-8."""
    from src.processing.parser import parse_html_file

    html_file = tmp_path / "invalid_utf8.html"
    # Write invalid UTF-8 bytes
    html_file.write_bytes(b"<html><body>\xff\xfe Invalid UTF-8</body></html>")

    result = parse_html_file(html_file)
    # Should return empty string or handle gracefully
    assert isinstance(result, str)
