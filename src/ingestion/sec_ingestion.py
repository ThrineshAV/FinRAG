import json
from pathlib import Path
import argparse
import requests


# ============================================================
# Configuration
# ============================================================

SEC_HEADERS = {
    "User-Agent": "FinSight-RAG research avthrinesh@gmail.com"
}

RAW_DATA_DIR = Path("data/raw")


COMPANIES = {
    "apple": {
        "name": "Apple Inc.",
        "ticker": "AAPL",
        "cik": "0000320193",
    },
    "microsoft": {
        "name": "Microsoft Corporation",
        "ticker": "MSFT",
        "cik": "0000789019",
    },
    "nvidia": {
        "name": "NVIDIA Corporation",
        "ticker": "NVDA",
        "cik": "0001045810",
    },
    "tesla": {
        "name": "Tesla, Inc.",
        "ticker": "TSLA",
        "cik": "0001318605",
    },
    "amazon": {
        "name": "Amazon.com, Inc.",
        "ticker": "AMZN",
        "cik": "0001018724",
    },
}


# ============================================================
# Get SEC filing information
# ============================================================

def get_company_filings(cik: str) -> dict:
    """
    Retrieve company filing metadata from SEC EDGAR.
    """

    url = (
        f"https://data.sec.gov/"
        f"submissions/CIK{cik}.json"
    )

    response = requests.get(
        url,
        headers=SEC_HEADERS,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# Find latest 10-K
# ============================================================

def find_latest_10k(
    filings_data: dict
) -> dict:
    """
    Find the latest 10-K filing.
    """

    recent = (
        filings_data[
            "filings"
        ][
            "recent"
        ]
    )

    for index, form in enumerate(
        recent["form"]
    ):

        if form == "10-K":

            return {
                "accession_number":
                    recent["accessionNumber"][index],

                "filing_date":
                    recent["filingDate"][index],

                "primary_document":
                    recent["primaryDocument"][index],

                "report_date":
                    recent["reportDate"][index],
            }

    raise ValueError(
        "No 10-K filing found."
    )


# ============================================================
# Save metadata
# ============================================================

def save_metadata(
    output_path: Path,
    company_info: dict,
    filing: dict,
    filing_url: str
):
    """
    Save structured metadata for the filing.
    """

    metadata = {
        "company": company_info["name"],
        "ticker": company_info["ticker"],
        "cik": company_info["cik"],
        "filing_type": "10-K",
        "filing_date": filing["filing_date"],
        "report_date": filing["report_date"],
        "fiscal_year": filing["filing_date"][:4],
        "accession_number":
            filing["accession_number"],
        "primary_document":
            filing["primary_document"],
        "source_url": filing_url,
    }

    metadata_path = output_path.with_suffix(
        ".json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Metadata saved to: {metadata_path}"
    )


# ============================================================
# Download filing
# ============================================================

def download_filing(
    company_key: str,
    company_info: dict
) -> Path:
    """
    Download the latest 10-K for a company.
    """

    print(
        f"\nFetching "
        f"{company_info['name']}..."
    )

    filings_data = get_company_filings(
        company_info["cik"]
    )

    filing = find_latest_10k(
        filings_data
    )

    accession_number = (
        filing["accession_number"]
    )

    primary_document = (
        filing["primary_document"]
    )

    accession_no_dashes = (
        accession_number.replace(
            "-",
            ""
        )
    )

    filing_url = (
        "https://www.sec.gov/Archives/"
        "edgar/data/"
        f"{int(company_info['cik'])}/"
        f"{accession_no_dashes}/"
        f"{primary_document}"
    )

    print(
        f"Filing date: "
        f"{filing['filing_date']}"
    )

    print(
        f"Report date: "
        f"{filing['report_date']}"
    )

    print(
        f"Document: "
        f"{primary_document}"
    )

    print(
        f"URL: {filing_url}"
    )

    response = requests.get(
        filing_url,
        headers=SEC_HEADERS,
        timeout=30
    )

    response.raise_for_status()

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = (
        f"{company_key}_10k_"
        f"{filing['filing_date']}.html"
    )

    output_path = (
        RAW_DATA_DIR /
        filename
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            response.text
        )

    print(
        f"Filing saved to: "
        f"{output_path}"
    )

    # Save metadata JSON
    save_metadata(
        output_path,
        company_info,
        filing,
        filing_url
    )

    return output_path


# ============================================================
# Main
# ============================================================


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Download SEC 10-K filings"
    )

    parser.add_argument(
        "--company",
        required=True,
        choices=COMPANIES.keys(),
        help="Company to download"
    )

    args = parser.parse_args()

    company_key = args.company

    company_info = COMPANIES[
        company_key
    ]

    download_filing(
        company_key,
        company_info
    )

    print(
        "\nSEC ingestion completed successfully."
    )