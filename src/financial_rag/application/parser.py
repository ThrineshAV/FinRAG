from pathlib import Path
import json
import logging

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ============================================================
# Project paths
# ============================================================

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")


# ============================================================
# Parse SEC HTML
# ============================================================

def parse_html_file(file_path: Path) -> str:
    """
    Parse an SEC HTML filing and return cleaned text.
    """

    try:
        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:
            html_content = file.read()
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Failed to read HTML file %s: %s", file_path.name, exc)
        return ""

    try:
        soup = BeautifulSoup(
            html_content,
            "lxml"
        )

        # Remove non-document elements
        for element in soup([
            "script",
            "style",
            "noscript",
            "svg"
        ]):
            element.decompose()

        # Extract visible text
        text = soup.get_text(
            separator="\n"
        )

        # Normalize whitespace
        lines = []

        for line in text.splitlines():

            line = " ".join(
                line.split()
            )

            if line:
                lines.append(line)

        return "\n".join(lines)

    except Exception as exc:
        logger.warning("Failed to parse HTML file %s: %s", file_path.name, exc)
        return ""


# ============================================================
# Load metadata
# ============================================================

def load_metadata(
    html_file: Path
) -> dict:
    """
    Load metadata associated with an SEC filing.
    """

    metadata_file = (
        html_file.with_suffix(".json")
    )

    if not metadata_file.exists():

        print(
            f"Warning: Metadata file not found "
            f"for {html_file.name}"
        )

        return {}

    with open(
        metadata_file,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# Save processed document
# ============================================================

def save_processed_document(
    source_file: Path,
    text: str,
    metadata: dict
) -> Path:
    """
    Save cleaned text and metadata together.
    """

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        PROCESSED_DATA_DIR /
        f"{source_file.stem}.json"
    )

    document = {
        "metadata": metadata,
        "text": text
    }

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            document,
            file,
            indent=2,
            ensure_ascii=False
        )

    return output_file


# ============================================================
# Process document
# ============================================================

def process_document(
    file_path: Path
) -> Path:

    print(
        f"\nProcessing: {file_path.name}"
    )

    text = parse_html_file(
        file_path
    )

    print(
        f"Extracted characters: "
        f"{len(text):,}"
    )

    metadata = load_metadata(
        file_path
    )

    output_file = save_processed_document(
        file_path,
        text,
        metadata
    )

    print(
        f"Saved: {output_file}"
    )

    return output_file


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    html_files = list(
        RAW_DATA_DIR.glob("*.html")
    )

    if not html_files:

        print(
            "No HTML files found in data/raw/"
        )

        exit()

    for file_path in html_files:

        process_document(
            file_path
        )

    print(
        "\nDocument parsing completed successfully."
    )