from pathlib import Path
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# Project paths
# ============================================================

PROCESSED_DATA_DIR = Path("data/processed")
CHUNKS_DATA_DIR = Path("data/chunks")


# ============================================================
# Chunking configuration
# ============================================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ============================================================
# Text splitter
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)


# ============================================================
# Load processed document
# ============================================================

def load_document(file_path: Path) -> dict:
    """
    Load processed document containing text and metadata.
    """

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# Create chunks
# ============================================================

def create_chunks(
    text: str,
    metadata: dict
) -> list[dict]:
    """
    Split document text into chunks and attach
    financial metadata to every chunk.
    """

    text_chunks = text_splitter.split_text(
        text
    )

    chunks = []

    for index, chunk_text in enumerate(
        text_chunks
    ):

        chunk = {
            "chunk_index": index,

            "text": chunk_text,

            "metadata": {
                "company": metadata.get(
                    "company"
                ),

                "ticker": metadata.get(
                    "ticker"
                ),

                "cik": metadata.get(
                    "cik"
                ),

                "filing_type": metadata.get(
                    "filing_type"
                ),

                "filing_date": metadata.get(
                    "filing_date"
                ),

                "report_date": metadata.get(
                    "report_date"
                ),

                "fiscal_year": metadata.get(
                    "fiscal_year"
                ),

                "accession_number": metadata.get(
                    "accession_number"
                ),

                "primary_document": metadata.get(
                    "primary_document"
                ),

                "source_url": metadata.get(
                    "source_url"
                )
            }
        }

        chunks.append(
            chunk
        )

    return chunks


# ============================================================
# Save chunks
# ============================================================

def save_chunks(
    source_file: Path,
    chunks: list[dict]
) -> Path:
    """
    Save structured chunks as JSON.
    """

    CHUNKS_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        CHUNKS_DATA_DIR /
        f"{source_file.stem}_chunks.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            indent=2,
            ensure_ascii=False
        )

    return output_file


# ============================================================
# Process one document
# ============================================================

def process_document(
    file_path: Path
) -> Path:

    print(
        f"\nProcessing: {file_path.name}"
    )

    document = load_document(
        file_path
    )

    text = document.get(
        "text",
        ""
    )

    metadata = document.get(
        "metadata",
        {}
    )

    print(
        f"Characters: {len(text):,}"
    )

    chunks = create_chunks(
        text,
        metadata
    )

    print(
        f"Chunks created: {len(chunks)}"
    )

    output_file = save_chunks(
        file_path,
        chunks
    )

    print(
        f"Saved: {output_file}"
    )

    return output_file


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    json_files = list(
        PROCESSED_DATA_DIR.glob("*.json")
    )

    if not json_files:

        print(
            "No processed JSON documents found "
            "in data/processed/"
        )

        exit()

    for file_path in json_files:

        process_document(
            file_path
        )

    print(
        "\nMetadata-aware chunking completed successfully."
    )