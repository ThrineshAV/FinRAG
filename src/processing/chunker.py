from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROCESSED_DATA_DIR = Path("data/processed")
CHUNKS_DATA_DIR = Path("data/chunks")


# ---------------------------------------------------------
# Chunking configuration
# ---------------------------------------------------------

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ---------------------------------------------------------
# Create text splitter
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Chunk a document
# ---------------------------------------------------------

def chunk_document(file_path: Path):
    """
    Read a processed document and split it into chunks.
    """

    print(f"\nChunking: {file_path.name}")

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        text = file.read()

    chunks = text_splitter.split_text(text)

    print(f"Characters: {len(text):,}")
    print(f"Chunks created: {len(chunks):,}")

    return chunks


# ---------------------------------------------------------
# Save chunks
# ---------------------------------------------------------

def save_chunks(
    source_file: Path,
    chunks: list[str]
):
    """
    Save chunks as a text file for inspection.
    """

    CHUNKS_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        CHUNKS_DATA_DIR /
        f"{source_file.stem}_chunks.txt"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        for index, chunk in enumerate(chunks):

            file.write(
                f"\n{'=' * 80}\n"
            )

            file.write(
                f"CHUNK {index}\n"
            )

            file.write(
                f"{'=' * 80}\n\n"
            )

            file.write(chunk)
            file.write("\n")

    print(f"Saved chunks to: {output_file}")


# ---------------------------------------------------------
# Process all documents
# ---------------------------------------------------------

def process_all_documents():

    files = list(
        PROCESSED_DATA_DIR.glob("*.txt")
    )

    if not files:
        print(
            "No processed documents found "
            "in data/processed/"
        )
        return

    for file_path in files:

        chunks = chunk_document(
            file_path
        )

        save_chunks(
            file_path,
            chunks
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    process_all_documents()

    print(
        "\nDocument chunking completed successfully."
    )