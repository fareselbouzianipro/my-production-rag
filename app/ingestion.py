from pathlib import Path

DATA_DIR = Path("data/raw")


def load_documents() -> list[dict[str, str]]:
    """Load Markdown documents from the raw data directory."""
    documents = []

    for file_path in sorted(DATA_DIR.glob("*.md")):
        content = file_path.read_text(encoding="utf-8")

        documents.append({"source": file_path.name, "content": content})

    return documents


def split_into_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split text into overlapping character-based chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        print(chunk)

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start += chunk_size - overlap

    return chunks


def ingest_documents() -> list[dict[str, str]]:
    """Load documents and convert them into source-aware chunks."""
    documents = load_documents()
    chunks = []

    for document in documents:
        document_chunks = split_into_chunks(document["content"])

        for index, chunk_text in enumerate(document_chunks):
            chunks.append(
                {
                    "chunk_id": f"{document['source']}:{index}",
                    "source": document["source"],
                    "content": chunk_text,
                }
            )

    return chunks
