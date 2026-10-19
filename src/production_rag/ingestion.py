from pathlib import Path

from production_rag.config import settings
from production_rag.pdf_extraction import extract_passages


def load_passages(corpus_dir: Path | None = None) -> list[dict]:
    """Extract section-labelled passages from every PDF of the corpus."""
    corpus_dir = corpus_dir or settings.corpus_dir
    passages = []

    for path in sorted(corpus_dir.glob("*.pdf")):
        passages.extend(extract_passages(path))

    return passages


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

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start += chunk_size - overlap

    return chunks


def chunk_passages(
    passages: list[dict],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """Split passages into chunks that keep their source, page and section."""
    chunks = []
    chunk_counts: dict[str, int] = {}

    for passage in passages:
        source = passage["source"]

        for text in split_into_chunks(passage["text"], chunk_size, overlap):
            index = chunk_counts.get(source, 0)
            chunk_counts[source] = index + 1

            chunks.append(
                {
                    "chunk_id": f"{source}:{index}",
                    "source": source,
                    "page": passage["page"],
                    "section": passage["section"],
                    "content": text,
                }
            )

    return chunks


def ingest_documents(corpus_dir: Path | None = None) -> list[dict]:
    """Load the corpus and convert it into source-aware chunks."""
    return chunk_passages(
        load_passages(corpus_dir),
        settings.chunk_size,
        settings.chunk_overlap,
    )
