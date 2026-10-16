"""Download the corpus PDFs listed in data/sources.toml.

Usage: uv run python scripts/download_corpus.py
"""

import hashlib
import sys
import tomllib
from pathlib import Path

import httpx

MANIFEST_PATH = Path("data/sources.toml")
CORPUS_DIR = Path("data/corpus")


def load_manifest(path: Path = MANIFEST_PATH) -> list[dict]:
    """Read the list of documents to download."""
    with path.open("rb") as file:
        return tomllib.load(file)["documents"]


def sha256_of(path: Path) -> str:
    """Compute the SHA-256 fingerprint of a file's content."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url: str, destination: Path) -> None:
    """Download a PDF, writing to a temporary file first so a failed
    download never leaves a half-written file behind."""
    temporary = destination.with_suffix(".part")

    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
        response.raise_for_status()
        with temporary.open("wb") as file:
            for block in response.iter_bytes():
                file.write(block)

    if not temporary.read_bytes().startswith(b"%PDF"):
        temporary.unlink()
        raise ValueError(f"{url} did not return a PDF")

    temporary.rename(destination)


def sync_document(document: dict) -> bool:
    """Make sure one document is present and matches its pinned checksum.

    Returns True if the document is valid, False otherwise.
    """
    destination = CORPUS_DIR / f"{document['id']}.pdf"
    # .get returns None is key is missing
    expected = document.get("sha256")

    if destination.exists() and expected and sha256_of(destination) == expected:
        print(f"[ok]       {document['id']} (already downloaded)")
        return True

    download(document["url"], destination)
    actual = sha256_of(destination)

    if expected is None:
        print(f"[new]      {document['id']} -> pin it in the manifest:")
        print(f'           sha256 = "{actual}"')
        return True

    if actual != expected:
        print(f"[mismatch] {document['id']}: the file at the URL has changed.")
        print(f"           expected {expected}")
        print(f"           got      {actual}")
        return False

    print(f"[ok]       {document['id']} (downloaded)")
    return True


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    results = [sync_document(document) for document in load_manifest()]

    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
