import re
from pathlib import Path

import pymupdf

# Margins as a fraction of the page size. Text outside them is page furniture
# (page numbers, running headers/footers, side tabs), not document content.
TOP_MARGIN = 0.07
BOTTOM_MARGIN = 0.93
RIGHT_EDGE = 0.92

# Pages with less text than this after cleaning are diagrams or title pages.
MIN_PAGE_CHARS = 50

HORIZONTAL = (1.0, 0.0)

# Characters the PDFs use that would add noise to chunks and embeddings.
CHARACTER_FIXES = {
    "\x07": "•",  # bullet glyph used in the IFAB Laws of the Game
    " ": " ",  # en space
    "‑": "-",  # non-breaking hyphen
    "\t": " ",
}


def clean_text(text: str) -> str:
    """Normalize special characters and collapse repeated spaces."""
    for character, replacement in CHARACTER_FIXES.items():
        text = text.replace(character, replacement)

    text = re.sub(r"•\s*•", "•", text)  # nested lists produce doubled bullets
    text = re.sub(r"•(?=\S)", "• ", text)

    return re.sub(r" {2,}", " ", text).strip()


def is_page_content(line: dict, width: float, height: float) -> bool:
    """Tell whether a line is document content rather than page furniture."""
    x0, y0, _, y1 = line["bbox"]

    if tuple(line["dir"]) != HORIZONTAL:
        return False  # rotated side tabs
    if y1 < height * TOP_MARGIN or y0 > height * BOTTOM_MARGIN:
        return False  # page numbers, running headers and footers

    return x0 <= width * RIGHT_EDGE  # chapter labels sit on the right edge


def extract_page_text(page: pymupdf.Page) -> str:
    """Extract the cleaned text of one page, one paragraph per line."""
    width, height = page.rect.width, page.rect.height
    paragraphs = []

    for block in page.get_text("dict")["blocks"]:
        lines = [
            clean_text("".join(span["text"] for span in line["spans"]))
            for line in block.get("lines", [])
            if is_page_content(line, width, height)
        ]
        paragraph = ""

        for line in filter(None, lines):
            # Lines inside a block are usually one paragraph wrapped by the PDF
            # layout, so they are joined with spaces. A bullet starts a new line.
            if line.startswith("•") or not paragraph:
                paragraph += ("\n" if paragraph else "") + line
            else:
                paragraph += " " + line

        if paragraph:
            paragraphs.append(paragraph)

    return "\n".join(paragraphs)


def extract_pages(path: Path) -> list[dict]:
    """Extract the cleaned text of every content page of a PDF."""
    pages = []

    with pymupdf.open(path) as document:
        for page in document:
            text = extract_page_text(page)

            if len(text) >= MIN_PAGE_CHARS:
                pages.append(
                    {
                        "source": path.stem,
                        "page": page.number + 1,
                        "text": text,
                    }
                )

    return pages
