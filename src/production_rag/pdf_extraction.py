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
    "\t": " ",
    "\N{EN SPACE}": " ",
    "\N{NON-BREAKING HYPHEN}": "-",
}

# Invisible control characters (\x07, \x08...) found in the PDFs' text.
CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b-\x1f]")

# A section heading starts with an article number: "18.", "5bis.", "3. No offence".
HEADING_NUMBER = re.compile(r"^\d+(bis|ter|quater)?\.(\s|$)")

# A table of contents entry ends with its page number: "02. STATUS OF PLAYERS 15".
TOC_ENTRY = re.compile(r"\s\d+$")

# IFAB running footer: "Laws of the Game 2026/27 | Law 11 | Offside".
FOOTER_CHAPTER = re.compile(r"Laws of the Game [\d/]+\s*\|\s*(.+)$")


def clean_text(text: str) -> str:
    """Normalize special characters and collapse repeated spaces."""
    for character, replacement in CHARACTER_FIXES.items():
        text = text.replace(character, replacement)

    text = CONTROL_CHARACTERS.sub("", text)
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


def is_bold(span: dict) -> bool:
    return "bold" in span["font"].lower()


def read_line(line: dict) -> tuple[str, bool]:
    """Return the cleaned text of a line and whether it is entirely bold."""
    text = clean_text("".join(span["text"] for span in line["spans"]))
    bold = all(is_bold(span) for span in line["spans"] if span["text"].strip())

    return text, bold


def join_lines(lines: list[str]) -> str:
    """Join lines wrapped by the PDF layout into one paragraph.

    A bullet starts a new line, so lists stay readable.
    """
    paragraph = ""

    for line in lines:
        if line.startswith("•") or not paragraph:
            paragraph += ("\n" if paragraph else "") + line
        else:
            paragraph += " " + line

    return paragraph


def read_page(page: pymupdf.Page) -> list[tuple[str, str]]:
    """Read a page as ("heading", text) and ("text", text) items, in order."""
    width, height = page.rect.width, page.rect.height
    items = []

    for block in page.get_text("dict")["blocks"]:
        lines = [
            read_line(line)
            for line in block.get("lines", [])
            if is_page_content(line, width, height)
        ]
        lines = [(text, bold) for text, bold in lines if text]

        # A heading is a run of bold lines at the start of a block, beginning
        # with an article number. The body of the section may follow in the
        # same block.
        heading = []
        while lines and lines[0][1] and (heading or HEADING_NUMBER.match(lines[0][0])):
            heading.append(lines.pop(0)[0])

        if heading:
            items.append(("heading", " ".join(heading)))
        if lines:
            items.append(("text", join_lines([text for text, _ in lines])))

    return items


def is_table_of_contents(items: list[tuple[str, str]]) -> bool:
    """Detect contents pages: titled "Contents", or mostly lines ending with
    a page number."""
    lines = [line for _, text in items for line in text.split("\n")]

    if lines[0].lower() in ("contents", "table of contents"):
        return True

    toc_lines = sum(1 for line in lines if TOC_ENTRY.search(line))
    return toc_lines / len(lines) >= 0.5


def read_footer_chapter(page: pymupdf.Page) -> str | None:
    """Read the chapter name from the running footer, if the page has one."""
    footer = pymupdf.Rect(
        0, page.rect.height * BOTTOM_MARGIN, page.rect.width, page.rect.height
    )
    match = FOOTER_CHAPTER.search(page.get_text("text", clip=footer).strip())

    if not match:
        return None

    return " – ".join(part.strip() for part in match[1].split("|"))


def read_toc_chapters(document: pymupdf.Document) -> list[tuple[int, str]]:
    """Read the top-level entries of the PDF's built-in table of contents,
    as (start page, title) pairs."""
    return [
        (page, clean_text(title))
        for level, title, page in document.get_toc()
        if level == 1
    ]


def chapter_at(toc_chapters: list[tuple[int, str]], page_number: int) -> str | None:
    """Find the chapter a page belongs to: the last one starting before it."""
    chapter = None

    for start, title in toc_chapters:
        if start > page_number:
            break
        chapter = title

    return chapter


def extract_passages(path: Path) -> list[dict]:
    """Extract the cleaned text of a PDF as passages labelled with their section.

    A passage is the text of one section on one page. The current section is
    carried over from page to page until a new heading appears. Chapters come
    from the running footer (IFAB) or the PDF's table of contents (FIFA).
    """
    passages = []
    chapter = None
    heading = None

    def add_passage(page_number: int, paragraphs: list[str]) -> None:
        if paragraphs:
            section = " > ".join(part for part in (chapter, heading) if part)
            passages.append(
                {
                    "source": path.stem,
                    "page": page_number,
                    "section": section or None,
                    "text": "\n".join(paragraphs),
                }
            )

    with pymupdf.open(path) as document:
        toc_chapters = read_toc_chapters(document)

        for page in document:
            page_number = page.number + 1
            items = read_page(page)

            page_chapter = read_footer_chapter(page) or chapter_at(
                toc_chapters, page_number
            )

            if sum(len(text) for _, text in items) < MIN_PAGE_CHARS:
                continue
            if is_table_of_contents(items) or page_chapter == "Contents":
                continue

            if page_chapter and page_chapter != chapter:
                chapter, heading = page_chapter, None

            paragraphs = []
            for kind, text in items:
                if kind == "heading":
                    add_passage(page_number, paragraphs)
                    heading = text
                    paragraphs = []
                paragraphs.append(text)

            add_passage(page_number, paragraphs)

    return passages
