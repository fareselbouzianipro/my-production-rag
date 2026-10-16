from pathlib import Path

import pymupdf

from production_rag.pdf_extraction import clean_text, extract_passages

# A5 pages, like the FIFA and IFAB documents: 420 x 595 points.
WIDTH, HEIGHT = 420, 595
CONTENT = "Anyone who manipulates the course or result of a match shall be sanctioned."


def new_page(pdf: pymupdf.Document) -> pymupdf.Page:
    return pdf.new_page(width=WIDTH, height=HEIGHT)


def write(page: pymupdf.Page, y: float, text: str, bold: bool = False) -> None:
    page.insert_text((40, y), text, fontsize=6, fontname="hebo" if bold else "helv")


def save(pdf: pymupdf.Document, tmp_path: Path) -> Path:
    path = tmp_path / "sample.pdf"
    pdf.save(path)
    return path


def test_clean_text_normalizes_special_characters():
    assert clean_text("football\N{NON-BREAKING HYPHEN}related") == "football-related"
    assert (
        clean_text("1.\t\N{EN SPACE}Scope  of   application")
        == "1. Scope of application"
    )


def test_clean_text_removes_control_characters():
    assert clean_text("\x07Special provisions\x08 12") == "Special provisions 12"


def test_extract_passages_drops_page_furniture(tmp_path):
    pdf = pymupdf.open()
    page = new_page(pdf)
    page.insert_text((40, 20), "21")  # page number in the top margin
    write(page, 300, CONTENT)
    page.insert_text((40, 580), "Regulations 2025")  # footer
    page.insert_text((410, 400), "Offences", rotate=90)  # side tab

    passages = extract_passages(save(pdf, tmp_path))

    assert passages == [
        {"source": "sample", "page": 1, "section": None, "text": CONTENT}
    ]


def test_extract_passages_splits_sections_on_headings(tmp_path):
    pdf = pymupdf.open()
    page = new_page(pdf)
    write(page, 100, "20. MANIPULATION OF FOOTBALL MATCHES", bold=True)
    write(page, 150, CONTENT)
    write(page, 250, "21. RACISM", bold=True)
    write(page, 300, CONTENT)
    page = new_page(pdf)
    write(page, 100, CONTENT)  # no heading: still article 21

    passages = extract_passages(save(pdf, tmp_path))

    assert [(p["page"], p["section"]) for p in passages] == [
        (1, "20. MANIPULATION OF FOOTBALL MATCHES"),
        (1, "21. RACISM"),
        (2, "21. RACISM"),
    ]
    assert passages[0]["text"] == f"20. MANIPULATION OF FOOTBALL MATCHES\n{CONTENT}"


def test_extract_passages_reads_chapter_from_ifab_footer(tmp_path):
    pdf = pymupdf.open()
    page = new_page(pdf)
    write(page, 100, "3. No offence", bold=True)
    write(page, 150, CONTENT)
    page.insert_text((40, 570), "Laws of the Game 2026/27 | Law 11 | Offside")

    passages = extract_passages(save(pdf, tmp_path))

    assert passages[0]["section"] == "Law 11 – Offside > 3. No offence"


def test_extract_passages_skips_contents_and_empty_pages(tmp_path):
    pdf = pymupdf.open()
    write(new_page(pdf), 300, "Diagram")
    toc = new_page(pdf)
    for y, entry in enumerate(["1. Object 8", "2. Scope 8", "3. Applicable law 9"]):
        write(toc, 100 + 50 * y, entry)
    write(new_page(pdf), 300, CONTENT)

    passages = extract_passages(save(pdf, tmp_path))

    assert [p["page"] for p in passages] == [3]
