import pymupdf

from production_rag.pdf_extraction import clean_text, extract_pages

CONTENT = "Article 20 forbids the manipulation of football matches and competitions."


def test_clean_text_normalizes_special_characters():
    assert clean_text("football‑related") == "football-related"
    assert clean_text("1.\t Scope  of   application") == "1. Scope of application"


def test_clean_text_merges_doubled_bullets():
    assert clean_text("• \x07must leave the field") == "• must leave the field"


def test_extract_pages_drops_page_furniture(tmp_path):
    # A5 page, like the FIFA documents: 420 x 595 points.
    pdf = pymupdf.open()
    page = pdf.new_page(width=420, height=595)
    page.insert_text((40, 20), "21")  # page number in the top margin
    page.insert_text((40, 300), CONTENT, fontsize=6)
    page.insert_text((40, 580), "Laws of the Game | Law 11")  # footer
    page.insert_text((410, 400), "Offences", rotate=90)  # side tab
    path = tmp_path / "sample.pdf"
    pdf.save(path)

    pages = extract_pages(path)

    assert pages == [{"source": "sample", "page": 1, "text": CONTENT}]


def test_extract_pages_skips_pages_without_content(tmp_path):
    pdf = pymupdf.open()
    pdf.new_page(width=420, height=595).insert_text((40, 300), "Diagram")
    pdf.new_page(width=420, height=595).insert_text((40, 300), CONTENT, fontsize=6)
    path = tmp_path / "sample.pdf"
    pdf.save(path)

    pages = extract_pages(path)

    assert [page["page"] for page in pages] == [2]
