import pytest

from production_rag.ingestion import chunk_passages, split_into_chunks

PASSAGES = [
    {
        "source": "fifa-disciplinary-code",
        "page": 21,
        "section": "20. MANIPULATION OF FOOTBALL MATCHES AND COMPETITIONS",
        "text": "a" * 30,
    },
    {
        "source": "fifa-disciplinary-code",
        "page": 22,
        "section": "21. RACISM",
        "text": "b" * 10,
    },
]


def test_split_into_chunks_preserves_all_text():
    text = "abcdefghijklmnopqrstuvwxyz"

    chunks = split_into_chunks(text, chunk_size=10, overlap=2)

    assert len(chunks) == 3
    assert chunks[0] == "abcdefghij"
    assert chunks[1] == "ijklmnopqr"
    assert chunks[2] == "qrstuvwxyz"


def test_split_into_chunks_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        split_into_chunks("hello", chunk_size=0)

    with pytest.raises(ValueError):
        split_into_chunks("hello", chunk_size=10, overlap=10)


def test_chunk_passages_keeps_page_and_section():
    chunks = chunk_passages(PASSAGES, chunk_size=20, overlap=5)

    assert [(chunk["page"], chunk["section"]) for chunk in chunks] == [
        (21, "20. MANIPULATION OF FOOTBALL MATCHES AND COMPETITIONS"),
        (21, "20. MANIPULATION OF FOOTBALL MATCHES AND COMPETITIONS"),
        (22, "21. RACISM"),
    ]


def test_chunk_passages_creates_unique_ids_per_source():
    chunks = chunk_passages(PASSAGES, chunk_size=20, overlap=5)

    assert [chunk["chunk_id"] for chunk in chunks] == [
        "fifa-disciplinary-code:0",
        "fifa-disciplinary-code:1",
        "fifa-disciplinary-code:2",
    ]
