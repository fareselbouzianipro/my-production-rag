import pytest

from app.ingestion import ingest_documents, load_documents, split_into_chunks


def test_load_documents():
    documents = load_documents()

    assert len(documents) == 3
    sources = {document["source"] for document in documents}
    assert sources == {
        "access-policy.md",
        "deployment.md",
        "logging-policy.md",
    }

    empty = [d["source"] for d in documents if not d["content"].strip()]
    assert not empty, f"Empty documents: {empty}"


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


def test_ingest_documents_returns_source_aware_chunks():
    chunks = ingest_documents()

    assert chunks
    assert all(chunk["chunk_id"] for chunk in chunks)
    assert all(chunk["source"] for chunk in chunks)
    assert all(chunk["content"].strip() for chunk in chunks)

    sources = {chunk["source"] for chunk in chunks}

    assert sources == {
        "access-policy.md",
        "deployment.md",
        "logging-policy.md",
    }

    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)
