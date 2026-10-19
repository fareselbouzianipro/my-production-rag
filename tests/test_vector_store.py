"""Integration tests: they need Qdrant running (docker compose up -d).

They use a separate collection so they never overwrite the real index.
"""

import pytest

from production_rag.config import settings
from production_rag.embeddings import create_embedding_model
from production_rag.vector_store import (
    create_vector_store,
    search_documents,
    store_documents,
)

TEST_COLLECTION = "test_football_regulations"

CHUNKS = [
    {
        "chunk_id": "ifab-laws-of-the-game:0",
        "source": "ifab-laws-of-the-game",
        "page": 111,
        "section": "Law 11 – Offside > 3. No offence",
        "content": (
            "There is no offside offence if a player receives the ball "
            "directly from a goal kick, a throw-in or a corner kick."
        ),
    },
    {
        "chunk_id": "fifa-disciplinary-code:0",
        "source": "fifa-disciplinary-code",
        "page": 21,
        "section": "20. MANIPULATION OF FOOTBALL MATCHES AND COMPETITIONS",
        "content": (
            "Anyone who unlawfully influences or manipulates the course or result "
            "of a match shall be sanctioned with a minimum five-year ban."
        ),
    },
]


@pytest.fixture(scope="module")
def model():
    return create_embedding_model()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "collection_name", TEST_COLLECTION)
    client = create_vector_store()
    yield client
    client.delete_collection(TEST_COLLECTION)


def test_create_vector_store(client):
    info = client.get_collection(TEST_COLLECTION)

    assert info.config.params.vectors.size == settings.vector_size


def test_store_documents_replaces_previous_content(client, model):
    store_documents(client, model, CHUNKS)
    indexed_count = store_documents(client, model, CHUNKS[:1])

    assert indexed_count == 1
    assert client.count(collection_name=TEST_COLLECTION, exact=True).count == 1


def test_search_documents_returns_chunk_metadata(client, model):
    store_documents(client, model, CHUNKS)

    results = search_documents(client, "Is a throw-in offside?", model, limit=1)

    assert results[0]["source"] == "ifab-laws-of-the-game"
    assert results[0]["page"] == 111
    assert results[0]["section"] == "Law 11 – Offside > 3. No offence"
    assert isinstance(results[0]["score"], float)
