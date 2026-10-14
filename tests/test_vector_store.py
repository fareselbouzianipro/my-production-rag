from production_rag.config import settings
from production_rag.embeddings import create_embedding_model
from production_rag.vector_store import (
    create_vector_store,
    search_documents,
    store_documents,
)


def test_create_vector_store():
    client = create_vector_store()

    assert client.collection_exists(settings.collection_name)

    info = client.get_collection(settings.collection_name)
    assert info.config.params.vectors.size == 384


def test_index_documents():
    client = create_vector_store()

    model = create_embedding_model()
    indexed_count = store_documents(client, model)

    assert indexed_count > 0

    result = client.query_points(
        collection_name=settings.collection_name,
        query=[0.0] * 384,
        limit=1,
    )

    # The collection should contain the indexed chunks.
    assert (
        client.count(
            collection_name=settings.collection_name,
            exact=True,
        ).count
        == indexed_count
    )


def test_search_documents_finds_relevant_source():
    client = create_vector_store()
    model = create_embedding_model()
    store_documents(client, model)

    results = search_documents(
        client,
        "How can I undo a production release?",
        model,
        limit=3,
    )

    assert results
    assert results[0]["source"] == "deployment.md"
    assert results[0]["content"]
    assert isinstance(results[0]["score"], float)
