from app.vector_store import (
    COLLECTION_NAME,
    create_vector_store,
    index_documents,
    search_documents,
)


def test_create_vector_store():
    client = create_vector_store()

    assert client.collection_exists(COLLECTION_NAME)

    info = client.get_collection(COLLECTION_NAME)
    assert info.config.params.vectors.size == 384


def test_index_documents():
    client = create_vector_store()

    indexed_count = index_documents(client)

    assert indexed_count > 0

    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=[0.0] * 384,
        limit=1,
    )

    # The collection should contain the indexed chunks.
    assert (
        client.count(
            collection_name=COLLECTION_NAME,
            exact=True,
        ).count
        == indexed_count
    )


def test_search_documents_finds_relevant_source():
    client = create_vector_store()
    index_documents(client)

    results = search_documents(
        client,
        "How can I undo a production release?",
        limit=3,
    )

    assert results
    assert results[0]["source"] == "deployment.md"
    assert results[0]["content"]
    assert isinstance(results[0]["score"], float)
