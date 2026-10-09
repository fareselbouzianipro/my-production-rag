from app.vector_store import COLLECTION_NAME, create_vector_store


def test_create_vector_store():
    client = create_vector_store()

    assert client.collection_exists(COLLECTION_NAME)

    info = client.get_collection(COLLECTION_NAME)
    assert info.config.params.vectors.size == 384
