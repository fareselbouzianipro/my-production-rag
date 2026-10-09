from qdrant_client import QdrantClient, models

COLLECTION_NAME = "acme_docs"
VECTOR_SIZE = 384


def create_vector_store() -> QdrantClient:
    """Connect to Qdrant and create the collection if needed."""

    client = QdrantClient(url="http://localhost:6333")

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )

    return client
