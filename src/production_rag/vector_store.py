from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from production_rag.config import settings
from production_rag.embeddings import embed_texts


def create_collection(client: QdrantClient) -> None:
    client.create_collection(
        collection_name=settings.collection_name,
        vectors_config=models.VectorParams(
            size=settings.vector_size,
            distance=models.Distance.COSINE,
        ),
    )


def create_vector_store() -> QdrantClient:
    """Connect to Qdrant and create the collection if needed."""

    client = QdrantClient(url=settings.qdrant_url)

    if not client.collection_exists(settings.collection_name):
        create_collection(client)

    return client


def store_documents(
    client: QdrantClient, model: TextEmbedding, chunks: list[dict]
) -> int:
    """Embed chunks and replace the collection content with them.

    The collection is recreated so chunks from a previous indexing run
    (e.g. with another chunk size) never mix with the new ones.
    """
    client.delete_collection(settings.collection_name)
    create_collection(client)

    if not chunks:
        return 0

    vectors = embed_texts(model, [chunk["content"] for chunk in chunks])

    points = [
        models.PointStruct(id=index, vector=vector, payload=chunk)
        for index, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    client.upsert(collection_name=settings.collection_name, points=points, wait=True)
    return len(points)


def search_documents(
    client: QdrantClient, query: str, model: TextEmbedding, limit: int = 3
) -> list[dict]:
    """Return the most semantically relevant document chunks."""
    if not query.strip():
        raise ValueError("Query must not be empty")
    if limit <= 0:
        raise ValueError("Limit must be positive")

    query_vector = embed_texts(model, [query])[0]

    response = client.query_points(
        collection_name=settings.collection_name,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )

    return [{"score": point.score, **point.payload} for point in response.points]
