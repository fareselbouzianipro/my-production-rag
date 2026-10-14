from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from production_rag.config import settings
from production_rag.embeddings import embed_texts
from production_rag.ingestion import ingest_documents


def create_vector_store() -> QdrantClient:
    """Connect to Qdrant and create the collection if needed."""

    client = QdrantClient(url=settings.qdrant_url)

    if not client.collection_exists(settings.collection_name):
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=models.VectorParams(
                size=settings.vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    return client


def store_documents(client: QdrantClient, model: TextEmbedding) -> int:
    """Embed document chunks and store them in Qdrant."""
    chunks = ingest_documents()

    if not chunks:
        return 0

    vectors = embed_texts(model, [chunk["content"] for chunk in chunks])

    points = [
        models.PointStruct(
            id=index,
            vector=vector,
            payload={
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "content": chunk["content"],
            },
        )
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

    return [
        {
            "score": point.score,
            "source": point.payload["source"],
            "chunk_id": point.payload["chunk_id"],
            "content": point.payload["content"],
        }
        for point in response.points
    ]
