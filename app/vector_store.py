from qdrant_client import QdrantClient, models

from app.embeddings import create_embedding_model, embed_texts
from app.ingestion import ingest_documents

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


def index_documents(client: QdrantClient) -> int:
    """Embed document chunks and store them in Qdrant."""
    chunks = ingest_documents()

    if not chunks:
        return 0

    model = create_embedding_model()
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

    client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
    return len(points)
