from production_rag.embeddings import create_embedding_model
from production_rag.ingestion import ingest_documents
from production_rag.vector_store import (
    create_vector_store,
    search_documents,
    store_documents,
)


def main() -> None:
    client = create_vector_store()
    model = create_embedding_model()

    chunks = ingest_documents()
    indexed_count = store_documents(client, model, chunks)
    print(f"Indexed {indexed_count} chunks.")

    query = "Can a player be offside when receiving the ball from a throw-in?"
    results = search_documents(client, query, model, limit=3)

    print(f"\nQuery: {query}\n")

    for rank, result in enumerate(results, start=1):
        print(
            f"[{rank}] score={result['score']:.3f}, "
            f"source={result['source']}, page={result['page']}\n"
            f"    section={result['section']}\n"
        )
        print(result["content"][:300])
        print("-" * 60)


if __name__ == "__main__":
    main()
