from app.vector_store import create_vector_store, index_documents, search_documents


def main() -> None:
    client = create_vector_store()
    indexed_count = index_documents(client)
    print(f"Indexed {indexed_count} chunks.")

    query = "How can I undo a production release?"
    results = search_documents(client, query, limit=3)

    print(f"\nQuery: {query}\n")

    for rank, result in enumerate(results, start=1):
        print(
            f"[{rank}] score={result['score']:.3f}, "
            f"source={result['source']}, "
            f"chunk_id={result['chunk_id']}\n"
        )
        print(result["content"][:300])
        print("-" * 60)


if __name__ == "__main__":
    main()
