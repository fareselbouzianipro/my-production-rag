from production_rag.embeddings import create_embedding_model
from production_rag.generation import generate_answer
from production_rag.vector_store import create_vector_store, search_documents


def answer_question(question: str, limit: int = 3) -> dict:
    """Retrieve relevant chunks and generate a grounded answer."""

    if not question.strip():
        raise ValueError("Question must not be empty")

    client = create_vector_store()
    model = create_embedding_model()
    chunks = search_documents(
        client,
        question,
        model,
        limit,
    )

    if not chunks:
        return {
            "answer": "I don't know based on the available documents.",
            "sources": [],
            "retrieved_chunks": [],
        }
    answer = generate_answer(question, chunks)

    sources = list(dict.fromkeys(chunk["source"] for chunk in chunks))

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": chunks,
    }
