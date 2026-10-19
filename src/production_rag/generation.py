import httpx

from production_rag.config import settings


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    """Generate an answer based on retrieved document chunks"""

    if not question.strip():
        raise ValueError("Question must not be empty")
    if not context_chunks:
        return "I don't know based on the available documents."

    context = "\n\n".join(
        f"Source: [{chunk['source']}, p. {chunk['page']}]\n"
        f"Section: {chunk['section'] or 'unknown'}\n"
        f"{chunk['content']}"
        for chunk in context_chunks
    )

    prompt = f"""You are an assistant answering questions about football regulations
(IFAB Laws of the Game, FIFA regulations).
Answer the question using only the context below.
If the context does not contain enough information, say you don't know.
Do not invent facts.
Cite your sources as they appear in the context, such as
[ifab-laws-of-the-game, p. 111].

Context:
{context}

Question: {question}

Answer:"""

    response = httpx.post(
        f"{settings.ollama_url}/api/generate",
        json={
            "model": settings.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "seed": 42},
        },
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()["response"].strip()
