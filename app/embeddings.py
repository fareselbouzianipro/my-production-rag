from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def create_embedding_model() -> TextEmbedding:
    """Initialize the local embedding model."""
    return TextEmbedding(model_name=MODEL_NAME)


def embed_texts(
    model: TextEmbedding,
    texts: list[str],
) -> list[list[float]]:
    """Convert texts into numerical vectors."""
    return [embedding.tolist() for embedding in model.embed(texts)]


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = sum(a * a for a in vector_a) ** 0.5
    norm_b = sum(b * b for b in vector_b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cannot compare zero vectors")

    return dot_product / (norm_a * norm_b)
