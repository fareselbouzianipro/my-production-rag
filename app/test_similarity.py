from app.embeddings import (
    cosine_similarity,
    create_embedding_model,
    embed_texts,
)


def test_semantically_related_texts_are_more_similar():
    model = create_embedding_model()

    texts = [
        "How can I roll back a production deployment?",
        "How can I revert a release to the previous version?",
        "What is the company's policy for working remotely?",
    ]

    vectors = embed_texts(model, texts)

    related_similarity = cosine_similarity(vectors[0], vectors[1])
    unrelated_similarity = cosine_similarity(vectors[0], vectors[2])

    print(f"Related similarity:   {related_similarity:.3f}")
    print(f"Unrelated similarity: {unrelated_similarity:.3f}")

    assert related_similarity > unrelated_similarity
