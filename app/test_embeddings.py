from app.embeddings import create_embedding_model, embed_texts


def test_embed_texts_returns_384_dimensional_vectors():
    model = create_embedding_model()
    vectors = embed_texts(
        model,
        [
            "Deployments use the CI/CD pipeline.",
            "The on-call engineer can roll back a deployment.",
        ],
    )

    assert len(vectors) == 2
    assert all(len(vector) == 384 for vector in vectors)
    assert all(all(isinstance(value, float) for value in vector) for vector in vectors)
