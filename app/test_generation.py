from app.generation import generate_answer


def test_generate_answer_returns_response():
    chunks = [
        {
            "source": "deployment.md",
            "content": (
                "The on-call engineer can roll back to the "
                "previous stable version using the deployment dashboard."
            ),
        }
    ]

    answer = generate_answer(
        "How can an engineer roll back a deployment?",
        chunks,
    )

    assert answer.strip()
    assert "deployment.md" in answer.lower()
