import pytest

from production_rag import generation
from production_rag.generation import generate_answer

CHUNKS = [
    {
        "source": "deployment.md",
        "content": (
            "The on-call engineer can roll back to the "
            "previous stable version using the deployment dashboard."
        ),
    }
]


class FakeResponse:
    """Stands in for the Ollama HTTP response, so tests don't call a real LLM."""

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"response": "  Use the deployment dashboard [deployment.md]  "}


def test_generate_answer_sends_context_and_question(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured.update(json)
        return FakeResponse()

    monkeypatch.setattr(generation.httpx, "post", fake_post)

    answer = generate_answer("How can an engineer roll back a deployment?", CHUNKS)

    assert answer == "Use the deployment dashboard [deployment.md]"
    assert "Source: deployment.md" in captured["prompt"]
    assert "How can an engineer roll back a deployment?" in captured["prompt"]


def test_generate_answer_without_context_does_not_call_llm(monkeypatch):
    def fail_post(*args, **kwargs):
        pytest.fail("LLM should not be called without context")

    monkeypatch.setattr(generation.httpx, "post", fail_post)

    answer = generate_answer("How can an engineer roll back a deployment?", [])

    assert "don't know" in answer


def test_generate_answer_rejects_empty_question():
    with pytest.raises(ValueError):
        generate_answer("   ", CHUNKS)
