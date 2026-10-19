import pytest

from production_rag import generation
from production_rag.generation import generate_answer

CHUNKS = [
    {
        "source": "ifab-laws-of-the-game",
        "page": 111,
        "section": "Law 11 – Offside > 3. No offence",
        "content": (
            "There is no offside offence if a player receives the ball "
            "directly from a goal kick, a throw-in or a corner kick."
        ),
    }
]
QUESTION = "Can a player be offside from a throw-in?"


class FakeResponse:
    """Stands in for the Ollama HTTP response, so tests don't call a real LLM."""

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"response": "  No [ifab-laws-of-the-game, p. 111]  "}


def test_generate_answer_sends_context_and_question(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured.update(json)
        return FakeResponse()

    monkeypatch.setattr(generation.httpx, "post", fake_post)

    answer = generate_answer(QUESTION, CHUNKS)

    assert answer == "No [ifab-laws-of-the-game, p. 111]"
    assert "[ifab-laws-of-the-game, p. 111]" in captured["prompt"]
    assert "Law 11 – Offside > 3. No offence" in captured["prompt"]
    assert QUESTION in captured["prompt"]


def test_generate_answer_without_context_does_not_call_llm(monkeypatch):
    def fail_post(*args, **kwargs):
        pytest.fail("LLM should not be called without context")

    monkeypatch.setattr(generation.httpx, "post", fail_post)

    answer = generate_answer(QUESTION, [])

    assert "don't know" in answer


def test_generate_answer_rejects_empty_question():
    with pytest.raises(ValueError):
        generate_answer("   ", CHUNKS)
