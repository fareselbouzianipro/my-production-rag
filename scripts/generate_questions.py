import json
import random
from pathlib import Path

import anthropic
from pydantic import BaseModel

from production_rag.ingestion import load_passages

# A stronger model than the one under evaluation, so the eval set is reliable
# and not biased toward the system's own model.
GENERATOR_MODEL = "claude-opus-5-5"

# Start small to check the output file, then raise to 80.
N_QUESTIONS = 5
OUTPUT_PATH = Path("data/eval/generated_questions.json")

QUESTION_PROMPT = """You are helping build an evaluation set for a question-answering \
system about football regulations.

Read the passage below and write ONE question that:
- can be answered using only this passage;
- is self-contained: never refer to "the passage", "this text" or "this article";
- sounds like a real question from a player, referee, club official or fan.

Then write the answer, based only on the passage, in one or two sentences.

Passage (from {source}, section "{section}"):
{text}
"""


def sample_passages(n: int) -> list[dict]:
    passages = load_passages()
    # Passages need a section, since the expected answer is identified by it.
    clean_passages = [
        passage
        for passage in passages
        if passage["section"] and len(passage["text"].strip()) >= 300
    ]

    random.seed(42)
    return random.sample(clean_passages, n)


class GeneratedQuestion(BaseModel):
    question: str
    answer: str


def generate_question(client: anthropic.Anthropic, passage: dict) -> dict:
    """Ask Claude to write a question and its answer from one passage."""
    prompt = QUESTION_PROMPT.format(
        source=passage["source"],
        section=passage["section"],
        text=passage["text"],
    )

    response = client.messages.parse(
        model=GENERATOR_MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
        output_format=GeneratedQuestion,
    )

    return response.parsed_output.model_dump()


def main() -> None:
    client = anthropic.Anthropic()
    passages = sample_passages(N_QUESTIONS)
    questions = []

    for index, passage in enumerate(passages, start=1):
        generated = generate_question(client, passage)
        questions.append(
            {
                "question": generated["question"],
                "expected_passages": [
                    {"source": passage["source"], "section": passage["section"]}
                ],
                "expected_answer": generated["answer"],
            }
        )
        print(f"[{index}/{len(passages)}] {generated['question']}")

    OUTPUT_PATH.write_text(
        json.dumps(questions, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved {len(questions)} questions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
