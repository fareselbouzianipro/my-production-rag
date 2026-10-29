"""Evaluate retrieval on the evaluation set and save the run to experiments/.

Usage: uv run python scripts/evaluate_retrieval.py --name baseline

Metrics, computed for each k in K_VALUES (see docs/ROADMAP.md):
- recall@k:    did we retrieve everything we need?
- precision@k: how much of what we retrieve is noise?
- mrr@k:       is the right passage ranked high?
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from statistics import mean

from production_rag.config import settings
from production_rag.embeddings import create_embedding_model
from production_rag.evaluation import (
    passage_key,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from production_rag.vector_store import create_vector_store, search_documents

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = PROJECT_ROOT / "data" / "eval" / "questions.json"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
K_VALUES = (1, 3, 5)

METRICS = {
    "recall": recall_at_k,
    "precision": precision_at_k,
    "mrr": reciprocal_rank_at_k,
}


def save_run(name: str, metrics: dict, per_question: list[dict]) -> Path:
    """Save the configuration, scores and per-question results of a run."""
    date = datetime.now().astimezone().isoformat(timespec="seconds")
    run = {
        "name": name,
        "date": date,
        "config": {
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "k_values": list(K_VALUES),
        },
        "metrics": metrics,
        "questions": per_question,
    }

    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    path = EXPERIMENTS_DIR / f"{date.replace(':', '-')}_{name}.json"
    path.write_text(
        json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def score_question(results: list[dict], expected_passages: list[dict]) -> dict:
    """Compute every metric at every k for one question."""
    return {
        f"{name}@{k}": metric(results, expected_passages, k=k)
        for name, metric in METRICS.items()
        for k in K_VALUES
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Experiment name, e.g. baseline")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    client = create_vector_store()
    model = create_embedding_model()

    per_question = []

    for index, item in enumerate(questions, start=1):
        question = item["question"]
        expected_passages = item["expected_passages"]

        results = search_documents(client, question, model, limit=max(K_VALUES))
        scores = score_question(results, expected_passages)

        per_question.append(
            {
                "question": question,
                "expected": expected_passages,
                "retrieved": [
                    {
                        "chunk_id": result["chunk_id"],
                        "source": result["source"],
                        "section": result["section"],
                        "score": round(result["score"], 4),
                    }
                    for result in results
                ],
                "metrics": scores,
            }
        )

        print(f"\n[{index}/{len(questions)}] {question}")
        print("  Expected:")
        for passage in expected_passages or [{"source": "NONE"}]:
            print(f"    - {passage['source']} | {passage.get('section')}")

        print("  Retrieved:")
        for rank, result in enumerate(results, start=1):
            print(
                f"    {rank}. {result['source']} | {result['section']} "
                f"(score={result['score']:.3f})"
            )

        if not expected_passages:
            print("  Evaluation: UNANSWERABLE — excluded from metrics.")
            continue

        retrieved = {passage_key(result) for result in results}
        missing = [p for p in expected_passages if passage_key(p) not in retrieved]
        for passage in missing:
            print(f"  Missing: {passage['source']} | {passage['section']}")

    answerable = [q["metrics"] for q in per_question if q["expected"]]
    metrics = {"questions": len(questions), "answerable": len(answerable)}
    for key in answerable[0]:
        metrics[key] = round(mean(scores[key] for scores in answerable), 4)

    print("\n=== RETRIEVAL METRICS ===")
    print(f"Questions evaluated: {len(questions)} ({len(answerable)} answerable)")
    print(f"{'k':>3} | {'recall':>7} | {'precision':>9} | {'mrr':>6}")
    for k in K_VALUES:
        print(
            f"{k:>3} | {metrics[f'recall@{k}']:>7.3f} | "
            f"{metrics[f'precision@{k}']:>9.3f} | {metrics[f'mrr@{k}']:>6.3f}"
        )

    path = save_run(args.name, metrics, per_question)
    print(f"\nRun saved to {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
