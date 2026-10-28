"""Evaluate retrieval on the evaluation set and save the run to experiments/.

Usage: uv run python scripts/evaluate_retrieval.py --name baseline
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
    recall_at_k,
    reciprocal_rank_at_k,
)
from production_rag.vector_store import create_vector_store, search_documents

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = PROJECT_ROOT / "data" / "eval" / "questions.json"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
K = 3


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
            "k": K,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Experiment name, e.g. baseline")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    client = create_vector_store()
    model = create_embedding_model()

    recalls = []
    reciprocal_ranks = []
    per_question = []

    for index, item in enumerate(questions, start=1):
        question = item["question"]
        expected_passages = item["expected_passages"]

        results = search_documents(
            client,
            question,
            model,
            limit=K,
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

        recall = recall_at_k(results, expected_passages, k=K)
        mrr = reciprocal_rank_at_k(results, expected_passages, k=K)

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
                "recall": recall,
                "mrr": mrr,
            }
        )

        if not expected_passages:
            print("  Evaluation: UNANSWERABLE — excluded from Recall/MRR.")
            continue

        recalls.append(recall)
        reciprocal_ranks.append(mrr)

        retrieved = {passage_key(result) for result in results}
        missing = [p for p in expected_passages if passage_key(p) not in retrieved]

        print(f"  Recall@{K}: {recall:.3f}")
        print(f"  MRR@{K}:    {mrr:.3f}")
        for passage in missing:
            print(f"  Missing: {passage['source']} | {passage['section']}")

    print("\n=== RETRIEVAL BASELINE ===")
    print(f"Questions evaluated: {len(questions)}")
    print(f"Answerable questions: {len(recalls)}")
    print(f"Recall@{K}: {mean(recalls):.3f}" if recalls else "Recall: N/A")
    print(
        f"MRR@{K}:    {mean(reciprocal_ranks):.3f}" if reciprocal_ranks else "MRR: N/A"
    )

    metrics = {
        "questions": len(questions),
        "answerable": len(recalls),
        f"recall@{K}": round(mean(recalls), 4) if recalls else None,
        f"mrr@{K}": round(mean(reciprocal_ranks), 4) if reciprocal_ranks else None,
    }
    path = save_run(args.name, metrics, per_question)
    print(f"\nRun saved to {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
