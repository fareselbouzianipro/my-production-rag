import json
from pathlib import Path
from statistics import mean

from production_rag.embeddings import create_embedding_model
from production_rag.evaluation import (
    passage_key,
    recall_at_k,
    reciprocal_rank_at_k,
)
from production_rag.vector_store import create_vector_store, search_documents

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = PROJECT_ROOT / "data" / "eval" / "questions.json"
K = 3


def main() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    client = create_vector_store()
    model = create_embedding_model()

    recalls = []
    reciprocal_ranks = []

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

        if not expected_passages:
            print("  Evaluation: UNANSWERABLE — excluded from Recall/MRR.")
            continue

        recall = recall_at_k(results, expected_passages, k=K)
        mrr = reciprocal_rank_at_k(results, expected_passages, k=K)

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


if __name__ == "__main__":
    main()
