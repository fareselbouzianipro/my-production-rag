from collections.abc import Sequence
from typing import Any


def passage_key(item: dict[str, Any]) -> tuple[str, str | None]:
    """Identify a passage by its source and section.

    Unlike chunk ids, this stays valid when the chunking strategy changes.
    """
    return item.get("source"), item.get("section")


def recall_at_k(
    results: Sequence[dict[str, Any]],
    expected_passages: Sequence[dict[str, Any]],
    k: int = 3,
) -> float | None:
    """Measure the fraction of expected passages found in top-k results."""

    if k <= 0:
        raise ValueError("k must be positive")

    expected = {passage_key(passage) for passage in expected_passages}

    if not expected:
        return None

    retrieved = {passage_key(result) for result in results[:k]}

    return len(retrieved & expected) / len(expected)


def reciprocal_rank_at_k(
    results: Sequence[dict[str, Any]],
    expected_passages: Sequence[dict[str, Any]],
    k: int = 3,
) -> float | None:
    """Measure the rank of the first relevant result within top-k."""
    if k <= 0:
        raise ValueError("k must be positive")

    expected = {passage_key(passage) for passage in expected_passages}

    if not expected:
        return None

    for rank, result in enumerate(results[:k], start=1):
        if passage_key(result) in expected:
            return 1.0 / rank

    return 0.0
