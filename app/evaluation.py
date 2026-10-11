from collections.abc import Sequence
from typing import Any


def recall_at_k(
    results: Sequence[dict[str, Any]],
    expected_sources: Sequence[str],
    k: int = 3,
) -> float | None:
    """Measure the fraction of expected sources found in top-k results."""

    if k <= 0:
        raise ValueError("k must be positive")

    expected = set(expected_sources)

    if not expected:
        return None

    retrieved = {result["source"] for result in results[:k] if "source" in result}

    return len(retrieved & expected) / len(expected)


def reciprocal_rank_at_k(
    results: Sequence[dict[str, Any]],
    expected_sources: Sequence[str],
    k: int = 3,
) -> float | None:
    """Measure the rank of the first relevant result within top-k."""
    if k <= 0:
        raise ValueError("k must be positive")

    expected = set(expected_sources)

    if not expected:
        return None

    for rank, result in enumerate(results[:k], start=1):
        if result.get("source") in expected:
            return 1.0 / rank

    return 0.0
