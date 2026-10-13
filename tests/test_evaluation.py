import pytest

from production_rag.evaluation import recall_at_k, reciprocal_rank_at_k


def test_recall_at_k_counts_unique_expected_sources():
    results = [
        {"source": "deployment.md"},
        {"source": "deployment.md"},
        {"source": "access-policy.md"},
    ]

    expected = ["deployment.md", "access-policy.md"]

    assert recall_at_k(results, expected, k=2) == 0.5
    assert recall_at_k(results, expected, k=3) == 1.0


def test_reciprocal_rank_uses_first_relevant_result():
    results = [
        {"source": "deployment.md"},
        {"source": "access-policy.md"},
        {"source": "logging-policy.md"},
    ]

    assert reciprocal_rank_at_k(
        results,
        ["logging-policy.md"],
        k=3,
    ) == pytest.approx(1 / 3)


def test_unanswerable_question_is_excluded_from_metrics():
    results = [{"source": "deployment.md"}]

    assert recall_at_k(results, [], k=3) is None
    assert reciprocal_rank_at_k(results, [], k=3) is None


def test_k_must_be_positive():
    with pytest.raises(ValueError):
        recall_at_k([], ["deployment.md"], k=0)
