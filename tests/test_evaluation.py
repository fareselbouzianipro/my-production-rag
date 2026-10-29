import pytest

from production_rag.evaluation import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)

OFFSIDE = {"source": "ifab-laws-of-the-game", "section": "Law 11 – Offside"}
THROW_IN = {"source": "ifab-laws-of-the-game", "section": "Law 15 – The Throw-in"}
MANIPULATION = {"source": "fifa-disciplinary-code", "section": "20. MANIPULATION"}


def test_recall_at_k_counts_unique_expected_passages():
    results = [OFFSIDE, OFFSIDE, THROW_IN]

    expected = [OFFSIDE, THROW_IN]

    assert recall_at_k(results, expected, k=2) == 0.5
    assert recall_at_k(results, expected, k=3) == 1.0


def test_recall_at_k_requires_the_right_section_not_just_the_source():
    results = [THROW_IN]

    assert recall_at_k(results, [OFFSIDE], k=3) == 0.0


def test_reciprocal_rank_uses_first_relevant_result():
    results = [OFFSIDE, THROW_IN, MANIPULATION]

    assert reciprocal_rank_at_k(results, [MANIPULATION], k=3) == pytest.approx(1 / 3)


def test_precision_at_k_counts_noise_in_retrieved_chunks():
    # Two chunks of the expected section, one unrelated chunk.
    results = [OFFSIDE, MANIPULATION, OFFSIDE]

    assert precision_at_k(results, [OFFSIDE], k=1) == 1.0
    assert precision_at_k(results, [OFFSIDE], k=3) == pytest.approx(2 / 3)


def test_precision_at_k_counts_missing_results_as_not_relevant():
    assert precision_at_k([OFFSIDE], [OFFSIDE], k=3) == pytest.approx(1 / 3)


def test_unanswerable_question_is_excluded_from_metrics():
    results = [OFFSIDE]

    assert recall_at_k(results, [], k=3) is None
    assert reciprocal_rank_at_k(results, [], k=3) is None
    assert precision_at_k(results, [], k=3) is None


def test_k_must_be_positive():
    with pytest.raises(ValueError):
        recall_at_k([], [OFFSIDE], k=0)
