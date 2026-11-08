"""Tests for the retrieval evaluation. Stage: 5.

What this file is:
    Proves the search scoring, and that the labelled questions still match the handbook.

What it contains:
    - The question file loads, with unique ids and known kinds.
    - Every expected piece id exists in the handbook, so renaming a heading without updating
      the questions fails here, not silently in a score.
    - hit_at and reciprocal_rank on known rankings.
    - score: hits per mode and per kind, the misses listed, and unanswerable questions left out.

Used by:
    pytest
"""

from pathlib import Path

import pytest
from evals.retrieval_eval import (
    KINDS,
    QUESTIONS_PATH,
    hit_at,
    load_questions,
    reciprocal_rank,
    score,
)

from dental_agent.clinic.index import split_sections

HANDBOOK = Path(__file__).parent.parent / "data" / "docs" / "patient-handbook.md"


def test_the_question_file_loads_with_unique_ids_and_known_kinds():
    questions = load_questions(QUESTIONS_PATH)
    assert len({item["id"] for item in questions}) == len(questions) >= 30
    assert {item["kind"] for item in questions} == set(KINDS)


def test_every_expected_piece_exists_in_the_handbook():
    pieces = {item.piece_id for item in split_sections(HANDBOOK.read_text(), HANDBOOK.name)}
    for question in load_questions(QUESTIONS_PATH):
        missing = set(question["expected"]) - pieces
        assert not missing, (question["id"], missing)


def test_only_unanswerable_questions_expect_nothing():
    for question in load_questions(QUESTIONS_PATH):
        assert (question["kind"] == "unanswerable") == (not question["expected"]), question["id"]


@pytest.mark.parametrize(
    ("ranking", "k", "expected"),
    [(["a", "b", "c"], 3, True), (["x", "y", "a"], 3, True), (["x", "y", "z", "a"], 3, False)],
)
def test_hit_at_looks_only_at_the_top_k(ranking, k, expected):
    assert hit_at(ranking, {"a"}, k) is expected


def test_reciprocal_rank_is_one_over_the_first_answering_position():
    assert reciprocal_rank(["x", "a", "b"], {"a", "b"}) == 0.5
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_score_counts_hits_by_kind_and_lists_the_misses():
    questions = [
        {"id": "d01", "kind": "direct", "question": "q1", "expected": ["a"]},
        {"id": "w01", "kind": "reworded", "question": "q2", "expected": ["b"]},
        {"id": "u01", "kind": "unanswerable", "question": "q3", "expected": []},
    ]
    rankings = {"d01": ["a", "x"], "w01": ["x", "y", "z", "b"], "u01": ["a"]}
    result = score(questions, rankings)
    assert result["hits"] == 1
    assert result["scored"] == 2
    assert result["by_kind"]["direct"] == {"hits": 1, "scored": 1}
    assert result["by_kind"]["reworded"] == {"hits": 0, "scored": 1}
    assert result["misses"] == ["w01"]
    assert result["mrr"] == pytest.approx((1.0 + 0.25) / 2)
