"""CPU contract tests for the original MATH500 sample runner; no model calls."""

from types import SimpleNamespace

import pytest

from scripts.run_math500_original_samples import (
    build_prompt,
    extract_boxed_answer,
    select_task_records,
)


def test_extracts_last_balanced_boxed_answer_with_nested_braces():
    text = r"work \boxed{wrong}, then \boxed{\frac{1}{2}}"
    assert extract_boxed_answer(text) == r"\frac{1}{2}"


def test_rejects_missing_or_unbalanced_boxed_answer():
    assert extract_boxed_answer("no final answer") is None
    assert extract_boxed_answer(r"final \boxed{unfinished") is None


def test_prompt_preserves_problem_and_excludes_reference_fields():
    problem = "ORIGINAL_PROBLEM_SENTINEL"
    prompt = build_prompt(problem)
    assert prompt.endswith(problem)
    assert r"\boxed{ANSWER}" in prompt
    assert "solution" not in prompt.lower()
    assert "reference" not in prompt.lower()


def test_failure_recovery_only_selects_ids_from_fixed_first_three():
    records = [
        SimpleNamespace(replay={"annotations": {"unique_id": f"id-{n}"}})
        for n in range(3)
    ]
    assert select_task_records(records, ["id-1", "id-0"]) == records[:2]
    with pytest.raises(ValueError, match="distinct"):
        select_task_records(records, ["id-0", "id-0"])
    with pytest.raises(ValueError, match="not in the fixed first-three"):
        select_task_records(records, ["id-9"])
