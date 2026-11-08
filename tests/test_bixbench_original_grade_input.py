import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "grade_input",
    Path(__file__).resolve().parents[1]
    / "scripts/prepare_bixbench_original_grade_input.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def example():
    original = {
        "question_id": "fixture-1",
        "question": "fixture question",
        "ideal": "fixture reference",
        "eval_mode": "str_verifier",
    }
    trajectory = {
        "problem_id": "fixture-1",
        "ideal_answer": "fixture reference",
        "agent_answer": {"fixture-1": "unchanged answer"},
    }
    return original, trajectory


def test_exact_join_preserves_original_question_reference_and_mode():
    original, trajectory = example()
    row = MODULE.grade_row(original, trajectory, lambda value: value)
    assert row == {
        "question_id": "fixture-1",
        "question": original["question"],
        "target": original["ideal"],
        "predicted": "unchanged answer",
        "evaluation_mode": "str_verifier",
    }
    assert trajectory["agent_answer"] == {"fixture-1": "unchanged answer"}


@pytest.mark.parametrize(
    "key,value", [("problem_id", "different"), ("ideal_answer", "rewritten")]
)
def test_identity_or_reference_mismatch_rejected(key, value):
    original, trajectory = example()
    trajectory[key] = value
    with pytest.raises(ValueError):
        MODULE.grade_row(original, trajectory, lambda answer: answer)


def test_empty_answer_is_retained_not_dropped_or_replaced():
    original, trajectory = example()
    trajectory["agent_answer"] = ""
    assert (
        MODULE.grade_row(original, trajectory, lambda value: value)["predicted"] == ""
    )


@pytest.mark.parametrize("mode", ["str_verifier", "range_verifier", "llm_verifier"])
def test_publisher_raw_mode_mapped_to_csv_without_changing_rule(mode):
    original, trajectory = example()
    original["eval_mode"] = mode
    row = MODULE.grade_row(original, trajectory, lambda value: value)
    assert row["evaluation_mode"] == mode
    assert "eval_mode" not in row


def test_explicit_csv_mode_is_supported_but_conflicts_rejected():
    original, trajectory = example()
    original["evaluation_mode"] = original.pop("eval_mode")
    assert (
        MODULE.grade_row(original, trajectory, lambda value: value)["evaluation_mode"]
        == "str_verifier"
    )
    original["eval_mode"] = "range_verifier"
    with pytest.raises(ValueError, match="conflicting"):
        MODULE.grade_row(original, trajectory, lambda value: value)


def test_missing_original_mode_cannot_silently_default_to_llm():
    original, trajectory = example()
    original.pop("eval_mode")
    with pytest.raises(ValueError, match="missing"):
        MODULE.grade_row(original, trajectory, lambda value: value)


def test_plan_has_no_import_download_model_or_file_output(tmp_path):
    args = SimpleNamespace(
        author_root=tmp_path / "author",
        data_root=tmp_path / "data",
        native_output=tmp_path / "native",
        output=tmp_path / "out",
        limit=3,
        execute=False,
    )
    report = MODULE.prepare(args)
    assert report["status"] == "PLAN_ONLY"
    assert not report["scoring_executed"] and not list(tmp_path.iterdir())


def test_no_output_inside_original_data(tmp_path):
    args = SimpleNamespace(
        author_root=tmp_path / "author",
        data_root=tmp_path / "data",
        native_output=tmp_path / "native",
        output=tmp_path / "data" / "out",
        limit=3,
        execute=False,
    )
    with pytest.raises(ValueError, match="fresh"):
        MODULE.prepare(args)
