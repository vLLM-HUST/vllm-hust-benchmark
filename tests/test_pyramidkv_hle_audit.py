"""HLE source inventory checks; no task solving or grading."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "hle_audit", Path(__file__).parents[1] / "scripts/audit_pyramidkv_hle.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def row():
    return {
        "id": "fixture",
        "question": "synthetic question",
        "answer": "one",
        "image": "",
        "image_preview": None,
        "rationale_image": "reference-only-image",
        "answer_type": "exactMatch",
        "Verified_Classes": "Gold subset",
        "category": "Math",
    }


def test_gold_rationale_image_is_not_a_solver_image():
    result = audit.summarize_row(row(), "Gold_subset", 0)
    assert result["rationale_image_present"]
    assert not result["requires_image_handling"]
    assert "question" not in result and "answer" not in result


def test_preview_only_tasks_cannot_silently_become_text_only():
    source = row()
    source["image_preview"] = {"bytes": "synthetic image", "path": None}
    result = audit.summarize_row(source, "Gold_subset", 0)
    assert result["requires_image_handling"] and result["image_preview_present"]
    assert not result["problem_image_present"]


@pytest.mark.parametrize("answer", [False, True, 0, 1, 4.73, "False", "1"])
def test_revised_answers_preserve_types_and_zero_values(answer):
    source = {**row(), "answer": answer}
    result = audit.summarize_row(source, "Revision_subset", 0)
    assert result["answer_value_type"] == type(answer).__name__
    assert result["answer_sha256"] == audit.sha(audit.canonical(answer))


@pytest.mark.parametrize("answer", [None, {}, [], float("nan"), float("inf"), ""])
def test_invalid_answer_is_rejected(answer):
    with pytest.raises((TypeError, ValueError)):
        audit.summarize_row({**row(), "answer": answer}, "Revision_subset", 0)


def test_changed_source_rejected_before_output(tmp_path):
    (tmp_path / "Gold_subset.jsonl").write_text("{}\n")
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        audit.audit(SimpleNamespace(assets=tmp_path, output=output))
    assert not output.exists()
