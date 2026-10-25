"""Synthetic saved-result fixtures only, no original assets or model calls."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

spec = importlib.util.spec_from_file_location(
    "structeval_saved_audit",
    Path(__file__).parents[1] / "scripts/audit_structeval_saved_nonrender.py",
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def saved_fixture(tmp_path, indices=(0, 1, 2)):
    rows = [
        {
            "task_id": f"fixture-{i}",
            "rendering": False,
            "input_type": "Text",
            "output_type": "JSON",
            "prompt": "inert original fixture",
        }
        for i in range(6)
    ]
    run = tmp_path / "run"
    (run / "official-workflow").mkdir(parents=True)
    originals = [rows[i] for i in indices]
    generated = [{**r, "generation": "inert fixture output"} for r in originals]
    scored = [
        {**r, "render_score": 0, "key_validation_score": 0, "final_eval_score": 0}
        for r in generated
    ]
    receipt = {
        "source_id": "STRUCTEVAL",
        "status": "END_TO_END_EXECUTED_AND_SCORED",
        "max_tokens": None,
        "source_revision": "936be10548688c6d417cd435dd906158f19ba14b",  # pragma: allowlist secret - public dataset revision
        "runner_revision": "7781339a5b033965ac664133543ebd10773643dd",  # pragma: allowlist secret - public runner revision
        "original_row_indices": list(indices),
        "selected_samples": [r["task_id"] for r in originals],
        "unique_tasks": len(indices),
    }
    for relative, payload in [
        ("receipt.json", receipt),
        ("original-selected.private.json", originals),
        ("official-workflow/inference.json", generated),
        ("official-workflow/evaluation.json", scored),
    ]:
        (run / relative).write_text(json.dumps(payload), encoding="utf-8")
    return rows, run, scored


def test_saved_wrong_answers_reused_without_inference(tmp_path):
    rows, run, scored = saved_fixture(tmp_path)
    observed = []

    def rescore(records, output):
        observed.append(records)
        print("private detail must not be in summary")
        return deepcopy(scored)

    result = audit.audit_runs(
        rows, [run], tmp_path / "audit", rescore, lambda row: None
    )
    assert result["unique_original_tasks"] == 3
    assert result["new_model_calls"] == 0 and not result["source_checkbox_eligible"]
    assert len(observed) == 1
    assert "private detail" not in json.dumps(result)
    assert "inert original fixture" not in json.dumps(result)
    assert (tmp_path / "audit/run-0000/rescore-private.log").is_file()


@pytest.mark.parametrize(
    "failure",
    ["running", "budget", "rendering", "generation", "score", "missing_score"],
)
def test_invalid_or_incompatible_saved_runs_rejected(tmp_path, failure):
    rows, run, scored = saved_fixture(tmp_path)
    if failure in {"running", "budget"}:
        receipt = audit.read_json(run / "receipt.json")
        if failure == "running":
            receipt["status"] = "STARTED"
        else:
            receipt.pop("max_tokens")
        (run / "receipt.json").write_text(json.dumps(receipt))
    elif failure == "rendering":
        rows[0]["rendering"] = True
    elif failure == "generation":
        scored[0]["generation"] = "changed"
        (run / "official-workflow/evaluation.json").write_text(json.dumps(scored))

    def rescore(records, output):
        result = deepcopy(scored)
        if failure == "score":
            result[0]["final_eval_score"] = 1
        elif failure == "missing_score":
            result[0].pop("final_eval_score")
        return result

    with pytest.raises(ValueError):
        audit.audit_runs(rows, [run], tmp_path / "audit", rescore, lambda row: None)


def test_duplicate_saved_runs_are_not_counted_twice(tmp_path):
    rows, run, scored = saved_fixture(tmp_path)
    with pytest.raises(ValueError, match="distinct"):
        audit.audit_runs(
            rows, [run, run], tmp_path / "audit", lambda a, b: scored, lambda row: None
        )


def test_overlapping_tasks_in_distinct_runs_are_rejected(tmp_path):
    rows, run, scored = saved_fixture(tmp_path)
    other = tmp_path / "second-run"
    shutil.copytree(run, other)
    with pytest.raises(ValueError, match="overlapping original tasks"):
        audit.audit_runs(
            rows,
            [run, other],
            tmp_path / "audit",
            lambda a, b: deepcopy(scored),
            lambda row: None,
        )
