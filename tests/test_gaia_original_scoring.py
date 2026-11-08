"""Invented score fixtures; no gated task, real prediction or official score claim."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "gaia_score",
    Path(__file__).resolve().parents[1] / "scripts/score_gaia_original_predictions.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    data = tmp_path / "original-fixture"
    data.mkdir()
    scorer = tmp_path / "fixture-scorer.py"
    scorer.write_text(
        "def question_scorer(prediction, reference):\n print('fixture-only-private-evaluation')\n return prediction == reference\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        MODULE, "SCORER_SHA", hashlib.sha256(scorer.read_bytes()).hexdigest()
    )
    records = [
        SimpleNamespace(
            task={"task_id": f"invented-{i}"},
            replay={"source_row_index": i, "annotations": {"level": "1"}},
            outcome={"reference_answer": f"fixture-reference-{i}"},
        )
        for i in range(3)
    ]

    def original_records(source_id, **kwargs):
        assert source_id == "GAIA" and kwargs == {
            "data_root": data,
            "files": [MODULE.VALIDATION_FILE],
        }
        return iter(records)

    monkeypatch.setattr(MODULE, "iter_workload", original_records)
    predictions = tmp_path / "saved-fixture.jsonl"
    predictions.write_text(
        json.dumps({"task_id": "invented-1", "prediction": None})
        + "\n"
        + json.dumps({"task_id": "invented-0", "prediction": "fixture-reference-0"})
        + "\n",
        encoding="utf-8",
    )
    return {
        "data_root": data,
        "scorer_file": scorer,
        "predictions_file": predictions,
        "output": tmp_path / "fresh-private-score",
    }


def test_id_join_null_failure_and_unscored_tasks_remain_visible(inputs, capsys):
    report = MODULE.score(**inputs)
    assert report["correct"] == 1 and report["accuracy"] == 0.5
    assert (
        report["selected_original_validation_tasks"] == 2
        and report["unscored_original_validation_tasks"] == 1
    )
    assert [x["original_row_index"] for x in report["tasks"]] == [1, 0]
    assert report["tasks"][0]["prediction_is_null"]
    assert report["model_calls"] == report["native_agent_tasks_executed"] == 0
    assert (
        not report["source_checkbox_eligible"]
        and not report["generation_provenance_verified"]
    )
    assert "fixture-only-private-evaluation" not in capsys.readouterr().out
    assert (
        "fixture-only-private-evaluation"
        in (inputs["output"] / "author-stdout-private.txt").read_text()
    )
    assert (
        "fixture-reference"
        not in (inputs["output"] / "receipt-private.json").read_text()
    )


@pytest.mark.parametrize(
    "kind",
    ["unknown-or-test-id", "duplicate", "oracle-field", "numeric-prediction", "empty"],
)
def test_invalid_prediction_contract_never_silently_filters_or_scores(inputs, kind):
    rows = [{"task_id": "invented-0", "prediction": "text"}]
    if kind == "unknown-or-test-id":
        rows[0]["task_id"] = "invented-test-id"
    elif kind == "duplicate":
        rows.append(dict(rows[0]))
    elif kind == "oracle-field":
        rows[0]["reference_answer"] = "fixture-oracle"
    elif kind == "numeric-prediction":
        rows[0]["prediction"] = 1
    else:
        rows = []
    inputs["predictions_file"].write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        MODULE.score(**inputs)
    assert not inputs["output"].exists()


def test_changed_author_implementation_is_rejected_before_output(inputs):
    inputs["scorer_file"].write_text(
        "raise RuntimeError('fixture changed')", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unchanged pinned official"):
        MODULE.score(**inputs)
    assert not inputs["output"].exists()


def test_existing_output_is_preserved(inputs):
    inputs["output"].mkdir()
    sentinel = inputs["output"] / "existing.txt"
    sentinel.write_text("fixture retained", encoding="utf-8")
    with pytest.raises(FileExistsError):
        MODULE.score(**inputs)
    assert sentinel.read_text() == "fixture retained"


def test_original_scorer_failure_is_retained_not_converted_to_zero(inputs, monkeypatch):
    def fail(*args):
        raise RuntimeError("fixture author failure")

    monkeypatch.setattr(
        MODULE, "load_author", lambda p: SimpleNamespace(question_scorer=fail)
    )
    with pytest.raises(RuntimeError):
        MODULE.score(**inputs)
    report = json.loads((inputs["output"] / "receipt-private.json").read_text())
    assert (
        report["status"] == "FAILED_ORIGINAL_GAIA_SCORING_RETAINED"
        and "accuracy" not in report
    )
    assert not report["source_checkbox_eligible"]


def test_default_plan_does_not_read_assets_or_execute_scorer(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["score_gaia_original_predictions.py"])
    monkeypatch.setattr(
        MODULE, "score", lambda **kw: pytest.fail("plan must not score")
    )
    MODULE.main()
    assert json.loads(capsys.readouterr().out)["status"] == "PLAN_ONLY"
