"""Software fixtures only; real QReCC native workflow remains unverified."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "qrecc_score",
    Path(__file__).resolve().parents[1] / "scripts/score_qrecc_original_qa.py",
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    data = tmp_path / "test.json"
    rows = [
        {"Conversation_no": 1, "Turn_no": i + 1, "Answer": f"fixture {i}"}
        for i in range(3)
    ]
    data.write_text(json.dumps(rows), encoding="utf-8")
    scorer = tmp_path / "evaluate_qa.py"
    scorer.write_text(
        "def compute_exact(gold, pred): return int(gold == pred)\n"
        "def compute_f1(gold, pred): return float(gold == pred)\n",
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.json"
    items = [
        {
            "original_row_index": i,
            "Conversation_no": 1,
            "Turn_no": i + 1,
            "prediction": f"fixture {i}",
        }
        for i in range(3)
    ]
    predictions.write_text(json.dumps(items), encoding="utf-8")
    monkeypatch.setattr(
        RUNNER, "TEST_SHA", hashlib.sha256(data.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        RUNNER, "SCORER_SHA", hashlib.sha256(scorer.read_bytes()).hexdigest()
    )
    return dict(
        data_file=data,
        scorer_file=scorer,
        predictions_file=predictions,
        output=tmp_path / "out",
    )


def test_unchanged_functions_score_all_selected_items_including_wrong_answers(inputs):
    path = inputs["predictions_file"]
    items = json.loads(path.read_text("utf-8"))
    items[1]["prediction"] = "wrong fixture answer"
    path.write_text(json.dumps(items), encoding="utf-8")
    result = RUNNER.score(**inputs)
    assert result["samples"] == 3
    assert result["exact_match"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["tasks"][1]["exact_match"] == 0
    assert not result["source_checkbox_eligible"]
    assert not result["retrieval_executed"]
    assert not result["generation_provenance_verified"]


@pytest.mark.parametrize(
    "change", ["duplicate", "reference-leak", "wrong-turn", "reordered", "too-small"]
)
def test_prediction_contract_does_not_silently_filter(inputs, change):
    path = inputs["predictions_file"]
    items = json.loads(path.read_text("utf-8"))
    if change == "duplicate":
        items[1] = items[0]
    elif change == "reference-leak":
        items[0]["Answer"] = "reference must not be accepted as a prediction field"
    elif change == "wrong-turn":
        items[0]["Turn_no"] = 100
    elif change == "reordered":
        items.reverse()
    else:
        items = items[:2]
    path.write_text(json.dumps(items), encoding="utf-8")
    with pytest.raises(ValueError):
        RUNNER.score(**inputs)
    assert not inputs["output"].exists()


@pytest.mark.parametrize("key", ["data_file", "scorer_file"])
def test_pinned_identity_rejects_changed_input_before_import(inputs, key):
    with inputs[key].open("a", encoding="utf-8") as stream:
        stream.write("\n# changed fixture\n")
    with pytest.raises(ValueError):
        RUNNER.score(**inputs)
    assert not inputs["output"].exists()
