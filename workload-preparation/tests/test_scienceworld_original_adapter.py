"""Small schema fixtures, not claimed publisher tasks or native execution."""

import json
from zipfile import ZipFile

import pytest

from vllm_hust_workload_data.scienceworld import adapt_case, original_rows


def archive(tmp_path, cases=None):
    path = tmp_path / "fixture.zip"
    if cases is None:
        cases = [
            {
                "variationIdx": 27,
                "fold": "test",
                "taskDescription": "fixture goal",
                "path": [{"action": "fixture-gold-action", "score": "1.0"}],
            }
        ]
    payload = {
        "3": {"taskIdx": 3, "taskName": "fixture-task", "goldActionSequences": cases}
    }
    with ZipFile(path, "w") as bundle:
        bundle.writestr("original.json", json.dumps(payload))
    return path


def test_native_identity_and_reference_separation(tmp_path):
    rows = list(original_rows(archive(tmp_path)))
    record = adapt_case(rows[0], source={"source_id": "SCIENCEWORLD"}, index=0)
    assert record.task["task_id"] == "3:27"
    assert record.task["input"]["taskDescription"] == "fixture goal"
    assert "fixture-gold-action" not in json.dumps(record.input_record())
    assert (
        record.outcome_record()["publisher_gold_path"][0]["action"]
        == "fixture-gold-action"
    )
    assert record.replay["annotations"]["fold"] == "test"
    assert record.replay["annotations"]["original_variation_offset"] == 0
    with pytest.raises(ValueError, match="not a ready model request"):
        record.request_body()


def test_reference_copy_is_independent(tmp_path):
    row = next(original_rows(archive(tmp_path)))
    record = adapt_case(row, source={"source_id": "SCIENCEWORLD"}, index=0)
    row["path"][0]["action"] = "changed"
    assert (
        record.outcome_record()["publisher_gold_path"][0]["action"]
        == "fixture-gold-action"
    )


def test_duplicate_variations_rejected(tmp_path):
    case = {
        "variationIdx": 27,
        "fold": "test",
        "taskDescription": "fixture",
        "path": [],
    }
    with pytest.raises(ValueError, match="duplicate"):
        list(original_rows(archive(tmp_path, [case, case])))


def test_unknown_fields_not_silently_dropped(tmp_path):
    case = {
        "variationIdx": 27,
        "fold": "test",
        "taskDescription": "fixture",
        "path": [],
        "unknown": 1,
    }
    with pytest.raises(ValueError, match="explicit mapping"):
        list(original_rows(archive(tmp_path, [case])))
