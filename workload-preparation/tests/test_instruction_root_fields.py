"""Invented fixtures check source-root context, not native task/model success."""

import json

import pytest

from vllm_hust_workload_data.assets import iter_rows


def test_root_fields_preserved_separately_and_isolated(tmp_path):
    source = tmp_path / "task.json"
    doc = {
        "Definition": ["KEEP EXACT\n"],
        "Instances": [
            {"id": "a", "input": "original input", "output": ["reference"]},
            {"id": "b", "input": "next input", "output": ["other reference"]},
        ],
    }
    source.write_text(json.dumps(doc), encoding="utf-8")
    rows = iter_rows(source, records_path=["Instances"], root_fields=["Definition"])
    first = next(rows)
    assert first["Definition"] == ["KEEP EXACT\n"]
    first["Definition"].append("mutate private result")
    assert next(rows)["Definition"] == ["KEEP EXACT\n"]
    assert "Definition" not in doc["Instances"][0]


@pytest.mark.parametrize("failure", ["missing", "collision", "duplicate"])
def test_invalid_root_contract_rejected_without_payload(tmp_path, failure):
    source = tmp_path / "task.json"
    doc = {"Definition": ["PRIVATE_FIXTURE"], "Instances": [{"input": "x"}]}
    fields = ["Definition"]
    if failure == "missing":
        fields = ["Missing"]
    elif failure == "collision":
        doc["Instances"][0]["Definition"] = ["collision"]
    else:
        fields *= 2
    source.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError) as error:
        list(iter_rows(source, records_path=["Instances"], root_fields=fields))
    assert "PRIVATE_FIXTURE" not in str(error.value)


@pytest.mark.parametrize("fields", [None, {}, "", True, [""], [1]])
def test_invalid_root_field_types_fail_closed(tmp_path, fields):
    source = tmp_path / "task.json"
    source.write_text('{"Instances":[]}', encoding="utf-8")
    with pytest.raises(ValueError):
        list(iter_rows(source, records_path=["Instances"], root_fields=fields))
