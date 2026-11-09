"""Structural fixtures only, not scientific Agent or dataset execution."""

from copy import deepcopy
import hashlib
import json
import zipfile

import pytest

from vllm_hust_workload_data.adapters import adapt_row
from vllm_hust_workload_data import bixbench
from vllm_hust_workload_data.bindings import bind_source
from vllm_hust_workload_data.consumer import resolve_input
from vllm_hust_workload_data.pipeline import prepared_rows, run
from vllm_hust_workload_data.registry import adapter_spec


def original_fixture():
    return {
        "question_id": "fixture-q1",
        "capsule_uuid": "33b801bb-9b47-4a0a-9314-05325c82fde7",
        "data_folder": "CapsuleFolder-33b801bb-9b47-4a0a-9314-05325c82fde7.zip",
        "question": "fixture scientific question",
        "ideal": "reference ideal fixture",
        "distractors": ["reference distractor fixture"],
        "answer": False,
        "hypothesis": "reference hypothesis fixture",
        "result": "reference result fixture",
        "future_extension": {"values": [None, True, 1]},
    }


def adapt(row):
    return adapt_row(
        "BIXBENCH",
        row,
        source={"source_id": "BIXBENCH"},
        spec=adapter_spec("BIXBENCH"),
        index=0,
    )


def test_all_original_fields_roundtrip_without_reference_input_leak():
    row = original_fixture()
    record = adapt(row)
    assert bixbench.restore_original_task(record) == row
    assert record.task["input"] == {
        key: row[key] for key in ("question", "data_folder")
    }
    assert "reference ideal fixture" not in json.dumps(record.input_record())
    assert "future_extension" not in json.dumps(record.input_record())
    assert record.runtime["kind"] == "native_environment"
    assert record.runtime["execution"] == "not_executed"
    with pytest.raises(ValueError, match="not a ready model request"):
        record.request_body()
    record.outcome["original_metadata"]["future_extension"]["values"].append(2)
    assert row["future_extension"]["values"] == [None, True, 1]


@pytest.mark.parametrize(
    "field,value",
    [
        ("question", None),
        ("ideal", 1),
        ("data_folder", "../escape.zip"),
        ("question_id", ""),
    ],
)
def test_invalid_original_contract_has_no_fallback(field, value):
    row = original_fixture()
    row[field] = value
    with pytest.raises(ValueError):
        adapt(row)


def test_input_reference_collision_is_not_silently_restored():
    record = adapt(original_fixture())
    record.task["input"]["ideal"] = "changed fixture"
    with pytest.raises(ValueError, match="separation lost"):
        bixbench.restore_original_task(record)


def test_capsule_accessor_checks_identity_without_extracting(tmp_path, monkeypatch):
    row = original_fixture()
    path = tmp_path / row["data_folder"]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("Data/fixture.csv", "value\n1\n")
    asset = {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    monkeypatch.setattr(bixbench, "capsule_asset", lambda name: deepcopy(asset))
    record = adapt(row)
    assert bixbench.original_capsule_path(record, data_root=tmp_path) == path
    assert not (tmp_path / "Data").exists()
    with path.open("ab") as handle:
        handle.write(b"changed fixture")
    with pytest.raises(ValueError, match="identity mismatch"):
        bixbench.original_capsule_path(record, data_root=tmp_path)


def test_full_index_roundtrip_does_not_pretend_missing_capsule_ready(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    row = original_fixture()
    (data / "fixture.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest = bind_source(
        "BIXBENCH",
        data_root=data,
        files=["fixture.jsonl"],
        revision="fixture-not-publisher",
    )
    binding = tmp_path / "binding.json"
    binding.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "prepared"
    report = run(
        {
            "sources": [
                {
                    "source_id": "BIXBENCH",
                    "data_root": str(data),
                    "manifest": str(binding),
                }
            ]
        },
        output=output,
    )
    assert report["native_tasks_run"] == 0
    indexed = list(prepared_rows(output / "workload.sqlite"))
    record = resolve_input(
        output / "workload.sqlite", indexed[0]["record_id"], data_root=data
    )
    assert bixbench.restore_original_task(record) == row
    with pytest.raises(FileNotFoundError):
        bixbench.original_capsule_path(record, data_root=data)
