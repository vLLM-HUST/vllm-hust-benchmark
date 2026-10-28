"""Tiny inert Parquet/mapping fixtures; no official data or model execution."""

import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
runner = importlib.import_module("prepare_jsonschemabench_original_records")


def test_original_parquet_values_ids_and_file_views_remain_lossless(
    tmp_path, monkeypatch
):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from vllm_hust_workload_data.consumer import resolve_input
    from vllm_hust_workload_data.pipeline import prepared_rows
    from vllm_hust_workload_data.registry import adapter_spec

    data = tmp_path / "data"
    data.mkdir()
    rows = [
        {
            "json_schema": ' {"properties": {"answer": {"type": "string"}}} ',
            "unique_id": "same-original-id",
        },
        {"json_schema": "false", "unique_id": "same-original-id"},
    ]
    files = []
    for name in ["Github_easy/test.parquet", "data/test.parquet"]:
        path = data / name
        path.parent.mkdir(exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path)
        files.append(
            {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    pins = tmp_path / "pins.json"
    mapping = json.loads(runner.MAPPING.read_text(encoding="utf-8"))
    pins.write_text(
        json.dumps(
            {"dataset": {"revision": mapping["source"]["revision"], "files": files}}
        )
    )
    monkeypatch.setattr(runner, "PINS", pins)
    output = tmp_path / "prepared"
    receipt = runner.prepare(data_root=data, output=output)
    assert receipt["original_rows_in_selected_views"] == 4
    assert receipt["model_calls"] == 0 and receipt["source_checkbox_eligible"] is False
    assert adapter_spec("JSONSCHEMABENCH")["mapping"] == "whole_json_schema"
    database = output / "prepared/workload.sqlite"
    refs = list(prepared_rows(database))
    assert len(refs) == 4 and len({item["record_id"] for item in refs}) == 4
    for index, ref in enumerate(refs):
        record = resolve_input(database, ref["record_id"], data_root=data)
        assert record.task["task_id"] == rows[index % 2]["unique_id"]
        assert record.task["input"] == {
            "json_schema": rows[index % 2]["json_schema"],
            "feature_inputs": {},
        }
        assert record.outcome is None
        assert record.source["file"] == files[index // 2]["path"]
    with pytest.raises(ValueError, match="fresh output"):
        runner.prepare(data_root=data, output=output)
    with pytest.raises(ValueError, match="registered"):
        runner.prepare(
            data_root=data, output=tmp_path / "other", files=["unknown.parquet"]
        )


def test_official_mapping_does_not_create_prompts_answers_or_feature_views():
    from vllm_hust_workload_data.mapping_contract import load_mapping

    mapping = load_mapping(runner.MAPPING)
    assert set(mapping["input"]) == {"json_schema"}
    assert mapping["identity"]["task_id"]["path"] == ["unique_id"]
    assert mapping["features"] == mapping["outcome"] == {}
