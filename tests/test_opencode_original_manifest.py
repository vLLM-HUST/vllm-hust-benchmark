"""Original publisher code corpus inventory, not a fabricated benchmark."""

import importlib.util
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from vllm_hust_workload_data.registry import source_spec


def test_opencode_pinned_inventory_is_acquirable_without_model_calls():
    root = Path(__file__).resolve().parents[1]
    spec = source_spec("OPENCODE")
    assert len(spec["files"]) == 50
    assert sum(item["bytes"] for item in spec["files"]) == 6861113102
    assert spec["access"]["gated"] is False
    loader = importlib.util.spec_from_file_location(
        "opencode_asset_preparer", root / "scripts/prepare_reused_research_assets.py"
    )
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    paths = [f"data/train-{index:05d}-of-00050.parquet" for index in (0, 24, 49)]
    _, selected = module.selection("OPENCODE", paths)
    assert [item["path"] for item in selected] == paths
    assert sum(item["bytes"] for item in selected) == 422115928
    assert all(item["role"] == "data" for item in selected)


def test_opencode_prepare_preserves_all_original_fields_without_execution(tmp_path):
    from vllm_hust_workload_data.consumer import resolve_input

    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root = Path(__file__).resolve().parents[1]
    loader = importlib.util.spec_from_file_location(
        "opencode_records_preparer", root / "scripts/prepare_reused_source_records.py"
    )
    runner = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(runner)
    original = {
        "id": "fixture-original-id",
        "input": "fixture coding question",
        "output": "fixture teacher code",
        "domain": "generic",
        "generation_algorithm": "self-instruct",
        "llm_judgement": "{}",
        "unit_tests": "['fixture assertion text; never executed']",
        "tests_execution_status": "[]",
        "average_test_score": 0.5,
    }
    filename = "data/train-00000-of-00050.parquet"
    data = tmp_path / "original"
    path = data / filename
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([original]), path)
    raw = path.read_bytes()
    catalog = tmp_path / "catalog"
    (catalog / "data/catalogs").mkdir(parents=True)
    for name in (
        "source_manifests_v1.json",
        "adapter_registry_v1.json",
        "adapter_extensions_v1.json",
    ):
        payload = json.loads(
            (root / "workload-preparation/data/catalogs" / name).read_text("utf-8")
        )
        if name == "source_manifests_v1.json":
            payload["sources"]["OPENCODE"]["files"] = [
                {
                    "path": filename,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw),
                    "rows": 1,
                    "role": "data",
                }
            ]
        (catalog / "data/catalogs" / name).write_text(json.dumps(payload), "utf-8")
    output = tmp_path / "prepared"
    report = runner.prepare(
        "OPENCODE", data_root=data, output=output, catalog_root=catalog
    )
    database = output / "prepared/workload.sqlite"
    with sqlite3.connect(database) as connection:
        (record_id,) = connection.execute("SELECT record_id FROM records").fetchone()
    record = resolve_input(database, record_id, data_root=data)
    assert record.task["input"] == {"input": original["input"]}
    assert {
        **record.task["input"],
        **record.outcome_record(),
        **record.replay["annotations"],
    } == original
    assert path.read_bytes() == raw
    assert not report["native_execution_complete"]


def test_opencode_delivery_receipt_is_bounded_to_original_corpus_profile():
    root = Path(__file__).resolve().parents[1]
    directory = root / "docs/workload-data"
    item = json.loads(
        (directory / "dataset-validation-status.json").read_text("utf-8")
    )["delivery_inventory"]["OPENCODE"]
    receipt = json.loads((root / item["delivery_receipt"]).read_text("utf-8"))
    stage = json.loads(
        (directory / "validation/opencode-original-readiness-20261028.json").read_text(
            "utf-8"
        )
    )
    assert item["checked"] and receipt["source_checkbox_eligible"]
    assert not stage["source_checkbox_eligible"]
    assert receipt["acquired_original_shards"] == [0, 24, 49]
    assert receipt["original_rows_fully_equal_and_indexed"] == 300000
    assert receipt["original_fields_preserved"] == 9
    assert receipt["native_profile"]["original_cases"] == 9
    assert receipt["native_profile"]["stop_count"] == 7
    assert receipt["native_profile"]["length_count"] == 2
    assert receipt["provider_and_research"]["installed_payloads_byte_equal"] == 57
    assert (
        receipt["provider_and_research"]["fixed_current_public_and_research_readbacks"]
        == 9
    )
    assert not receipt["host_execution_of_original_tests_or_generated_code"]
    assert not receipt["quality_scored"]
    assert not receipt["independent_quality_benchmark_claim"]
    assert not receipt["whole_50_shard_execution_claim"]
