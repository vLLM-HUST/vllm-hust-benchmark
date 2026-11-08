"""Invented contract fixtures, not real WebArena execution or data evidence."""

from contextlib import closing
from copy import deepcopy
import hashlib
import json

import pytest

from vllm_hust_workload_data import iter_workload
from vllm_hust_workload_data.registry import adapter_spec
from vllm_hust_workload_data.pipeline import prepared_rows, run
from vllm_hust_workload_data.webarena import original_task_packages


def fixture(root, *, duplicate=False):
    rows = [
        {
            "task_id": i,
            "intent": "invented fixture only",
            "sites": ["shopping_admin"],
            "start_url": "__SHOPPING_ADMIN__/admin",
            "require_login": True,
            "storage_state": ".auth/shopping_admin_state.json",
            "require_reset": False,
            "publisher_extra": {"blank": "", "nested": [None, i]},
            "eval": {"reference_answers": {"exact_match": "fixture-gold" + str(i)}},
        }
        for i in range(2)
    ]
    if duplicate:
        rows[1]["task_id"] = 0
    payloads = {
        "LICENSE": b"invented license fixture",
        "README.md": b"invented README fixture",
        "config_files/test.raw.json": json.dumps(rows).encode(),
    }
    files = []
    for name, raw in payloads.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        files.append(
            {
                "path": name,
                "role": "raw_source",
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return rows, {
        "source_id": "WEBARENA",
        "revision": adapter_spec("WEBARENA")["code_revision"],
        "unit": "source_rows",
        "files": files,
    }


def test_unified_reader_preserves_all_fields_and_separates_evaluator(tmp_path):
    rows, manifest = fixture(tmp_path)
    records = list(iter_workload("WEBARENA", data_root=tmp_path, manifest=manifest))
    assert len(records) == len(rows)
    for record, raw in zip(records, rows, strict=True):
        source_input = record.input_record()
        reference = record.outcome_record()
        reconstructed = {
            **record.task["input"]["original_task"],
            "eval": reference["original_eval"],
        }
        assert reconstructed == raw
        assert "fixture-gold" not in json.dumps(source_input)
        assert record.runtime["execution"] == "not_executed"
        assert not record.replay["synthetic_fallback"]
        with pytest.raises(ValueError, match="not a ready model request"):
            record.request_body()


def test_changed_asset_rejected_before_reading(tmp_path):
    _, manifest = fixture(tmp_path)
    (tmp_path / "config_files/test.raw.json").write_text("[]")
    with pytest.raises(ValueError, match="identity mismatch"):
        list(original_task_packages(tmp_path, manifest=manifest))


def test_duplicate_original_ids_rejected(tmp_path):
    _, manifest = fixture(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="distinct original task IDs"):
        list(original_task_packages(tmp_path, manifest=manifest))


def test_mutation_detected_when_partial_iterator_closes(tmp_path):
    _, manifest = fixture(tmp_path)
    with pytest.raises(ValueError, match="identity mismatch"):
        with closing(original_task_packages(tmp_path, manifest=manifest)) as records:
            next(records)
            (tmp_path / "config_files/test.raw.json").write_text("changed")


def test_revision_mismatch_rejected(tmp_path):
    _, manifest = fixture(tmp_path)
    manifest = deepcopy(manifest)
    manifest["revision"] = "invented-other-revision"
    with pytest.raises(ValueError, match="revision mismatch"):
        list(iter_workload("WEBARENA", data_root=tmp_path, manifest=manifest))


def test_duplicate_asset_bindings_rejected(tmp_path):
    _, manifest = fixture(tmp_path)
    manifest["files"].append(deepcopy(manifest["files"][0]))
    with pytest.raises(ValueError, match="distinct original WebArena asset"):
        list(original_task_packages(tmp_path, manifest=manifest))


def test_preparation_index_and_resolver_preserve_original_task(tmp_path):
    from vllm_hust_workload_data.consumer import resolve_input

    original_root = tmp_path / "original"
    rows, manifest = fixture(original_root)
    path = tmp_path / "fixture-manifest.json"
    path.write_text(json.dumps(manifest))
    output = tmp_path / "prepared"
    report = run(
        {
            "sources": [
                {
                    "source_id": "WEBARENA",
                    "data_root": str(original_root),
                    "manifest": str(path),
                }
            ]
        },
        output=output,
    )
    assert report["model_calls"] == 0
    assert report["scoring"] == "NOT_RUN"
    database = output / "workload.sqlite"
    index = list(prepared_rows(database))
    assert len(index) == len(rows)
    for item, original in zip(index, rows, strict=True):
        record = resolve_input(database, item["record_id"], data_root=original_root)
        restored = {
            **record.task["input"]["original_task"],
            "eval": record.outcome_record()["original_eval"],
        }
        assert restored == original
