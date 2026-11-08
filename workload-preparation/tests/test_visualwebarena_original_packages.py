"""Invented multi-split controls, not actual browser or original task evidence."""

from contextlib import closing
import hashlib
import json

import pytest

from vllm_hust_workload_data import iter_workload
from vllm_hust_workload_data.consumer import resolve_input
from vllm_hust_workload_data.pipeline import prepared_rows, run
from vllm_hust_workload_data.registry import adapter_spec
from vllm_hust_workload_data.webarena import SOURCE_TASKS, original_task_packages


def fixture(root):
    originals = {}
    payloads = {"LICENSE": b"invented license", "README.md": b"invented README"}
    for name in SOURCE_TASKS["VISUALWEBARENA"]:
        split = name.split("test_", 1)[1].removesuffix(".raw.json")
        rows = [
            {
                "task_id": i,
                "sites": [split],
                "intent": "invented fixture only",
                "image": f"environment_docker/input_images/{split}/original.png",
                "start_url": f"__{split.upper()}__/original",
                "require_reset": False,
                "storage_state": f".auth/{split}.json",
                "unknown_original_field": [None, "", i],
                "eval": {
                    "eval_types": ["url_match"],
                    "reference_url": f"gold-{split}-{i}",
                },
            }
            for i in (0, 1)
        ]
        originals[split] = rows
        payloads[name] = json.dumps(rows).encode()
    entries = []
    for name, raw in payloads.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        entries.append(
            dict(
                path=name,
                bytes=len(raw),
                role="raw_source",
                sha256=hashlib.sha256(raw).hexdigest(),
            )
        )
    return originals, dict(
        source_id="VISUALWEBARENA",
        revision=adapter_spec("VISUALWEBARENA")["code_revision"],
        files=entries,
    )


def test_original_split_ids_images_and_eval_remain_lossless(tmp_path):
    originals, manifest = fixture(tmp_path)
    records = list(
        iter_workload("VISUALWEBARENA", data_root=tmp_path, manifest=manifest)
    )
    assert len(records) == len({r.task["task_id"] for r in records}) == 6
    assert [r.task["task_id"] for r in records] == [
        f"{split}:{i}" for split in originals for i in (0, 1)
    ]
    for record in records:
        split = record.task["original_split"]
        original_id = record.task["input"]["original_task"]["task_id"]
        rebuilt = {
            **record.task["input"]["original_task"],
            "eval": record.outcome_record()["original_eval"],
        }
        assert rebuilt == originals[split][original_id]
        assert "gold-" not in json.dumps(record.input_record())
        assert record.source["source_id"] == "VISUALWEBARENA"
        assert record.runtime["execution"] == "not_executed"
        with pytest.raises(ValueError, match="not a ready model request"):
            record.request_body()


def test_three_original_assets_are_not_indexed_three_times(tmp_path):
    root = tmp_path / "original"
    originals, manifest = fixture(root)
    pins = tmp_path / "fixture.json"
    pins.write_text(json.dumps(manifest))
    output = tmp_path / "prepared"
    run(
        {
            "sources": [
                {
                    "source_id": "VISUALWEBARENA",
                    "data_root": str(root),
                    "manifest": str(pins),
                }
            ]
        },
        output=output,
    )
    rows = list(prepared_rows(output / "workload.sqlite"))
    assert len(rows) == 6
    ids = set()
    for row in rows:
        restored = resolve_input(
            output / "workload.sqlite", row["record_id"], data_root=root
        )
        ids.add(restored.task["task_id"])
        original = restored.task["input"]["original_task"]
        assert {
            **original,
            "eval": restored.outcome_record()["original_eval"],
        } == originals[restored.task["original_split"]][original["task_id"]]
    assert len(ids) == 6


def test_partial_iterator_closure_checks_actual_selected_asset(tmp_path):
    _, manifest = fixture(tmp_path)
    with pytest.raises(ValueError, match="identity mismatch"):
        with closing(original_task_packages(tmp_path, manifest=manifest)) as records:
            next(records)
            (tmp_path / SOURCE_TASKS["VISUALWEBARENA"][0]).write_text("changed")


def test_all_three_original_splits_required(tmp_path):
    _, manifest = fixture(tmp_path)
    manifest["files"] = [
        r for r in manifest["files"] if r["path"] != SOURCE_TASKS["VISUALWEBARENA"][2]
    ]
    with pytest.raises(ValueError, match="pinned original"):
        list(iter_workload("VISUALWEBARENA", data_root=tmp_path, manifest=manifest))
