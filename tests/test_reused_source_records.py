"""Tiny software fixtures only: no publisher data or native benchmark claims."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "reuse_records", ROOT / "scripts/prepare_reused_source_records.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture(params=["LONGALPACA", "NATURALQUESTIONS"])
def original_fixture(request, tmp_path):
    sid = request.param
    row = (
        {
            "instruction": "fixture instruction",
            "input": "fixture context",
            "output": "fixture answer",
        }
        if sid == "LONGALPACA"
        else {"question": "fixture question", "answer": ["fixture answer", "alternate"]}
    )
    name = "fixture.json" if sid == "LONGALPACA" else "fixture.jsonl"
    raw = json.dumps([row] if sid == "LONGALPACA" else row).encode() + b"\n"
    data = tmp_path / "data"
    data.mkdir()
    (data / name).write_bytes(raw)
    catalog = tmp_path / "catalog"
    (catalog / "data/catalogs").mkdir(parents=True)
    source = json.loads(
        (
            ROOT / "workload-preparation/data/catalogs/source_manifests_v1.json"
        ).read_text("utf-8")
    )
    source["sources"][sid]["files"] = [
        {
            "path": name,
            "role": "raw_source",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
    ]
    (catalog / "data/catalogs/source_manifests_v1.json").write_text(
        json.dumps(source), encoding="utf-8"
    )
    return sid, row, name, data, catalog


def test_prepare_and_resolve_preserve_fields_without_copying_data(
    original_fixture, tmp_path
):
    from vllm_hust_workload_data.consumer import resolve_input

    sid, row, name, data, catalog = original_fixture
    output = tmp_path / "prepared"
    receipt = runner.prepare(
        sid, data_root=data, output=output, catalog_root=catalog, limit=1
    )
    assert receipt["source_rows_fully_bound"] == 1
    assert receipt["preparation"]["status"] == "PREPARED_REFERENCES_ONLY"
    assert not receipt["native_execution_complete"]
    assert not receipt["source_checkbox_eligible"]
    database = output / "prepared/workload.sqlite"
    with sqlite3.connect(database) as connection:
        record_id = connection.execute("SELECT record_id FROM records").fetchone()[0]
    record = resolve_input(database, record_id, data_root=data)
    assert b"fixture answer" not in database.read_bytes()
    assert not (output / name).exists()
    if sid == "LONGALPACA":
        assert record.task["input"]["instruction"] == row["instruction"]
        assert record.outcome["reference_output"] == row["output"]
    else:
        assert record.task["input"]["question"] == row["question"]
        assert record.outcome["answers"] == row["answer"]
    assert (
        "answers" not in record.task["input"]
        and "reference_output" not in record.task["input"]
    )
    with pytest.raises(ValueError, match="fresh output"):
        runner.prepare(sid, data_root=data, output=output, catalog_root=catalog)


def test_unknown_files_and_changed_identity_fail_closed(original_fixture, tmp_path):
    sid, _, name, data, catalog = original_fixture
    with pytest.raises(ValueError, match="registered"):
        runner.prepare(
            sid,
            data_root=data,
            output=tmp_path / "out",
            files=["unknown"],
            catalog_root=catalog,
        )
    (data / name).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from pinned"):
        runner.prepare(
            sid, data_root=data, output=tmp_path / "out", catalog_root=catalog
        )
    assert not (tmp_path / "out").exists()


def test_data_only_profiles_do_not_imply_nq_retrieval():
    profiles = {
        sid: json.loads(path.read_text("utf-8"))
        for sid, path in runner.MAPPINGS.items()
    }
    assert all(profile["features"] == {} for profile in profiles.values())
    assert profiles["NATURALQUESTIONS"]["families"] == ["general-single-turn-assistant"]


def test_revision_drift_never_downgrades_to_unpinned_local_snapshot(
    original_fixture, tmp_path
):
    sid, _, _, data, catalog = original_fixture
    path = catalog / "data/catalogs/source_manifests_v1.json"
    manifest = json.loads(path.read_text("utf-8"))
    manifest["sources"][sid]["revision"] = "different-revision"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="re-review required"):
        runner.prepare(
            sid, data_root=data, output=tmp_path / "out", catalog_root=catalog
        )
    assert not (tmp_path / "out").exists()
