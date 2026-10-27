"""Small invented integrity fixtures, not workload data or native validation."""

from copy import deepcopy
import gzip
import hashlib
import io
import importlib.util
import json
from pathlib import Path
import tarfile

import pytest

from vllm_hust_workload_data.copilot import (
    extract_verified_archive,
    qualify_day_manifest,
    verify_original_directory,
    verify_session,
)
from vllm_hust_workload_data.adapters import adapt_row
from vllm_hust_workload_data.registry import adapter_spec, source_spec


def test_registered_release_uses_original_all_shards_not_requests():
    source = source_spec("COPILOT_AGENT_2026")
    spec = adapter_spec("COPILOT_AGENT_2026")
    assert source["revision"] == "ghcp-coding-agent-2026"
    assert source["release_id"] == 400522856
    assert len(source["files"]) == 606
    assert sum(item["rows"] for item in source["files"]) == 301026
    assert len({item["path"].split("/")[0] for item in source["files"]}) == 7
    assert all(item["role"] == "raw_source" for item in source["files"])
    assert all(item["path"].endswith(".jsonl.gz") for item in source["files"])
    assert spec["kind"] == "request_metadata"
    assert spec["input_fields"] == spec["outcome_fields"] == []
    assert spec["id_field"] == ""


def metadata_spec():
    return {
        "kind": "request_metadata",
        "mapping": "copilot_original_session_metadata",
        "required_fields": ["session_id", "date", "turns"],
        "input_fields": [],
        "outcome_fields": [],
        "annotation_fields": ["session_id", "date", "turns"],
        "id_field": "",
        "requirements": ["Original session metadata only; no prompt or replay"],
    }


def test_adapter_preserves_nested_relations_without_requests_or_dedup():
    row = {
        "session_id": "same-session-label",
        "date": "2026-06-01",
        "turns": [
            {
                "turn_id": "t1",
                "llm_calls": [
                    {"message_id": "m1", "model": "Model A", "tokens": {"cached": 4}}
                ],
                "tool_batches": [
                    {
                        "request_id": "m1",
                        "function_calls": [{"name": "edit", "status": 2}],
                    }
                ],
            }
        ],
    }
    original = deepcopy(row)
    records = [
        adapt_row(
            "COPILOT_AGENT_2026",
            row,
            source={"source_id": "COPILOT_AGENT_2026", "file_sha256": "fixture-only"},
            spec=metadata_spec(),
            index=index,
        )
        for index in (0, 1)
    ]
    assert records[0].task["task_id"] != records[1].task["task_id"]
    assert records[0].task["identity_kind"] == "file_sha256_row_index"
    assert records[0].replay["annotations"] == original
    assert records[0].replay["unmapped_field_names"] == []
    assert records[0].task["input"] == {}
    assert records[0].outcome_record() is None
    with pytest.raises(ValueError, match="not a ready model request"):
        records[0].request_body()
    records[0].replay["annotations"]["turns"].clear()
    assert row == original
    assert records[1].replay["annotations"] == original


@pytest.mark.parametrize("change", ["kind", "input", "outcome", "annotation", "id"])
def test_adapter_rejects_task_projection_or_session_only_identity(change):
    row = {"session_id": "fixture", "date": "2026-06-01", "turns": []}
    spec = metadata_spec()
    if change == "kind":
        spec["kind"] = "request_replay"
    elif change == "input":
        spec["input_fields"] = ["turns"]
    elif change == "outcome":
        spec["outcome_fields"] = ["turns"]
    elif change == "annotation":
        spec["annotation_fields"] = ["session_id"]
    else:
        spec["id_field"] = "session_id"
    with pytest.raises(ValueError, match="unchanged nested metadata"):
        adapt_row("COPILOT_AGENT_2026", row, source={}, spec=spec, index=0)


def archive_fixture(tmp_path, *, unsafe=None, manifest_first=False):
    row = {"session_id": "fixture", "date": "2026-06-01", "turns": []}
    payload = gzip.compress((json.dumps(row) + "\n").encode())
    manifest = {
        "complete": True,
        "date": row["date"],
        "n_sessions": 1,
        "n_shards": 1,
        "total_bytes": len(payload),
        "shards": [{"shard_idx": 0, "n_sessions": 1, "size_bytes": len(payload)}],
    }
    entries = [
        ("date=2026-06-01/shard-0000.jsonl.gz", payload),
        ("date=2026-06-01/manifest.json", json.dumps(manifest).encode()),
    ]
    if manifest_first:
        entries.reverse()
    if unsafe:
        entries.insert(0, (unsafe, b"not original data"))
    archive = tmp_path / "original.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for name, data in entries:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))
    return archive, payload


def extract(archive, output, **kwargs):
    return extract_verified_archive(
        archive,
        output,
        expected_bytes=archive.stat().st_size,
        expected_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        **kwargs,
    )


@pytest.mark.parametrize("manifest_first", [False, True])
def test_extract_unchanged_gzip_without_order_assumptions(tmp_path, manifest_first):
    archive, payload = archive_fixture(tmp_path, manifest_first=manifest_first)
    result = extract(archive, tmp_path / "data")
    assert result["qualification"]["sessions"] == 1
    assert result["model_calls"] == 0
    target = tmp_path / "data/date=2026-06-01/shard-0000.jsonl.gz"
    assert target.read_bytes() == payload
    with pytest.raises(ValueError, match="fresh"):
        extract(archive, tmp_path / "data")


def test_existing_original_directory_verified_without_writes(tmp_path):
    archive, _ = archive_fixture(tmp_path)
    initial = extract(archive, tmp_path / "data")
    before = {p: p.stat().st_mtime_ns for p in (tmp_path / "data").rglob("*")}
    verified = verify_original_directory(
        tmp_path / "data", expected_days=["2026-06-01"]
    )
    assert verified["files"] == initial["files"]
    assert verified["days"] == [initial["qualification"]]
    assert before == {p: p.stat().st_mtime_ns for p in (tmp_path / "data").rglob("*")}


@pytest.mark.parametrize(
    "change", ["missing_day", "extra_file", "wrong_rows", "wrong_date"]
)
def test_existing_original_directory_rejects_partial_or_modified_data(tmp_path, change):
    archive, _ = archive_fixture(tmp_path)
    extract(archive, tmp_path / "data")
    day = tmp_path / "data/date=2026-06-01"
    if change == "extra_file":
        (day / "analysis.py").write_text("not an original asset")
    elif change == "wrong_rows":
        manifest = json.loads((day / "manifest.json").read_text())
        manifest["shards"][0]["n_sessions"] += 1
        (day / "manifest.json").write_text(json.dumps(manifest))
    elif change == "wrong_date":
        row = {"session_id": "fixture", "date": "2026-06-02", "turns": []}
        (day / "shard-0000.jsonl.gz").write_bytes(
            gzip.compress(json.dumps(row).encode())
        )
    days = ["2026-06-01", "2026-06-02"] if change == "missing_day" else ["2026-06-01"]
    with pytest.raises(ValueError):
        verify_original_directory(tmp_path / "data", expected_days=days)


@pytest.mark.parametrize(
    "name", ["../escape", "/escape", "date=x/analysis.py", "date=x/a\\b"]
)
def test_extract_rejects_unsafe_or_non_asset_members(tmp_path, name):
    archive, _ = archive_fixture(tmp_path, unsafe=name)
    with pytest.raises(ValueError):
        extract(archive, tmp_path / "data")
    assert not (tmp_path / "escape").exists()


def test_bad_archive_identity_does_not_create_output(tmp_path):
    archive, _ = archive_fixture(tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        extract_verified_archive(
            archive,
            tmp_path / "data",
            expected_bytes=archive.stat().st_size,
            expected_sha256="0" * 64,
        )
    assert not (tmp_path / "data").exists()


def preparer_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/prepare_copilot_original_assets.py"
    )
    spec = importlib.util.spec_from_file_location("copilot_assets_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_preparer_plan_is_non_mutating(tmp_path):
    preparer = preparer_module()
    output = tmp_path / "not_created"
    result = preparer.prepare(output)
    assert result["status"] == "PLAN_ONLY"
    assert result["archive_bytes"] == 3218886136
    assert result["archives"] == 7
    assert all(len(item[3]) == 64 for item in preparer.ARCHIVES)
    assert not output.exists()


def test_public_preparer_reuses_original_archive_without_network(tmp_path, monkeypatch):
    preparer = preparer_module()
    archive, payload = archive_fixture(tmp_path)
    reuse = tmp_path / "reused"
    reuse.mkdir()
    archive.rename(reuse / "date.2026-06-01.tar.gz")
    original = reuse / "date.2026-06-01.tar.gz"
    monkeypatch.setattr(
        preparer,
        "ARCHIVES",
        [
            (
                "2026-06-01",
                1,
                original.stat().st_size,
                hashlib.sha256(original.read_bytes()).hexdigest(),
            )
        ],
    )

    def no_network(*args, **kwargs):
        raise AssertionError("must reuse original archive")

    monkeypatch.setattr(preparer.subprocess, "run", no_network)
    result = preparer.prepare(
        tmp_path / "prepared", execute=True, reuse_archive_dir=reuse
    )
    assert result["status"] == "ORIGINAL_RELEASE_ASSETS_PREPARED_NOT_PROVIDER_DELIVERY"
    assert not result["source_checkbox_eligible"]
    assert result["days"][0]["sessions"] == 1
    target = tmp_path / "prepared/data/date=2026-06-01/shard-0000.jsonl.gz"
    assert target.read_bytes() == payload
    assert original.exists()


def test_nested_fields_remain_untouched_and_bad_containers_rejected():
    row = {
        "session_id": "fixture",
        "date": "2026-06-01",
        "turns": [
            {
                "turn_id": "t",
                "llm_calls": [{"unknown_extension": [1, None]}],
                "tool_batches": [{"function_calls": [{"status": 2}]}],
            }
        ],
    }
    before = deepcopy(row)
    verify_session(row)
    assert row == before
    row["turns"][0]["tool_batches"][0]["function_calls"] = ["not an object"]
    with pytest.raises(ValueError, match="payload suppressed"):
        verify_session(row)


def fixture():
    manifest = {
        "complete": True,
        "date": "2026-06-01",
        "n_sessions": 3,
        "n_shards": 2,
        "total_bytes": 30,
        "shards": [
            {"shard_idx": 0, "n_sessions": 2, "size_bytes": 10},
            {"shard_idx": 1, "n_sessions": 1, "size_bytes": 20},
        ],
    }
    members = [
        {"path": "date=2026-06-01/shard-0000.jsonl.gz", "bytes": 10, "rows": 2},
        {"path": "date=2026-06-01/shard-0001.jsonl.gz", "bytes": 20, "rows": 1},
    ]
    return manifest, members


def test_complete_day_order_independent_and_no_mutation():
    manifest, members = fixture()
    before = deepcopy((manifest, members))
    result = qualify_day_manifest(manifest, members[::-1])
    assert result == {
        "date": "2026-06-01",
        "shards": 2,
        "sessions": 3,
        "compressed_shard_bytes": 30,
    }
    assert (manifest, members) == before


@pytest.mark.parametrize(
    "change", ["missing", "extra", "duplicate", "rows", "bytes", "bool"]
)
def test_reject_incomplete_or_modified_original_members(change):
    manifest, members = fixture()
    if change == "missing":
        members.pop()
    elif change == "extra":
        members.append({"path": "../shard-0002.jsonl.gz", "bytes": 0, "rows": 0})
    elif change == "duplicate":
        members.append(deepcopy(members[0]))
    elif change == "bool":
        members[0]["rows"] = True
    else:
        members[0][change] += 1
    with pytest.raises(ValueError):
        qualify_day_manifest(manifest, members)


@pytest.mark.parametrize("change", ["incomplete", "date", "total", "duplicate", "bool"])
def test_reject_inconsistent_publisher_manifest(change):
    manifest, members = fixture()
    if change == "incomplete":
        manifest["complete"] = False
    elif change == "date":
        manifest["date"] = "2026-6-1"
    elif change == "total":
        manifest["total_bytes"] += 1
    elif change == "duplicate":
        manifest["shards"][1] = deepcopy(manifest["shards"][0])
    else:
        manifest["shards"][0]["shard_idx"] = False
    with pytest.raises(ValueError):
        qualify_day_manifest(manifest, members)
