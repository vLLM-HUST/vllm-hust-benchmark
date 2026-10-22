"""Invented fixtures check indexing, never represent a real task run."""

import json
import sqlite3

import pytest

from vllm_hust_workload_data.bindings import bind_source
from vllm_hust_workload_data.consumer import resolve_input
from vllm_hust_workload_data.pipeline import plan, prepared_rows, run


def config(tmp_path, malformed=False):
    root = tmp_path / "raw"
    root.mkdir()
    rows = [{"question": "INPUT_SENTINEL", "answer": "ANSWER_SENTINEL"}]
    if malformed:
        rows.append({"answer": "missing required question"})
    (root / "fixture.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    manifest = bind_source(
        "GSM8K", data_root=root, files=["fixture.jsonl"], revision="software-fixture"
    )
    assert manifest["upstream_revision_verified"] is False
    binding = tmp_path / "manifest.json"
    binding.write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "sources": [
            {"source_id": "GSM8K", "data_root": str(root), "manifest": str(binding)}
        ]
    }, root


def test_index_contains_references_not_payload_or_research_tables(tmp_path):
    cfg, root = config(tmp_path)
    output = tmp_path / "prepared"
    report = run(cfg, output=output)
    assert report["model_calls"] == 0
    assert report["scoring"] == "NOT_RUN"
    database = output / "workload.sqlite"
    rows = list(prepared_rows(database))
    assert len(rows) == 1
    with sqlite3.connect(database) as connection:
        assert {
            r[0]
            for r in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        } == {"sources", "records"}
        contents = "\n".join(connection.iterdump())
    assert "INPUT_SENTINEL" not in contents
    assert "ANSWER_SENTINEL" not in contents
    record = resolve_input(database, rows[0]["record_id"], data_root=root)
    assert "INPUT_SENTINEL" in json.dumps(record.input_record())
    assert "ANSWER_SENTINEL" not in json.dumps(record.input_record())
    assert record.outcome_record()["answer"] == "ANSWER_SENTINEL"
    assert record.replay["selection"] == {"files": ["fixture.jsonl"], "limit": None}
    (root / "fixture.jsonl").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        resolve_input(database, rows[0]["record_id"], data_root=root)
    with pytest.raises(FileExistsError):
        run(cfg, output=output)


def test_native_tasks_never_fall_back_to_qbi(tmp_path):
    for sid in ("GAIA", "AGENTBENCH", "TAUBENCH"):
        result = plan({"sources": [{"source_id": sid, "data_root": str(tmp_path)}]})
        assert (
            result["sources"][0]["blocking_reason"]
            == "OFFICIAL_RUNNER_ADAPTER_REQUIRED_NO_QBI_FALLBACK"
        )
    with pytest.raises(ValueError):
        plan({"sources": [{"source_id": "GAIA", "qbi_root": str(tmp_path)}]})


def test_failed_source_does_not_publish_complete_index(tmp_path):
    cfg, root = config(tmp_path)
    binding_path = tmp_path / "manifest.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    # Corruption after binding must not publish a complete database.
    (root / "fixture.jsonl").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        run(cfg, output=tmp_path / "failed")
    assert not (tmp_path / "failed/workload.sqlite").exists()
    assert not (tmp_path / "failed/manifest.json").exists()
    report = run(cfg, output=tmp_path / "partial", allow_partial=True)
    assert report["status"] == "PARTIAL"
    assert report["sources"][0]["status"] == "FAILED"
    assert list(prepared_rows(tmp_path / "partial/workload.sqlite")) == []
    assert binding["source_id"] == "GSM8K"
