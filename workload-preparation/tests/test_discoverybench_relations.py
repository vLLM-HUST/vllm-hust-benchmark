"""Inert original-schema fixtures; no real tasks, Agent or neural execution."""

import csv
import hashlib
import io
import json

import pytest

from vllm_hust_workload_data.discoverybench import (
    KEY,
    adapt_case,
    original_rows,
    restore_original_case,
)


def fixture(tmp_path):
    meta = {
        "id": 0,
        "domain": "fixture",
        "domain_knowledge": "original hint",
        "workflow_tags": "original tags",
        "datasets": [
            {"name": "data.csv", "description": "fixture", "columns": {"raw": []}}
        ],
        "hypotheses": {
            "main": "reference-main-marker",
            "intermediate": [{"text": "original intermediate hint"}],
        },
        "intermediate": ["original annotation"],
        "queries": [
            [{"qid": 0, "question_type": "fixture", "question": "original question"}]
        ],
    }
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=["dataset", "metadataid", "query_id", "gold_hypo"]
    )
    writer.writeheader()
    for index in (0, 1):
        writer.writerow(
            {
                "dataset": "fixture",
                "metadataid": str(index),
                "query_id": "0",
                "gold_hypo": "reference-key-marker",
            }
        )
    values = {
        KEY: stream.getvalue().encode(),
        "discoverybench/real/test/fixture/metadata_0.json": json.dumps(meta).encode(),
        "discoverybench/real/test/fixture/metadata_1.json": b"\xff original upstream bad encoding",
        "discoverybench/real/test/fixture/data.csv": b"x\n1\n",
    }
    files = []
    for name, raw in values.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        files.append(
            {
                "path": name,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "role": "data" if name == KEY else "native_asset",
            }
        )
    return meta, {"relation_files": files}


def test_original_order_reference_separation_and_exact_metadata_restore(tmp_path):
    meta, spec = fixture(tmp_path)
    rows = list(original_rows(tmp_path / KEY, spec))
    assert len(rows) == 2
    record = adapt_case(rows[0], source={"source_id": "DISCOVERYBENCH"}, index=0)
    assert record.task["task_id"] == "fixture/0/0"
    assert "reference-main-marker" not in json.dumps(record.input_record())
    assert "reference-key-marker" not in json.dumps(record.input_record())
    assert (
        restore_original_case(record, data_root=tmp_path, spec=spec)["metadata"] == meta
    )
    assert record.outcome == {"references": {"gold_hypo": "reference-key-marker"}}
    with pytest.raises(ValueError, match="not a ready model request"):
        record.request_body()


def test_original_decode_failure_kept_as_blocked_not_filtered_or_guessed(tmp_path):
    _, spec = fixture(tmp_path)
    rows = list(original_rows(tmp_path / KEY, spec))
    record = adapt_case(rows[1], source={"source_id": "DISCOVERYBENCH"}, index=1)
    assert record.task["task_id"] == "fixture/1/0"
    assert record.task["input"]["original_query"] is None
    assert record.runtime["blockers"] == ["ORIGINAL_METADATA_DECODE_FAILED"]
    assert record.outcome["references"]["gold_hypo"] == "reference-key-marker"
    with pytest.raises(ValueError, match="publisher task blocked"):
        restore_original_case(record, data_root=tmp_path, spec=spec)


def test_changed_original_file_fails_identity_not_blocked_record(tmp_path):
    _, spec = fixture(tmp_path)
    (tmp_path / "discoverybench/real/test/fixture/data.csv").write_text("changed")
    with pytest.raises(ValueError, match="identity mismatch"):
        list(original_rows(tmp_path / KEY, spec))


def test_original_missing_filename_not_repaired_or_aliased(tmp_path):
    meta, spec = fixture(tmp_path)
    meta["datasets"][0]["name"] = "does-not-exist.csv"
    name = "discoverybench/real/test/fixture/metadata_0.json"
    raw = json.dumps(meta).encode()
    (tmp_path / name).write_bytes(raw)
    entry = next(item for item in spec["relation_files"] if item["path"] == name)
    entry.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    rows = list(original_rows(tmp_path / KEY, spec))
    assert rows[0]["original_query"]["question"] == "original question"
    assert rows[0]["blockers"] == [
        "ORIGINAL_DATA_REFERENCE_UNRESOLVED:discoverybench/real/test/fixture/does-not-exist.csv"
    ]
    assert not (
        tmp_path / "discoverybench/real/test/fixture/does-not-exist.csv"
    ).exists()


def test_changed_reference_rejected_on_restore(tmp_path):
    _, spec = fixture(tmp_path)
    rows = list(original_rows(tmp_path / KEY, spec))
    record = adapt_case(rows[0], source={"source_id": "DISCOVERYBENCH"}, index=0)
    record.outcome["references"]["gold_hypo"] = "changed reference"
    with pytest.raises(ValueError, match="relation changed"):
        restore_original_case(record, data_root=tmp_path, spec=spec)
