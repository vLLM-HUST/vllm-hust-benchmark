"""Software-only original-asset reference checks; no official tasks or model."""

import hashlib
import json
import zipfile

import pytest

from vllm_hust_workload_data import provider


def test_native_references_do_not_publish_task_or_scorer_payload(tmp_path, monkeypatch):
    data = tmp_path / "data"
    relative = "tau2/domains/telecom/tasks_full.json"
    path = data / relative
    path.parent.mkdir(parents=True)
    raw = json.dumps(
        [
            {
                "id": "fixture-id",
                "user_scenario": "private fixture",
                "evaluation_criteria": {"answer": "private fixture"},
            }
        ]
    ).encode()
    path.write_bytes(raw)
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("fixture/data/" + relative, raw)
    monkeypatch.setattr(
        provider,
        "TAU2_ORIGINAL_ARCHIVE_SHA256",
        hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    rows = provider.tau2_task_references(
        official_archive=archive, data_root=data, task_ids=["fixture-id"]
    )
    assert len(rows) == 1
    assert rows[0]["native_task_id"] == "fixture-id"
    assert rows[0]["input_ref"]["row_index"] == 0
    assert "private fixture" not in json.dumps(rows)
    with pytest.raises(ValueError, match="unique"):
        provider.tau2_task_references(
            official_archive=archive,
            data_root=data,
            task_ids=["fixture-id", "fixture-id"],
        )
    path.write_text("[]")
    with pytest.raises(ValueError, match="differs"):
        provider.tau2_task_references(
            official_archive=archive, data_root=data, task_ids=["fixture-id"]
        )


def test_unknown_archive_fails_before_reading_private_inputs(tmp_path):
    archive = tmp_path / "not-an-official-archive"
    archive.write_bytes(b"software fixture")
    with pytest.raises(ValueError, match="identity"):
        provider.tau2_task_references(
            official_archive=archive,
            data_root=tmp_path / "does-not-exist",
            task_ids=["x"],
        )
