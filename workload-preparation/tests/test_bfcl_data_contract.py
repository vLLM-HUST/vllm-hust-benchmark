"""Invented CPU fixtures; no official task payloads, model calls or AST scores."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import zipfile

import pytest

from vllm_hust_workload_data import iter_workload, provider
from vllm_hust_workload_data.assets import iter_rows
from vllm_hust_workload_data.bindings import bind_source
from vllm_hust_workload_data.consumer import resolve_input
from vllm_hust_workload_data.pipeline import prepared_rows, run


def fixture_row(index):
    return {
        "id": f"simple_python_{index}",
        "question": [[{"role": "user", "content": "INPUT_FIXTURE"}]],
        "function": [
            {
                "name": "fixture.function",
                "parameters": {"type": "dict", "properties": {}},
            }
        ],
    }


def test_original_json_filename_is_explicit_jsonl_across_entire_consumer_chain(
    tmp_path,
):
    root = tmp_path / "raw"
    root.mkdir()
    filename = "BFCL_v4_simple_python.json"
    original = "\n".join(json.dumps(fixture_row(i)) for i in range(3)) + "\n"
    path = root / filename
    path.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="payload suppressed"):
        list(iter_rows(path))  # Default .json remains a JSON array, no auto guessing.
    assert list(iter_rows(path, row_format="jsonl")) == [
        fixture_row(i) for i in range(3)
    ]
    manifest = bind_source(
        "BFCL", data_root=root, files=[filename], revision="software-fixture"
    )
    assert manifest["upstream_revision_verified"] is False
    binding = tmp_path / "binding.json"
    binding.write_text(json.dumps(manifest), encoding="utf-8")
    records = list(iter_workload("BFCL", data_root=root, manifest=manifest))
    assert [r.task["input"] for r in records] == [
        {k: fixture_row(i)[k] for k in ("question", "function")} for i in range(3)
    ]
    assert all(r.outcome_record() is None for r in records)
    output = tmp_path / "prepared"
    report = run(
        {
            "sources": [
                {"source_id": "BFCL", "data_root": str(root), "manifest": str(binding)}
            ]
        },
        output=output,
    )
    assert report["model_calls"] == 0
    database = output / "workload.sqlite"
    indexed = list(prepared_rows(database))
    assert len(indexed) == 3
    with sqlite3.connect(database) as connection:
        serialized = "\n".join(connection.iterdump())
    assert "INPUT_FIXTURE" not in serialized
    for item, expected in zip(indexed, records):
        recovered = resolve_input(database, item["record_id"], data_root=root)
        assert recovered.task == expected.task
        assert recovered.source == expected.source
        assert recovered.outcome_record() is None
    assert path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("row_format", ["parquet", "array", {}, True])
def test_unreviewed_explicit_format_is_not_a_generic_guess(tmp_path, row_format):
    path = tmp_path / "fixture.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="payload suppressed"):
        list(iter_rows(path, row_format=row_format))


def test_native_preparation_keeps_scorer_separate_and_never_runs_it(
    tmp_path, monkeypatch
):
    archive = tmp_path / "fixture.zip"
    raw = b'{"fixture":"task"}\n' * 400
    gold = b'{"fixture":"SCORER_ONLY_SENTINEL"}\n' * 400
    files = {
        "BFCL_v4_simple_python.json": raw,
        "possible_answer/BFCL_v4_simple_python.json": gold,
    }
    prefix = "ShishirPatil-gorilla-6ea5797/berkeley-function-call-leaderboard/bfcl_eval/data/"
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, data in files.items():
            bundle.writestr(prefix + name, data)
        bundle.writestr("../not-extracted", "fixture")
    monkeypatch.setattr(
        provider,
        "BFCL_ARCHIVE_SHA256",
        hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        provider,
        "BFCL_SIMPLE_MEMBERS",
        {k: hashlib.sha256(v).hexdigest() for k, v in files.items()},
    )
    output = tmp_path / "original"
    receipt = provider.prepare_bfcl_simple_python(
        official_archive=archive, output=output
    )
    assert receipt["model_calls"] == receipt["scorer_runs"] == 0
    assert receipt["status"] == "ORIGINAL_FILES_PREPARED_NOT_EXECUTED"
    assert [f["role"] for f in receipt["files"]] == ["data", "scorer_reference"]
    assert (output / "BFCL_v4_simple_python.json").read_bytes() == raw
    assert (output / "possible_answer/BFCL_v4_simple_python.json").read_bytes() == gold
    assert not (tmp_path / "not-extracted").exists()
    with pytest.raises(FileExistsError):
        provider.prepare_bfcl_simple_python(official_archive=archive, output=output)
    archive.write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity mismatch"):
        provider.prepare_bfcl_simple_python(
            official_archive=archive, output=tmp_path / "not-created"
        )
    assert not (tmp_path / "not-created").exists()


def test_public_native_entry_help_and_endpoint_rejection_need_no_optional_runtime(
    tmp_path,
):
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts/run_bfcl_v4_simple_python_original.py"
    )
    assert (
        subprocess.run(
            [sys.executable, str(script), "--help"], capture_output=True
        ).returncode
        == 0
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--data-root",
            str(tmp_path),
            "--official-archive",
            str(tmp_path / "missing.zip"),
            "--tokenizer",
            str(tmp_path / "no-model"),
            "--output",
            str(tmp_path / "not-created"),
            "--endpoint",
            "https://example.com/v1",
        ],
        capture_output=True,
    )
    assert result.returncode == 2
    assert not (tmp_path / "not-created").exists()
    spec = importlib.util.spec_from_file_location("bfcl_runner", script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    raw = tmp_path / "not-an-archive"
    raw.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="archive identity"):
        runner.verify_assets(raw, tmp_path)
