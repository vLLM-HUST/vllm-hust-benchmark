"""CPU software fixtures only; no original scientific articles or model calls."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from vllm_hust_workload_data import provider
from vllm_hust_workload_data.adapters import adapt_row
from vllm_hust_workload_data.registry import adapter_spec
from vllm_hust_workload_data.catalog import download_plan


def fixture_record(**updates):
    row = {
        "_id": "fixture/0",
        "context": "Original fixture article.\n",
        "input": "Fixture question?",
        "answers": ["SCORER_ONLY_ANSWER"],
        "all_classes": ["SCORER_ONLY_CLASS"],
        "dataset": "qasper",
        "language": "en",
        "length": 5,
    }
    row.update(updates)
    return adapt_row(
        "LONGBENCH",
        row,
        source={"source_id": "LONGBENCH"},
        spec=adapter_spec("LONGBENCH"),
        index=0,
    )


def test_qasper_original_input_and_scorer_boundary():
    record = fixture_record()
    assert record.task["input"] == {
        "context": "Original fixture article.\n",
        "input": "Fixture question?",
    }
    assert "SCORER_ONLY" not in json.dumps(record.input_record())
    assert record.outcome_record()["answers"] == ["SCORER_ONLY_ANSWER"]
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts/run_longbench_v1_qasper_original.py"
    )
    spec = importlib.util.spec_from_file_location("qasper_runner", script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert (
        runner.render_prompt(record, "{context}\nQ:{input}\nA:")
        == "Original fixture article.\n\nQ:Fixture question?\nA:"
    )
    with pytest.raises(ValueError, match="source archive identity"):
        runner.official_resources(script)


def test_original_null_all_classes_is_preserved_not_fabricated():
    record = fixture_record(all_classes=None)
    assert record.outcome_record()["all_classes"] is None


def test_download_plan_explains_archive_preparation_not_a_fabricated_file_url():
    plan = download_plan("LONGBENCH")
    assert plan["preparation_command"] == "workload-data prepare-longbench-qasper"
    member = next(f for f in plan["files"] if f["role"] == "data")
    assert member["archive_member"] == "data/qasper.jsonl"
    assert "url" not in member


@pytest.mark.parametrize(
    "update", [{"dataset": "hotpotqa"}, {"context": ""}, {"input": []}, {"answers": []}]
)
def test_unreviewed_subsets_or_invalid_inputs_rejected(update):
    with pytest.raises(ValueError, match="qasper fields"):
        fixture_record(**update)


def test_preparation_only_copies_exact_member_to_fresh_directory(tmp_path, monkeypatch):
    raw = b'{"fixture":true}\n' * 200
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("data/qasper.jsonl", raw)
        bundle.writestr("../not-extracted.txt", "fixture")
    monkeypatch.setattr(
        provider,
        "LONGBENCH_V1_DATA_ARCHIVE_SHA256",
        hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        provider, "LONGBENCH_QASPER_SHA256", hashlib.sha256(raw).hexdigest()
    )
    output = tmp_path / "prepared"
    receipt = provider.prepare_longbench_qasper(data_archive=archive, output=output)
    assert (output / "qasper.jsonl").read_bytes() == raw
    assert {p.name for p in output.iterdir()} == {"qasper.jsonl"}
    assert receipt["status"] == "ORIGINAL_FILE_PREPARED_NOT_EXECUTED"
    assert receipt["model_calls"] == receipt["scorer_runs"] == 0
    with pytest.raises(FileExistsError):
        provider.prepare_longbench_qasper(data_archive=archive, output=output)
    archive.write_bytes(b"changed archive")
    with pytest.raises(ValueError, match="archive identity"):
        provider.prepare_longbench_qasper(
            data_archive=archive, output=tmp_path / "not-created"
        )
    assert not (tmp_path / "not-created").exists()


def test_help_and_invalid_model_route_without_optional_runtime(tmp_path):
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts/run_longbench_v1_qasper_original.py"
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
            "--output",
            str(tmp_path / "no-output"),
            "--endpoint",
            "https://example.com/v1",
        ],
        capture_output=True,
    )
    assert result.returncode == 2
    assert not (tmp_path / "no-output").exists()
