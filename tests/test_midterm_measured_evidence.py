"""Recompute retained measurements without importing an accelerator runtime."""

import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile

import pytest


REPORT = Path(__file__).resolve().parents[1] / "reports/vllm-hust-midterm-20261019"
spec = importlib.util.spec_from_file_location(
    "midterm_verify", REPORT / "verify_long_run.py"
)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


@pytest.fixture
def evidence(tmp_path):
    with tarfile.open(REPORT / "synthetic-b0-evidence.tar.gz") as archive:
        archive.extractall(tmp_path, filter="data")
    return tmp_path / "midterm-run-20261019-r4"


def test_retained_baseline_recomputes(evidence):
    result = verifier.verify(evidence)
    formal = result["phases"]["formal"]
    assert formal["window_tokens"] == 79437
    assert formal["output_tps"] == pytest.approx(264.79)
    assert formal["completed_in_window"] == 304
    assert formal["completed_in_drain"] == 8
    assert formal["drain_tokens"] == 435
    assert formal["failed"] == 0
    assert not result["candidate_measured"]


def test_source_mutation_is_rejected(evidence):
    source = evidence / "measure.py"
    source.write_text(source.read_text() + "\n# altered\n")
    with pytest.raises(AssertionError, match="measure.py"):
        verifier.verify(evidence)


def test_failed_run_is_rejected(evidence):
    path = evidence / "native-controller-r2/receipt.json"
    receipt = json.loads(path.read_text())
    receipt["exit_code"] = 1
    path.write_text(json.dumps(receipt))
    with pytest.raises(AssertionError):
        verifier.verify(evidence)


def test_inconsistent_token_events_are_rejected(evidence):
    path = evidence / "native/requests.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["events"][0]["tokens"] += 1
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(AssertionError):
        verifier.verify(evidence)


def test_calibration_arithmetic_and_release(tmp_path):
    with tarfile.open(REPORT / "calibration-evidence.tar.gz") as archive:
        archive.extractall(tmp_path, filter="data")
    root = tmp_path / "midterm-calibration-20261019"
    receipt = json.loads((root / "native-controller-r2/receipt.json").read_text())
    assert receipt["status"] == "passed"
    assert receipt["exit_code"] == 0
    assert receipt["owned_processes_gone"]
    for rank in (0, 1):
        data = json.loads((root / f"rank-{rank}.json").read_text())
        assert data["physical_device"] == rank + 4
        for record in data["records"]:
            times = record["seconds_per_iteration"]
            assert len(times) == 9 and min(times) > 0
            median = statistics.median(times)
            assert record["median_seconds"] == pytest.approx(median)
            assert record["median_rate"] == pytest.approx(record["numerator"] / median)


def test_candidate_negative_result_is_retained(evidence, tmp_path):
    with tarfile.open(REPORT / "synthetic-b1-evidence.tar.gz") as archive:
        archive.extractall(tmp_path, filter="data")
    output = tmp_path / "comparison.json"
    subprocess.run(
        [
            sys.executable,
            str(REPORT / "compare_long_runs.py"),
            str(evidence),
            str(tmp_path / "midterm-b1-20261019"),
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text())
    assert result["B1"]["phases"]["formal"]["window_tokens"] == 57585
    assert result["changes"]["throughput_gain_percent"] == pytest.approx(
        -27.50859171418861
    )
    assert not any(result["initial_serving_threshold_observations"].values())
    assert not result["task_quality_qualified"]
    assert not result["overall_midterm_acceptance"]
