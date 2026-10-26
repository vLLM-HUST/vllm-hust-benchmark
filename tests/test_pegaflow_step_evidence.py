"""Custody and numerical data boundaries for the PegaFlow scheduling study."""

import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "reports/pegaflow-step-simulator-20261026"


def read(name):
    raw = (ROOT / name).read_bytes()
    return json.loads(gzip.decompress(raw) if name.endswith(".gz") else raw)


def test_evidence_members_and_measurement_classification():
    members = {row["path"]: row for row in read("manifest.json")["members"]}
    for name, row in members.items():
        raw = (ROOT / name).read_bytes()
        assert len(raw) == row["bytes"]
        assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    assert members["private-prediction.json.gz"]["kind"] == "simulation"
    assert (
        members["data/validation-95-shared-observed.json.gz"]["kind"]
        == "normalized_measurement"
    )


def test_training_and_validation_tasks_are_disjoint():
    metadata = read("data/calibration-metadata.json")
    raw = gzip.decompress((ROOT / "data/calibration-steps.jsonl.gz").read_bytes())
    frames = [json.loads(line) for line in raw.splitlines()]
    assert len(frames) == 1326
    assert {row["case"] for row in frames} == set(metadata["training_cases"])
    for name in ("private", "shared"):
        workload = read(f"data/validation-95-{name}-workload.json.gz")
        observed = read(f"data/validation-95-{name}-observed.json.gz")
        assert len(workload["requests"]) == 95
        assert not set(metadata["training_tasks"]) & {
            row["task"] for row in workload["requests"]
        }
        assert {row["id"] for row in workload["requests"]} == {
            row["id"] for row in observed["requests"]
        }
        assert all(
            not {"first_s", "finish_s"} & row.keys() for row in workload["requests"]
        )


def test_summary_preserves_separate_measured_and_predicted_metrics():
    summary = read("summary.json")
    for name in ("private", "shared"):
        result = read(f"{name}-validation.json")
        assert summary["validation"][name]["measured"] == result["measured"]
        assert summary["validation"][name]["predicted"] == result["predicted"]
        prediction = read(f"{name}-prediction.json.gz")
        assert (
            prediction["summary"]["finish_span_s"]
            == result["predicted"]["finish_span_s"]
        )
