"""Classification, numerical consistency and lifecycle of P/D predictions."""

import gzip
import hashlib
import json
import math
from pathlib import Path

import pytest


REPORTS = Path(__file__).resolve().parents[1] / "reports"
FOLDERS = (
    "pegaflow-pd-serial-validation-20261027",
    "pegaflow-pd-capacity-batching-20261027",
    "pegaflow-pd-serving-scenario-20261027",
    "pegaflow-simulation-progress-20261027",
)


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


@pytest.mark.parametrize("folder", FOLDERS)
def test_report_inventory_and_provenance(folder):
    root = REPORTS / folder
    manifest = read(root / "manifest.json")
    assert manifest["scope"] == "Research simulation and diagnostic evidence"
    members = {m["path"]: m for m in manifest["members"]}
    assert set(members) == {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    for name, member in members.items():
        data = (root / name).read_bytes()
        assert len(data) == member["bytes"]
        assert hashlib.sha256(data).hexdigest() == member["sha256"]
    source = read(root / "provenance/reproduction-source.json")
    assert source["repository"] == "vLLM-HUST/pegaflow-hust"
    assert len(source["commit"]) == 40
    assert (
        source["scope"]
        == "Maintained reproduction entrypoints; executed snapshots retained separately"
    )
    if folder == FOLDERS[0]:
        assert members["observations.json.gz"]["kind"] == "normalized_measurement"
        assert members["prediction.json.gz"]["kind"] == "simulation"


def test_workload_identity_and_prediction_input_boundary():
    expected = read(REPORTS / FOLDERS[0] / "inputs/simulation-inputs.json.gz")
    protocol = read(REPORTS / FOLDERS[0] / "inputs/protocol.json.gz")
    tasks = {r["task"] for r in expected["rows"]}
    assert len(expected["rows"]) == 95 and len(tasks) == 8
    for folder in FOLDERS[:3]:
        bundle = read(REPORTS / folder / "inputs/simulation-inputs.json.gz")
        assert bundle["rows"] == expected["rows"]
        assert all(set(r) == {"id", "task", "tokens", "hashes"} for r in bundle["rows"])
        actual = read(REPORTS / folder / "inputs/protocol.json.gz")
        assert actual["arrival_condition"] == protocol["arrival_condition"]
    calibration = read(
        REPORTS / "pegaflow-step-simulator-20261026/data/calibration-metadata.json"
    )
    assert not tasks & set(calibration["training_tasks"])


@pytest.mark.parametrize(
    "folder,name",
    [
        (FOLDERS[0], "prediction.json.gz"),
        (FOLDERS[1], "serial2.json.gz"),
        (FOLDERS[1], "serial16.json.gz"),
        (FOLDERS[1], "batch16.json.gz"),
        (FOLDERS[1], "batch32.json.gz"),
        (FOLDERS[2], "graph-p8192-d2048.json.gz"),
        (FOLDERS[2], "graph-p2048-d2048.json.gz"),
    ],
)
def test_prediction_metrics_and_session_lifecycle(folder, name):
    result = read(REPORTS / folder / name)
    rows = result["per_request"]
    assert len(rows) == len({r["id"] for r in rows}) == 95
    summary = result.get("summary", result)
    ttft = sorted(r["first"] - r["ready"] for r in rows)
    assert summary["TTFT"]["mean"] == pytest.approx(sum(ttft) / 95)
    for percent in (50, 95, 99):
        assert summary["TTFT"][f"p{percent}"] == pytest.approx(
            ttft[math.ceil(percent * 95 / 100) - 1]
        )
    assert summary["finish_span"] == pytest.approx(max(r["end"] for r in rows))
    previous = {}
    for row in sorted(rows, key=lambda r: r["ready"]):
        task = row["id"].rsplit(":", 1)[0]
        assert row["ready"] >= previous.get(task, 0) - 1e-8
        previous[task] = row["end"]
        p, d = row["stages"]["P"], row["stages"]["D"]
        if "engine_end" in p:
            assert p["engine_end"] <= d["prepare"] <= row["first"] <= row["end"]
        else:
            assert p["end"] <= d["start"] <= row["first"] <= row["end"]


def test_capacity_and_target_architecture_claims_are_separate():
    capacity = read(REPORTS / FOLDERS[1] / "summary.json")
    assert capacity["batch16"]["native_reclaims"] == 0
    assert capacity["batch16"]["full_D_block_prefix_requests"] == 95
    assert capacity["batch16"]["TTFT"] == capacity["batch32"]["TTFT"]
    serving = read(REPORTS / FOLDERS[2] / "graph-p8192-d2048.json.gz")
    control = read(REPORTS / FOLDERS[2] / "graph-p2048-d2048.json.gz")
    assert serving["settings"]["graph_mode"] is True
    assert serving["settings"]["max_sequences_each"] == 128
    assert serving["summary"]["extrapolated_steps"] > 0
    assert control["summary"]["extrapolated_steps"] == 0
    assert "unvalidated" in serving["scope"]
    assert "session affinity" in serving["settings"]["routing"]
    bundle = read(REPORTS / FOLDERS[2] / "inputs/simulation-inputs.json.gz")
    suffixes = sum(r["tokens"] - 128 * len(r["hashes"]) for r in bundle["rows"])
    assert serving["summary"]["D_uncached_tokens"] == suffixes
