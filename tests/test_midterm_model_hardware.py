import importlib.util
import hashlib
import json
import io
import math
from pathlib import Path
import sqlite3
import tarfile

import pytest


ROOT = Path(__file__).resolve().parents[1] / "reports/vllm-hust-midterm-20261019"


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "model_hardware", ROOT / "recover_model_hardware.py"
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def inputs():
    return (
        dict(
            bench_started_wall=1100,
            bench_started_monotonic=100,
            bench_ended_wall=1120,
            bench_ended_monotonic=120,
        ),
        dict(start_times=[105] * 200, latencies=[10] * 200, completed=200, failed=0),
    )


def test_uses_request_span_not_client_startup(module):
    receipt, result = inputs()
    phase = module.request_phase(receipt, result)
    assert phase["start_wall"] == 1105
    assert phase["end_wall"] == 1115
    assert phase["duration_seconds"] == 10


def test_rejects_clock_jump(module):
    receipt, result = inputs()
    receipt["bench_ended_wall"] += 1
    with pytest.raises(ValueError, match="offset"):
        module.request_phase(receipt, result)


def test_rejects_outside_lifecycle(module):
    receipt, result = inputs()
    result["start_times"][0] = 99
    with pytest.raises(ValueError, match="outside"):
        module.request_phase(receipt, result)


def test_incomplete_capture_never_becomes_dual_card_total():
    directory = ROOT / "m01-recovery/model-r1"
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["status"] == "incomplete_hbm_coverage"
    assert summary["total_prorated_bytes"] is None
    assert summary["attainment_percent"] is None
    assert sorted(d["minimum_channel_coverage"] for d in summary["devices"]) == [0, 1]
    archive = json.loads((directory / "archive.json").read_text())
    concatenated = hashlib.sha256()
    for part in archive["parts"]:
        data = (directory / part["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == part["sha256"]
        concatenated.update(data)
    assert concatenated.hexdigest() == archive["concatenated_sha256"]


def test_complete_capture_recomputes_from_archived_databases(module, tmp_path):
    directory = ROOT / "m01-recovery/model-r2"
    summary = json.loads((directory / "summary.json").read_text())
    metadata = json.loads((directory / "archive.json").read_text())
    payload = b"".join((directory / p["file"]).read_bytes() for p in metadata["parts"])
    assert hashlib.sha256(payload).hexdigest() == metadata["concatenated_sha256"]
    assert summary["status"] == "complete_hbm_coverage"
    phase = summary["phase"]
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for name, digest in metadata["members"].items():
            assert (
                hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest
            )
        receipt = json.load(archive.extractfile("run/receipt.json"))
        result = json.load(archive.extractfile("run/mmlu-pro.json"))
        assert module.request_phase(receipt, result) == phase
        for d in summary["devices"]:
            device = d["channels"][0]["device"]
            names = [
                n
                for n in archive.getnames()
                if n.endswith(f"device_{device}/sqlite/hbm.db")
            ]
            assert len(names) == 1
            data = archive.extractfile(names[0]).read()
            assert hashlib.sha256(data).hexdigest() == d["sha256"]
            path = tmp_path / f"hbm-{device}.db"
            path.write_bytes(data)
            with sqlite3.connect(path) as connection:
                samples = connection.execute(
                    "SELECT device_id,timestamp,counts,hbmid,event_type FROM HBMOriginalData WHERE replayid=0"
                ).fetchall()
            measured = module.integrate(
                samples, phase["start_wall"] * 1e9, phase["end_wall"] * 1e9
            )
            assert module.has_complete_hbm_coverage(measured)
            # Python 3.12 uses compensated float summation; retain a two-ULP bound.
            assert measured["prorated_bytes"] == pytest.approx(
                d["prorated_bytes"], rel=0, abs=2 * math.ulp(d["prorated_bytes"])
            )
    assert sum(d["prorated_bytes"] for d in summary["devices"]) == pytest.approx(
        summary["total_prorated_bytes"],
        rel=0,
        abs=2 * math.ulp(summary["total_prorated_bytes"]),
    )
    assert summary["model_effective_flops"] is None
    assert summary["attainment_percent"] is None
