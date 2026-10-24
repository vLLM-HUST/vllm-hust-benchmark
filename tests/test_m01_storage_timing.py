import hashlib
import json
import math
import tarfile
from pathlib import Path

ROOT = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "vllm-hust-midterm-20261019"
    / "m01-recovery"
    / "storage-timing-20261024"
)


def test_same_run_timing_archive_and_arithmetic():
    index = json.loads((ROOT / "archive.json").read_text())
    archive = ROOT / index["archive"]
    assert archive.stat().st_size == index["bytes"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == index["sha256"]
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        assert all(member.isfile() for member in members)
        contents = {
            member.name: bundle.extractfile(member).read() for member in members
        }
    manifest = contents["SHA256SUMS"]
    assert hashlib.sha256(manifest).hexdigest() == index["raw_manifest_sha256"]
    lines = manifest.decode().splitlines()
    assert len(lines) == index["raw_entries"] == 19
    for line in lines:
        digest, name = line.split(maxsplit=1)
        assert hashlib.sha256(contents[name.lstrip("*")]).hexdigest() == digest
    for name, digest in index["derived_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest

    receipt = json.loads(contents["receipt.json"])
    assert receipt["client_exit_code"] == 0
    assert receipt["sessions_drained"] and receipt["all_devices_released"]
    assert receipt["measurement"]["completed"] == 200
    assert receipt["measurement"]["failed"] == 0
    cache = json.loads(contents["compiler-cache-initialization.json"])
    assert cache["initial_entries"] == []
    plan = json.loads(contents["execution-plan.json"])
    assert plan["strict_uninstrumented"] is False
    assert plan["host_prefill_counting"] and plan["all_measurement_cost_retained"]
    assert not plan["device_journals"] and not plan["model_profiler"]

    result = json.loads((ROOT / "arithmetic.json").read_text())
    assert result["run_id"] == receipt["run_id"] == index["run_id"]
    total = (
        sum(result["target_matrix"].values())
        + result["target_scalar"]["basic_subtotal"]
        + result["target_prefill"]["basic_subtotal"]
        + result["draft"]["basic_subtotal"]
    )
    assert total == result["basic_operation_flops"] == 1812905151088034
    assert result["participating_chips"] == 2
    seconds = result["synchronized_seconds"]
    assert seconds == receipt["measurement"]["synchronized_seconds"]
    assert math.isclose(
        result["basic_operation_tflops_per_chip"],
        total / (2 * seconds * 1e12),
        rel_tol=1e-14,
    )
    assert result["client_seconds"] <= seconds
    for key in ("reference_seconds", "efficiency_percent", "m01_attainment"):
        assert result[key] is None
        assert index[key] is None
