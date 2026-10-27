"""Verify sealed diagnostic evidence without promoting it to M01 acceptance."""

import hashlib
import json
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "reports"
    / "vllm-hust-midterm-20261019"
    / "m01-recovery"
    / "storage-position-20261024"
)


@pytest.mark.parametrize("run", ["r4", "r5"])
def test_position_run_custody_and_scope(run):
    root = EVIDENCE / run
    metadata = json.loads((root / "archive.json").read_text())
    summary_bytes = (root / "qualification-summary.json").read_bytes()
    assert (
        hashlib.sha256(summary_bytes).hexdigest()
        == metadata["qualification_summary_sha256"]
    )
    summary = json.loads(summary_bytes)
    archive = root / metadata["archive"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == metadata["sha256"]
    assert archive.stat().st_size == metadata["bytes"]
    with tarfile.open(archive) as bundle:
        manifest = bundle.extractfile("SHA256SUMS").read()
        assert hashlib.sha256(manifest).hexdigest() == metadata["raw_manifest_sha256"]
        assert len(manifest.decode().splitlines()) == metadata["raw_entries"] == 21
        for line in manifest.decode().splitlines():
            expected, name = line.split(maxsplit=1)
            assert (
                hashlib.sha256(bundle.extractfile(name.lstrip("*")).read()).hexdigest()
                == expected
            )
        receipt = json.load(bundle.extractfile("receipt.json"))
        cache = json.load(bundle.extractfile("compiler-cache-initialization.json"))
    assert metadata["run_id"] == summary["run_id"] == receipt["run_id"]
    assert receipt["run_id"] == f"capture-b1-storage-{run}"
    assert receipt["workload"]["completed"] == 200
    assert receipt["workload"]["failed"] == 0
    assert receipt["workload"]["total_output_tokens"] == 51200
    assert receipt["client_exit_code"] == 0
    assert receipt["all_devices_released"] and receipt["sessions_drained"]
    assert cache["initial_entries"] == []
    assert cache["performance_acceptance"] is False
    assert metadata["measurement_role"] == "diagnostic_qualification"
    for field in ("effective_flops", "reference_seconds", "m01_attainment"):
        assert metadata[field] is None
    if run == "r5":
        assert summary["position_qualification"]["full_run_address_mismatches"] == 0
        assert summary["position_qualification"]["all_stage_mismatches"] == 0
        extent = summary["second_draft_read_extent"]
        assert (
            extent["one_extra"] + extent["two_extra"]
            == extent["total_mismatch_rows"]
            == 4244
        )
        assert all(value is None for value in summary["formal_metrics"].values())
