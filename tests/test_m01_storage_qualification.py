"""Keep diagnostic qualification separate from formal M01 acceptance."""

import hashlib
import json
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "reports"
    / "vllm-hust-midterm-20261019"
    / "m01-recovery"
    / "storage-qualification-20261024"
)


def test_storage_raw_archive_and_qualification():
    metadata = json.loads((EVIDENCE / "archive.json").read_text())
    archive = EVIDENCE / metadata["archive"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == metadata["sha256"]
    assert archive.stat().st_size == metadata["bytes"]
    with tarfile.open(archive) as bundle:
        manifest = bundle.extractfile("SHA256SUMS").read()
        assert hashlib.sha256(manifest).hexdigest() == metadata["raw_manifest_sha256"]
        lines = manifest.decode().splitlines()
        assert len(lines) == metadata["verified_raw_entries"] == 21
        for line in lines:
            expected, name = line.split(maxsplit=1)
            payload = bundle.extractfile(name.lstrip("*")).read()
            assert hashlib.sha256(payload).hexdigest() == expected
        receipt = json.load(bundle.extractfile("receipt.json"))
        cache = json.load(bundle.extractfile("compiler-cache-initialization.json"))
    assert receipt["run_id"] == metadata["run_id"] == "capture-b1-storage-r3"
    assert receipt["workload"]["completed"] == 200
    assert receipt["workload"]["failed"] == 0
    assert receipt["workload"]["total_output_tokens"] == 51200
    assert receipt["sessions_drained"] and receipt["all_devices_released"]
    assert cache["initial_entries"] == []
    assert cache["performance_acceptance"] is False
    assert receipt["measurement_role"] == "diagnostic_qualification"
    for key in ("effective_flops", "m01_attainment"):
        assert receipt[key] is None
        assert metadata[key] is None
    assert metadata["reference_seconds"] is None


def test_storage_derived_audit_keeps_original_limits():
    metadata = json.loads((EVIDENCE / "archive.json").read_text())["audit_archive"]
    archive = EVIDENCE / metadata["file"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == metadata["sha256"]
    with tarfile.open(archive) as bundle:
        audit = json.load(bundle.extractfile("storage-r3-final-audit.json"))
        source = bundle.extractfile("audit_storage_r3.py").read()
        suites = ET.parse(bundle.extractfile("storage-r3-regression.xml"))
    assert hashlib.sha256(source).hexdigest() == audit["audit_source_sha256"]
    assert audit["original_retained_count_complete"] is False
    assert audit["original_unclassified_event_count"] == 155
    assert audit["derived_nonupdate_classification"]["unclassified"] == 0
    assert audit["m01_attainment"] is None
    assert audit["effective_flops"] is None
    assert audit["reference_time"] is None
    assert sum(int(s.get("tests", 0)) for s in suites.iter("testsuite")) == 47
    assert all(
        int(s.get("failures", 0)) == int(s.get("errors", 0)) == 0
        for s in suites.iter("testsuite")
    )
