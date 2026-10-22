import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/vllm-hust-midterm-20261019"
DATA = REPORT / "m01-recovery/fenced-model-20261022"


def test_original_receipts_and_audit_are_pinned():
    manifest = json.loads((DATA / "archive.json").read_text())
    for member in manifest["selected_members"]:
        assert (
            hashlib.sha256((DATA / member["name"]).read_bytes()).hexdigest()
            == member["sha256"]
        )
    archive = next(a for a in manifest["archives"] if a["in_git"])
    path = DATA / archive["name"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == archive["sha256"]
    with tarfile.open(path) as tar:
        actual = json.load(tar.extractfile("m01-fenced-r1-audit.json"))
    assert actual == json.loads((DATA / "audit.json").read_text())
    assert actual["completed"] == 200 and actual["failed"] == 0
    assert actual["model_effective_flops"] is actual["m01_attainment"] is None
    assert actual["ranks"][0]["rows"]["TASK"] is None
    assert actual["ranks"][1]["rows"]["TASK"] == 3462491
    assert actual["capture_gaps"] and actual["all_devices_released"]


def test_device_window_is_not_relabelled_request_time():
    envelope = json.loads((DATA / "device-envelope.json").read_text())
    audit = json.loads((DATA / "audit.json").read_text())
    assert (
        envelope["duration_seconds"]
        == (envelope["end_ns"] - envelope["start_ns"]) / 1e9
    )
    assert audit["device_envelope_seconds"] == envelope["duration_seconds"]
    assert audit["request_seconds"] < audit["device_envelope_seconds"]
    before, after = envelope["begin_fences"], envelope["end_fences"]
    assert {r["rank"] for r in before} == {0, 1}
    assert {(r["rank"], r["pid"]) for r in before} == {
        (r["rank"], r["pid"]) for r in after
    }
    assert all(r["device_sync_completed"] for r in before + after)
    assert all(r["sync_end_ns"] <= envelope["start_ns"] for r in before)
    assert all(
        envelope["start_ns"]
        <= r["sync_start_ns"]
        <= r["sync_end_ns"]
        <= envelope["end_ns"]
        for r in after
    )
    report = (REPORT / "中期测试报告.md").read_text()
    assert "完整采集未通过" in report
    assert "175.853432" in report and "125.031105" in report
