from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile

BASE = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019/m01-recovery/diagnostic-capture-20261023"
)


def test_failed_capture_preserves_identity_and_recovered_counts():
    manifest = json.loads((BASE / "manifest.json").read_text())
    records = {}
    for item in manifest["archives"]:
        archive = BASE / item["archive"]
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == item["sha256"]
        members = {}
        with tarfile.open(archive) as bundle:
            for member in bundle:
                assert (
                    not Path(member.name).is_absolute()
                    and ".." not in Path(member.name).parts
                )
                assert member.isfile() or member.isdir()
                if member.isfile():
                    raw = bundle.extractfile(member).read()
                    members[member.name] = hashlib.sha256(raw).hexdigest()
                    if member.name.startswith("m01-capture-b1-r1/") or member.name in (
                        "capture-r1-host-count-recovery.json",
                        "capture-r1-skipped-row-classification.json",
                    ):
                        records[member.name] = raw
        assert members == item["members"]
    prefix = "m01-capture-b1-r1/"
    receipt = json.loads(records[prefix + "receipt.json"])
    assert receipt["status"] == "failed" and receipt["client_exit_code"] == 1
    assert receipt["sessions_drained"] and receipt["all_devices_released"]
    assert prefix + "mmlu-pro.json" not in records
    ledger_raw = records[prefix + "scheduler-712941.json"]
    ledger = json.loads(ledger_raw)
    recovery = json.loads(records["capture-r1-host-count-recovery.json"])
    boundary = json.loads(records[prefix + "boundary.jsonl"])
    assert (
        recovery["low_perf_ns"] == boundary["calls"][2]["clock_after"]["perf_after_ns"]
    )
    assert (
        recovery["high_perf_ns"]
        == boundary["calls"][3]["clock_before"]["perf_before_ns"]
    )
    assert (
        recovery["source_sha256"]["/workspace/m01-capture-b1-r1/scheduler-712941.json"]
        == hashlib.sha256(ledger_raw).hexdigest()
    )
    events = [
        e
        for e in ledger["events"]
        if recovery["low_perf_ns"] <= e["perf_ns"] <= recovery["high_perf_ns"]
    ]
    updates = [e for e in events if e["event"] == "output_update"]
    per_request = Counter()
    for event in updates:
        per_request[event["request_key"]] += event["observation"]["appended_count"]
    assert len(per_request) == 200 and set(per_request.values()) == {256}
    assert sum(per_request.values()) == recovery["appended_output_tokens"] == 51200
    for field in ("raw_accepted_drafts", "retained_drafts"):
        assert sum(e["observation"][field] for e in updates) == recovery[field]
    assert recovery["raw_accepted_drafts"] - recovery["retained_drafts"] == 56
    skipped = json.loads(records["capture-r1-skipped-row-classification.json"])
    assert skipped["source_sha256"] == hashlib.sha256(ledger_raw).hexdigest()
    assert skipped["original_retained_count_complete"] is False
    assert skipped["classes"] == {
        "before_first_output_no_drafts": 16,
        "after_complete_output_budget": 124,
    }
    prior, classification = Counter(), Counter()
    for event in ledger["events"]:
        if event["event"] == "output_update":
            prior[event["request_key"]] += event["observation"]["appended_count"]
        elif event["event"] == "not_updated":
            state = (event["drafts"], prior[event["request_key"]])
            assert state in ((0, 0), (2, 256))
            classification[
                "before_first_output_no_drafts"
                if state == (0, 0)
                else "after_complete_output_budget"
            ] += 1
    assert dict(classification) == skipped["classes"]
    for record in (manifest, recovery, skipped):
        assert record["effective_flops"] is None and record["m01_attainment"] is None
    assert manifest["performance_acceptance"] is False
