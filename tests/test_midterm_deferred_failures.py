"""Failed model attempts remain failed, with independently checked provenance."""

import hashlib
import json

from test_midterm_vocab_prefill import BASE, archive_members


def test_deferred_failed_attempt_receipts():
    root = BASE / "deferred-failed-r10-r11"
    members = archive_members(root)
    runs = json.loads((root / "runs.json").read_text())["runs"]
    assert len(runs) == 2
    for row in runs:
        prefix = row["run"] + "/"
        data = members[prefix + "receipt.json"]
        assert hashlib.sha256(data).hexdigest() == row["receipt_sha256"]
        receipt = json.loads(data)
        assert receipt["status"] == row["status"] == "failed"
        assert receipt["all_devices_released"]
        assert receipt["ended"] > receipt["started"]
        for name, digest in receipt["input_hashes"].items():
            assert hashlib.sha256(members["source/" + name]).hexdigest() == digest
        log = members[prefix + "service.log"].decode()
        assert "aclrtMallocHostWithCfg, error code is 207001" in log
        assert 'host_state.py", line 250, in offload' in log
        assert not any(
            name.startswith(prefix + "moe-deferred-") and name.endswith(".ready")
            for name in members
        )
    result = json.loads(members["microflow-token-accounting-r10/mmlu-pro.json"])
    assert result["completed"] == 104 and result["failed"] == 96


def test_r11_allocator_failure_near_container_limit():
    members = archive_members(BASE / "deferred-failed-r10-r11")
    rows = [
        json.loads(line)
        for name, data in members.items()
        if name.startswith("microflow-token-accounting-r11/token-accounting-")
        and name.endswith(".jsonl")
        for line in data.splitlines()
    ]
    failures = [
        row
        for row in rows
        if row["event"] == "host_allocator_observation" and row["phase"] == "failure"
    ]
    assert len(failures) == 1
    failure = failures[0]
    remaining = int(failure["cgroup"]["memory.max"]) - int(
        failure["cgroup"]["memory.current"]
    )
    assert remaining == 5566464
    allocator = failure["allocator"]
    assert allocator["allocated_bytes.current"] == 9942664808
    assert allocator["active_bytes.current"] == 9942610216
    assert (
        allocator["allocated_bytes.current"] - allocator["active_bytes.current"]
        == 54592
    )
    assert failure["committed_bytes"] == 7848856832
    assert failure["reserved_bytes"] == 0
    assert failure["budget_bytes"] == 8589934592
    assert "oom_kill 0" in failure["cgroup"]["memory.events"]


def test_reclamation_preserves_files_and_does_not_change_serving_sources():
    members = archive_members(BASE / "deferred-failed-r10-r11")
    receipt = json.loads(members["source/old-profile-cache-reclaim-r1.json"])
    assert len(receipt["files"]) == 4130
    assert receipt["before"]["current"] - receipt["after"]["current"] == 18443829248
    assert all(row["size"] >= 0 and row["inode"] > 0 for row in receipt["files"])
    source = members["source/reclaim_old_profile_cache.py"].decode()
    assert "os.POSIX_FADV_DONTNEED" in source
    assert 'path.open("rb")' in source
    plan = json.loads(members["source/r12-retry-plan.json"])
    original = json.loads(members["microflow-token-accounting-r11/receipt.json"])
    assert plan["input_hashes"] == original["input_hashes"]
    assert plan["status"] == "frozen_before_retry"
