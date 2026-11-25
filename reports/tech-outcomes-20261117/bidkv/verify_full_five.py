"""Recompute public five-task metrics and verify controlled-custody receipts."""

import hashlib
import json
from pathlib import Path
import runpy


def read(path):
    return json.loads(path.read_bytes())


def verify(root):
    aliases = {"audit-final-records81.py": "audit-final-records81.py.source"}
    for name in ("manifest.json", "closure-manifest.json"):
        for item in read(root / name)["files"]:
            path = root / aliases.get(item["path"], item["path"])
            assert path.resolve().is_relative_to(root.resolve())
            raw = path.read_bytes()
            assert len(raw) == item["bytes"]
            assert hashlib.sha256(raw).hexdigest() == item["sha256"]
    summary = read(root / "summary.json")
    assert summary["formal_complete"] is True and summary["status"] == "completed"
    assert summary["planned_solver_requests_per_arm"] == 672
    assert summary["planned_judge_inputs"] == 720
    closure = read(root / "closure-and-quality-recompute.json")
    for kind in ("primary_owned_master", "final_readonly_audit"):
        value = closure[kind]
        assert value["phase"] == "completed"
        assert all(
            child["status"] == "completed" and child["actual_returncode"] == 0
            for child in value["children"]
        )
    audit = closure["final_readonly_audit"]
    assert (audit["audited_solver_requests"], audit["audited_judge_requests"]) == (
        1344,
        720,
    )
    for item in audit["quality_recomputation"]:
        assert item["byte_identical"] is True
        assert item["original_sha256"] == item["recomputed_sha256"]
        assert (
            hashlib.sha256(
                (root / (item["name"] + "-summary.json")).read_bytes()
            ).hexdigest()
            == item["original_sha256"]
        )
    source = root / "report/build_five_mod_report.py.source"
    accounting = runpy.run_path(str(source))
    report = read(root / "report/summary.json")
    assert report["builder_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert (
        report["source_capsule_manifest_sha256"]
        == hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    )
    inventory = read(root / "controlled-originals-index.json")
    hashes = {item["sha256"] for item in inventory["files"]}
    assert inventory["originals_not_redistributed"] is True
    stages = summary["stages"]
    assert len(stages) == 14 and len({stage["name"] for stage in stages}) == 14
    totals = {"solver": 0, "judge": 0}
    for stage in stages:
        count = stage["planned"]
        assert stage["status"] == "completed" and stage["exit_code"] == 0
        # campaign81 waits into exit_code; its optional cancellation-only
        # actual_client_returncode remains null and is not backfilled here.
        assert stage["unbound_sse_capture_files"] == 0
        assert stage["recorded"] == stage["sse_capture_files"] == count
        proof = stage["independent_verification"]
        assert proof["raw_sse_verified"] == proof["recorded"] == count
        assert proof["zero_reuse"]["passed"] is True
        rows = read(root / stage["numeric_records_file"])
        assert len(rows) == len({row["case_id"] for row in rows}) == count
        assert all(
            row["controlled_record_sha256"] in hashes
            and row["controlled_sse_sha256"] in hashes
            for row in rows
        )
        totals[stage["stage"]] += count
        if stage["stage"] == "solver":
            observed = accounting["observed"](stage, root, count)
            key = "native" if stage["arm"] == "native" else "candidate"
            assert observed == report["datasets"][stage["dataset"]][key]
            assert observed["protocol_and_controlled_audit_receipts_present"]
    assert totals == {"solver": 1344, "judge": 720}
    life = read(root / "full-owned-lifecycle-policy.json")
    assert set(life["services"]) == {"native", "bidkv", "judge"}
    for arm, service in life["services"].items():
        assert service["stop.json"]["value"]["exit_code"] == 0
        assert service["release"]["no_processes_both"] is True
        for journal in service["policy_journals"]:
            stats = journal["rows"][-1]["stats"]
            assert stats["enabled"] == (arm == "bidkv")
            assert stats["calls"] == stats["decisions"] == stats["selections"] == 0
            assert stats["failures"] == stats["invalid_selections"] == 0
    # Public data contains hashes/receipts, not the controlled raw streams.
    return {
        "formal_complete": True,
        "public_numeric_record_identity_checks": 2064,
        "public_solver_metric_records_recomputed": 1344,
        "public_judge_records_counted": 720,
        "controlled_sse_replay_receipt_requests": 2064,
        "public_original_sse_readback": False,
        "policy_calls": 0,
        "quality_original_byte_identical_recomputation_receipts": 4,
    }


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent / "five-results-v1")))
