from pathlib import Path
import hashlib
import json

r = Path(__file__).parent
for line in (r / "SHA256SUMS").read_text().splitlines():
    h, n = line.split("  ", 1)
    assert Path(n).name == n
    assert hashlib.sha256((r / n).read_bytes()).hexdigest() == h
p = json.loads((r / "report.json").read_text())
assert p["source_validation"]["files_verified"] == 6436
assert p["source_validation"]["tasks"] == p["cpu_validation"]["tasks_loaded"] == 642
assert p["cpu_validation"]["failures"] == []
assert all(v == "PASS" for v in p["cpu_validation"]["official_tool_imports"].values())
assert p["image"]["platforms"] == [{"os": "linux", "architecture": "amd64"}]
assert p["actual_swe_tasks_evaluated"] == 0
assert p["resolved_tasks"] is p["repair_success_rate"] is None
assert not p["official_agent_fix_patch_produced"] and not p["probes_executed"]
assert len(p["environment_attempts"]) == 2
assert all(
    x["exit_code"] == 1 and x["model_calls"] == 0 for x in p["environment_attempts"]
)
assert len(p["interface_proofs"]) == 2
for x in p["interface_proofs"]:
    assert (
        x["integrity"] == "PASS"
        and x["same_forwarded_request"]
        and x["release_verified"]
    )
    assert (
        not x["task_execution"]
        and x["api_requests"] == 2
        and x["successful_tool_calls"] == 0
    )
    assert len(x["private_archive_sha256"]) == 64
    assert [y["arm"] for y in x["rows"]] == ["native", "controller"]
    assert all(y["http_status"] == 400 and y["usage"] is None for y in x["rows"])
print(
    json.dumps(
        {
            "capsule_integrity": "PASS",
            "actual_swe_tasks_evaluated": 0,
            "repair_success_rate": None,
            "private_archives_reverified": False,
            "outcome": p["status"],
        }
    )
)
