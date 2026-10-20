from pathlib import Path
import hashlib
import json

root = Path(__file__).parent
for line in (root / "SHA256SUMS").read_text().splitlines():
    expected, relative = line.split("  ", 1)
    assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
    assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
r = json.loads((root / "report.json").read_text())
p = json.loads((root / "agent-http-profile.json").read_text())
assert r["status"] == "NATIVE_RUNNING" and r["user_approved"]
assert r["performance_validity"] == "UNFINISHED_NO_GAIN_CLAIM"
assert p["timeouts"] == {"connect": 180, "read": None, "write": 180, "pool": 180}
assert (
    hashlib.sha256((root / "agent-http-profile.json").read_bytes()).hexdigest()
    == r["agent_http_profile_sha256"]
)
assert r["native_unique_containers"] == 256 and not r["controller_started"]
assert (
    r["native_batch_start"]["arrival_time"] == 0
    and r["native_batch_start"]["client_concurrency_limit"] is None
)
assert r["warmup_graph"]["delta"] == {"0": 4, "1": 4}
assert r["http_profile_gate"]["official_model_config_json_serializable"]
assert (
    r["http_profile_gate"]["network_connections"]
    == r["http_profile_gate"]["model_requests"]
    == 0
)
assert (
    r["per_agent_receipts"]["native_files"]
    == r["per_agent_receipts"]["json_valid_live_trajectories"]
    == 256
)
assert r["per_agent_receipts"]["observer_errors"] == 0
assert r["tests"]["full_passed"] == 151 and r["tests"]["full_skipped"] == 4
assert not r["model_restarted"]
print(
    "PASS: approved read-timeout configuration and real start consistency; no terminal benefit claim"
)

t = json.loads((root / "terminal-report.json").read_text())
assert not t["performance_valid"] and not t["benefit_claim"]
assert t["timeouts"] == {"connect": 180, "read": None, "write": 180, "pool": 180}
assert t["protocol"]["tasks"] * t["protocol"]["replicas"] == 256
assert (
    t["protocol"]["arrival_time"] == 0
    and t["protocol"]["client_concurrency_limit"] is None
)
assert not t["model_restarted"] and t["scoring"]["deferred"] == 512
for arm in ["native", "controller"]:
    a = t["arms"][arm]
    assert a["submitted"] == 0 and a["statuses"] == {"RetryError": 256}
    assert a["http_receipts_read_null"] == 256
    assert not any(a["engine_before"].values()) and not any(a["engine_after"].values())
assert t["arms"]["native"]["terminal_exception_classification"] == {
    "wall_budget_1200_mapped_as_timeout": 256
}
assert t["arms"]["controller"]["terminal_exception_classification"] == {
    "wall_budget_1200_mapped_as_timeout": 254,
    "remote_disconnect": 2,
}
assert t["arms"]["native"]["preemptions_delta"] == 14
assert t["arms"]["controller"]["preemptions_delta"] == 0
assert t["archive"]["files_checked"] == 8183 and t["archive"]["bytes"] == 52154730
print(
    "PASS: terminal budget-alarm classification and failed-pair consistency; no benefit claim"
)
