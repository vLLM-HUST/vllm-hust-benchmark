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
