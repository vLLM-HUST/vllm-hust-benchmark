from pathlib import Path
import hashlib
import json

root = Path(__file__).parent
for line in (root / "SHA256SUMS").read_text().splitlines():
    expected, relative = line.split("  ", 1)
    assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
    assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
r = json.loads((root / "report.json").read_text())
assert not r["performance_comparison_valid"] and not r["benefit_claim"]
assert (
    r["protocol"]["official_tasks"] * r["protocol"]["independent_agents_per_task"]
    == 256
)
assert (
    r["protocol"]["arrival_time"] == 0
    and r["protocol"]["client_concurrency_limit"] is None
)
p = json.loads((root / "shared-pressure-profile.json").read_text())
assert (
    hashlib.sha256((root / "shared-pressure-profile.json").read_bytes()).hexdigest()
    == r["deployment"]["central_profile_sha256"]
)
assert (
    p["engine"]["max_num_seqs"] == 256
    and p["engine"]["kv_cache_memory_bytes"] == 17179869184
)
assert r["deployment"]["actual_rank_graph_replay_delta"] == {"0": 4, "1": 4}
for name in ["native", "controller"]:
    a = r["arms"][name]
    assert sum(a["statuses"].values()) == 256 and a["submitted"] == 0
    assert a["kv_peak_fraction"] > 0.99 and a["preemptions_delta"] > 0
    assert not any(a["engine_before"].values()) and not any(a["engine_after"].values())
    u = a["saved_usage"]
    assert u["prompt_tokens"] + u["completion_tokens"] == u["total_tokens"]
assert r["logged_events"]["controller"]["paused"] == 183
assert r["logged_events"]["controller"]["marked"] == 207
assert r["logged_events"]["controller"]["resumed"] == 181
assert (
    r["scoring"]["deferred_unresolved"] == 512 and not r["scoring"]["zero_score_claim"]
)
assert r["private_archive_verification"]["files_checked"] == 7672
assert r["capacity_disagreement"]["first_kv_100pct_remaining_capacity"] == 1155488
print(
    "PASS: public consistency and integrity; private evidence not re-audited; no performance benefit established"
)
