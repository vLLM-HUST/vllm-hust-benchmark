from pathlib import Path
import hashlib
import json

root = Path(__file__).parent
for line in (root / "SHA256SUMS").read_text().splitlines():
    expected, relative = line.split("  ", 1)
    assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
    assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
r = json.loads((root / "report.json").read_text())
assert not r["performance_valid"] and not r["benefit_claim"]
assert (
    r["protocol"]["official_tasks"] * r["protocol"]["independent_agents_per_task"]
    == 256
)
assert (
    r["protocol"]["arrival_time"] == 0
    and r["protocol"]["client_concurrency_limit"] is None
)
assert r["capacity_input"]["usable_capacity"] == 808 * 2048
assert r["capacity_input"]["per_program_buffer"] == 6 * 2048 + 2047
for name in ["native", "controller"]:
    a = r["arms"][name]
    assert sum(a["statuses"].values()) == 256 and a["submitted"] == 0
    assert not any(a["engine_before"].values()) and not any(a["engine_after"].values())
    u = a["saved_usage"]
    assert u["prompt_tokens"] + u["completion_tokens"] == u["total_tokens"]
assert r["arms"]["native"]["kv_peak_fraction"] == 1
assert 0.65 < r["arms"]["controller"]["kv_peak_fraction"] < 0.66
assert r["arms"]["native"]["preemptions_delta"] == 13
assert r["arms"]["controller"]["preemptions_delta"] == 0
assert r["archive"]["files_checked"] == 7649 and r["archive"]["bytes"] == 38851627
assert (
    r["scoring"]["deferred_unresolved"] == 512 and not r["scoring"]["zero_score_claim"]
)
print(
    "PASS: public integrity/consistency; all Agents failed; no performance benefit established"
)
