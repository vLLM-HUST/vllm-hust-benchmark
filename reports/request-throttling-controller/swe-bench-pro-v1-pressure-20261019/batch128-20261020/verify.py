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
    == 128
)
assert (
    r["protocol"]["arrival_time"] == 0
    and r["protocol"]["client_concurrency_limit"] is None
)
assert r["model"]["max_num_seqs"] == 256 and not r["model"]["restarted"]
assert r["arms"]["native"]["statuses"] == {"RetryError": 128}
assert r["arms"]["controller"]["statuses"] == {
    "RetryError": 118,
    "ContextWindowExceededError": 5,
    "LimitsExceeded": 5,
}
for arm in ["native", "controller"]:
    a = r["arms"][arm]
    assert a["submitted"] == 0 and sum(a["statuses"].values()) == 128
    assert a["preemptions_delta"] == 0 and a["kv_peak_fraction"] < 0.68
    assert not any(a["engine_before"].values()) and not any(a["engine_after"].values())
    assert (
        a["saved_usage"]["prompt_tokens"] + a["saved_usage"]["completion_tokens"]
        == a["saved_usage"]["total_tokens"]
    )
assert r["arms"]["native"]["exceptions"]["wall_budget"] == 128
assert r["arms"]["controller"]["exceptions"]["wall_budget"] == 118
assert r["step_limit_witness"]["controller_model_calls"] == [30] * 5
assert r["archive"]["bytes"] == 40209726 and r["archive"]["files_checked"] == 4338
assert (
    r["scoring"]["deferred_unresolved"] == 256 and not r["scoring"]["zero_score_claim"]
)
print("PASS:128-program failed-pair public integrity/consistency; no benefit claim")
