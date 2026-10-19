"""Public integrity and failure-boundary arithmetic; no private raw audit here."""

from pathlib import Path
import hashlib
import json

root = Path(__file__).parent
for line in (root / "SHA256SUMS").read_text().splitlines():
    expected, relative = line.split("  ", 1)
    assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
    assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
r = json.loads((root / "report.json").read_text())
assert r["status"] == "INDEPENDENT_PRIVATE_ARCHIVE_VERIFIED_FAILURE"
assert r["checked_files"] == 7603 and r["archive_bytes"] == 44846170
assert not r["benefit_claim"] and not r["valid_successful_work_comparison"]
assert not r["sensitive_raw_content_published"]
assert sum(a["terminal_tasks"] for a in r["arms"].values()) == 512
for a in r["arms"].values():
    assert (
        a["timeout_tasks"] == 256
        and a["agent_submitted"] == 0
        and a["unresolved_grades"] == 256
    )
    assert a["terminal_statuses"] == {"RetryError": 256}
    assert a["kv_occupancy_max"] < 0.16 and a["kv_samples_at_or_above_90pct"] == 0
    assert a["preemptions_counter_delta"] == 0 and a["program_release_receipts"] == 256
    u = a["saved_response_usage"]
    assert u["prompt_tokens"] + u["completion_tokens"] == u["total_tokens"]
assert r["arms"]["controller"]["initial_running_requests"] == 16
assert r["arms"]["controller"]["initial_waiting_requests"] == 25
assert r["logged_program_events"]["controller"]["paused"]["events"] == 418
assert r["logged_program_events"]["controller"]["marked_for_pause"]["events"] == 472
assert r["logged_program_events"]["controller"]["resumed"]["events"] == 406
assert (
    abs(
        r["elapsed_error_window_change_percent"]
        - 100
        * (
            r["arms"]["controller"]["elapsed_until_error_seconds"]
            / r["arms"]["native"]["elapsed_until_error_seconds"]
            - 1
        )
    )
    < 1e-8
)
print(
    json.dumps(
        {
            "public_integrity_and_arithmetic": "PASS",
            "private_archive_reverified": False,
            "performance_comparison_valid": False,
            "mod_benefit_established": False,
        }
    )
)
