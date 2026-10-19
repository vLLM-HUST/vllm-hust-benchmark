"""Public arithmetic/integrity checks only; private archive is not available here."""

from pathlib import Path
import hashlib
import json
import statistics

root = Path(__file__).parent
for line in (root / "SHA256SUMS").read_text().splitlines():
    expected, relative = line.split("  ", 1)
    path = Path(relative)
    assert not path.is_absolute() and ".." not in path.parts
    assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected
report = json.loads((root / "report.json").read_text())
config = json.loads((root / "source-and-config.json").read_text())
assert report["evidence_verification"] == "PASS"
assert not report["benefit_claim"] and not report["sensitive_raw_content_published"]
assert config["batch_size"] == 8 and config["arrival_time"] == 0
assert config["client_concurrency_limit"] is None
assert config["protocol"]["pair_order"] == [
    ["native", "controller"],
    ["controller", "native"],
    ["native", "controller"],
]
assert len(report["batches"]) == 6
assert sum(x["observed_terminal_tasks"] for x in report["batches"]) == 48
for arm in ["native", "controller"]:
    rows = [x for x in report["batches"] if x["arm"] == arm]
    summary = report["summary"][arm]
    assert len(rows) == summary["batches"] == 3
    assert summary["terminal_task_executions"] == 24
    assert summary["saved_model_responses"] == sum(x["model_responses"] for x in rows)
    assert summary["output_tokens"] == sum(
        x["usage"]["completion_tokens"] for x in rows
    )
    assert (
        abs(
            summary["median_agent_batch_seconds"]
            - statistics.median(x["agent_batch_elapsed_seconds"] for x in rows)
        )
        < 1e-8
    )
    events = report["logged_program_events"][arm]
    assert events["released"] == 24
    assert events["paused"] == events["marked_for_pause"] == events["resumed"] == 0
    for row in rows:
        assert row["observed_terminal_tasks"] == 8
        assert sum(row["terminal_statuses"].values()) == 8
        assert sum(row["response_finish_reasons"].values()) == row["model_responses"]
        assert row["original_grading_valid"] is False
        assert row["distinct_fresh_grader_containers"] == 8
        assert (
            row["usage"]["prompt_tokens"] + row["usage"]["completion_tokens"]
            == row["usage"]["total_tokens"]
        )
        assert row["price"] is None
        assert (
            abs(
                row["workflow_output_tokens_per_second"]
                - row["usage"]["completion_tokens"] / row["agent_batch_elapsed_seconds"]
            )
            < 1e-8
        )
        assert row["waiting_max"] == row["future_paused_tokens_max"] == 0
assert report["no_model_generation_during_regrade"]
assert (
    report["grading_window_model_token_counters"]["before"]
    == report["grading_window_model_token_counters"]["after"]
)
change = 100 * (
    report["summary"]["controller"]["median_agent_batch_seconds"]
    / report["summary"]["native"]["median_agent_batch_seconds"]
    - 1
)
assert abs(change - report["summary"]["elapsed_change_percent"]) < 1e-8
print(
    json.dumps(
        {
            "public_capsule_integrity_and_arithmetic": "PASS",
            "private_archive_reverified": False,
            "mod_benefit_established": False,
            "real_agent_task_executions": 48,
        }
    )
)
