"""Check public integrity/arithmetic; private raw evidence is not public."""

from pathlib import Path
import hashlib
import json

root = Path(__file__).parent
for line in (root / "SHA256SUMS").read_text().splitlines():
    expected, relative = line.split("  ", 1)
    assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
    assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
report = json.loads((root / "report.json").read_text())
config = json.loads((root / "source-and-config.json").read_text())
assert (
    config["official_unique_tasks"] * config["independent_agents_per_task"]
    == config["program_instances"]
    == 256
)
assert config["arrival_time"] == 0 and config["client_concurrency_limit"] is None
assert not config["model_dependencies_changed"]
assert not report["benefit_claim"] and not report["completed_256_pair"]
assert not report["pressure_saturation_established"]
assert not report["sensitive_raw_content_published"]
prior = report["prior_read_only_diagnosis"]["batches"]
assert len(prior) == 6 and sum(x["observations"] for x in prior) == 2104
assert sum(x["observer_error_samples"] for x in prior) == 0
assert all(x["sampled_overflow_count"] == x["sampled_waiting_max"] == 0 for x in prior)
for row in prior:
    assert (
        abs(
            row["raw_capacity_over_engine_report_percent"]
            - 100
            * (row["declared_capacity"] / row["engine_reported_cache_size_tokens"] - 1)
        )
        < 1e-8
    )
cpu = report["cpu_model_free_preflight"]
assert cpu["verification"] == "PASS" and cpu["model_calls"] == 0
assert cpu["independent_containers"] == 128 and cpu["checked_files"] == 266
assert cpu["owned_preflight_containers_remaining"] == 0
start = report["actual_start"]
assert (
    start["containers"]
    == start["unique_containers"]
    == start["batch_start"]["batch_size"]
    == 256
)
assert start["graph_gate"]["status"] == "ACTUAL_REPLAY_OBSERVED"
assert all(v > 0 for v in start["graph_gate"]["delta"].values())
print(
    json.dumps(
        {
            "public_integrity_and_arithmetic": "PASS",
            "private_archive_reverified": False,
            "completed_256_pair": False,
            "mod_benefit_established": False,
        }
    )
)
