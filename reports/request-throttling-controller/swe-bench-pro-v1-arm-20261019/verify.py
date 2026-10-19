from pathlib import Path
import hashlib
import json

root = Path(__file__).parent
for line in (root / "SHA256SUMS").read_text().splitlines():
    digest, name = line.split("  ", 1)
    assert Path(name).name == name
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
report = json.loads((root / "report.json").read_text())
config = json.loads((root / "source-and-config.json").read_text())
assert report["evidence_verification"] == "PASS"
assert report["agent_status"] == "LimitsExceeded"
assert not report["agent_submitted"] and report["model_calls"] == 30
assert report["patch_bytes"] == 10015 and report["grade_cli_exit"] == 0
assert list(report["official_result"].values()) == [False]
assert report["parsed_test_records"] == 0 and report["required_names"] == 681
assert report["usage"] == {
    "prompt_tokens": 411016,
    "completion_tokens": 8308,
    "total_tokens": 419324,
}
assert report["finish_reasons"] == {"stop": 30}
assert (
    report["separate_agent_and_grader_containers"]
    and report["task_containers_after"] == 0
)
assert report["official_parser_byte_replay"] == "PASS"
assert not report["quality_or_performance_claim"]
assert report["dollar_cost"] == "NOT_MEASURED"
assert config["official_core_modified"] is False and config["agent_step_limit"] == 30
runtime = json.loads((root / "runtime-preparation.json").read_text())
assert runtime["source_recheck"]["verified_files"] == 4695
assert runtime["cpu_bootstrap"] == "FAILED"
assert not runtime["dependencies_or_test_parameters_changed"]
assert not runtime["existing_runs_migrated"]
assert (
    runtime["private_github_authenticated"] and runtime["temporary_auth_file_removed"]
)
print(
    json.dumps(
        {
            "capsule_integrity": "PASS",
            "repair_success": False,
            "mod_performance_measured": False,
            "private_archive_reverified": False,
        }
    )
)
