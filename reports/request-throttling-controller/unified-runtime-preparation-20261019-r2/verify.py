from pathlib import Path
import hashlib
import json

root = Path(__file__).parent
for line in (root / "SHA256SUMS").read_text().splitlines():
    digest, name = line.split("  ", 1)
    assert Path(name).name == name
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
p = json.loads((root / "report.json").read_text())
assert p["job_status"] == "SUCCEEDED" and p["official_check_exit"] == 0
assert p["verified_files"] == 4695 and p["original_model_health_after"] == 200
assert p["temporary_auth_file_removed"]
assert not p["original_model_restarted"] and not p["existing_runs_migrated"]
assert not p["frozen_dependency_or_test_parameter_changes"]
assert not p["new_model_or_tensor_execution"]
assert len(p["loaded_source_files"]) == 4
assert all(len(x["sha256"]) == 64 for x in p["loaded_source_files"].values())
assert (
    p["source_commits"][0]["commit"] == "ce5e49034a40d34fe88d78e4707c629449c3a5a2"
)  # pragma: allowlist secret
assert (
    p["source_commits"][1]["commit"] == "a9dd32033c5669ba28976c9d208a69d762796ba9"
)  # pragma: allowlist secret
assert (
    p["central_validation"]["shared_profile_sha256"]
    == "5792631172af883cf247e05b214444c02a106831aa50162b8c6f3a0e63f24661"  # pragma: allowlist secret
)
print(
    json.dumps(
        {
            "capsule_integrity": "PASS",
            "official_bootstrap": "PASS",
            "model_performance_measured": False,
        }
    )
)
