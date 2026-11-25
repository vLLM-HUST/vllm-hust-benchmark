"""Verify prospective source identity and CPU preparation without model claims."""

import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def verify(folder):
    raw = (folder / "source-manifest.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "319045c9c2a67dc8025dbd6c84679837fe4ef1ad63310b3055cc7ac90be7ab81"
    )
    members = json.loads(raw)["files"]
    assert len(members) == 22
    for name, checksum in members.items():
        stored = (
            "original-README.md.source"
            if name == "README.md"
            else name + ".source"
            if name.endswith(".py")
            else name
        )
        assert hashlib.sha256((folder / stored).read_bytes()).hexdigest() == checksum
    plan = read(folder / "execution-plan.json")
    assert plan["scope"] == "supplemental_eos_historical_workload"
    assert plan["stages"] == ["native-r01", "bidkv-r01", "bidkv-r02", "native-r02"]
    assert plan["planned_total_requests"] == 48 and plan["requests_per_stage"] == 12
    assert plan["eos_preserved"] and plan["ignore_eos"] is False
    assert plan["output_budget_per_stage"] == sum(plan["max_tokens"]) == 24148
    assert plan["output_budget_total_upper"] == 96592
    assert plan["common_profile"]["kv_bytes_per_rank"] == 1677721600
    assert plan["concurrency"] == 12 and plan["request_rate"] == 4
    assert plan["clock"]["overall_limit_seconds"] == 9000
    assert not plan["clock"]["publication_parks"]
    assert not plan["clock"]["reset_or_extension"]
    assert plan["formal_five_dataset_tasks_budgets_and_graders_unchanged"]
    linux = read(folder / "linux-cpu-admission-v2.json")
    assert linux["actual_returncode"] == linux["model_requests"] == 0
    assert "38 passed" in linux["stdout"] and not linux["stderr"]
    assert linux["members_verified"] == 22
    assert linux["source_manifest_sha256"] == hashlib.sha256(raw).hexdigest()
    restore = read(folder / "cpu-restoration-summary.json")
    assert restore["phase"] == "passed-cpu-only-restoration"
    assert all(code == 0 for code in restore["actual_step_returncodes"])
    assert all(
        restore[key] == 0
        for key in (
            "model_startups",
            "generation_requests",
            "keepalive_jobs_created",
            "coder_policy_changes",
        )
    )
    return {"frozen_members": 22, "actual_linux_cpu_tests": 38, "model_result": False}


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).parent / "evoscientist-pressure02-v2")))
