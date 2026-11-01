"""Public orchestration helpers, no Docker/model calls in CI."""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location(
    "sab_public_scientific",
    Path(__file__).resolve().parents[1] / "scripts/run_scienceagentbench_scientific.py",
)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


@pytest.mark.parametrize(
    "path", ["../escape", "/absolute", "a/../../escape", "a\\escape"]
)
def test_asset_paths_stay_inside_declared_private_root(tmp_path, path):
    with pytest.raises(ValueError):
        entry.child_path(tmp_path, path)


def test_original_software_profile_is_complete_not_task_data():
    profile = json.loads(entry.PINS.read_text(encoding="utf-8"))
    assert len(profile["files"]) == 30
    assert sum(item["bytes"] for item in profile["files"]) == 206703
    assert {
        "agent.py",
        "compute_scores.py",
        "evaluation/harness/test_spec.py",
    }.issubset({item["path"] for item in profile["files"]})
    assert all(
        not item["path"].endswith((".csv", ".parquet", ".xlsx"))
        for item in profile["files"]
    )


def test_no_credentials_are_baked_into_original_recipe(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "owned-dummy-not-a-secret")
    seen = []

    def factory(platform, base, program):
        seen.append(os.environ.get("OPENAI_API_KEY"))
        return "ORIGINAL_RECIPE"

    module = SimpleNamespace(get_dockerfile_instance=factory)
    assert (
        entry.original_instance_recipe(
            module, "linux/arm64/v8", "owned/base:pin", "pred_case.py"
        )
        == "ORIGINAL_RECIPE"
    )
    assert seen == [None]
    assert (
        os.environ["OPENAI_API_KEY"]
        == "owned-dummy-not-a-secret"  # pragma: allowlist secret
    )
    with pytest.raises(ValueError):
        entry.original_instance_recipe(
            module, "linux/arm64/v8", "base\nRUN malicious", "pred.py"
        )


def test_wrong_docker_identity_starts_no_case(monkeypatch):
    monkeypatch.setattr(
        entry.subprocess, "check_output", lambda *args, **kwargs: '{"ID":"other"}'
    )
    args = SimpleNamespace(
        docker_host="unix:///private/owned.sock", docker_id="expected"
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        entry.run(args)


def test_saved_three_case_audit_does_not_claim_quality_or_source_delivery():
    report = json.loads(
        (
            entry.ROOT
            / "docs/workload-data/validation/scienceagentbench-fixed3-saved-audit-20261101.json"
        ).read_text(encoding="utf-8")
    )
    assert report["original_case_ids"] == [1, 2, 3]
    assert report["original_success"] == [0, 0, 0]
    assert report["audit_model_calls"] == report["audit_program_executions"] == 0
    assert report["source_checkbox_eligible"] is False and report["pending"]
    assert report["public_orchestrator_real_qualification"]["prepare_exit_code"] == 0
    assert report["public_orchestrator_real_qualification"]["run_exit_code"] == 0
