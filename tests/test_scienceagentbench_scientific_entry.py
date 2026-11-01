"""Public orchestration helpers, no Docker/model calls in CI."""

import importlib.util
import hashlib
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


def test_actual_ci_handoff_preserves_every_original_row():
    report = json.loads(
        (
            entry.ROOT
            / "docs/workload-data/validation/scienceagentbench-current-ci-research-handoff-20261101.json"
        ).read_text(encoding="utf-8")
    )
    assert report["complete_original_fields_per_row"] == 12
    assert report["complete_original_rows_roundtripped"] == 102
    assert report["public_research_readbacks"] == 102
    assert report["installed_git_payloads_verified"] == 78
    assert report["original_index_unchanged"]
    assert report["three_native_entries_discoverable"]
    assert not report["research_modified_or_observation_started"]
    assert report["new_model_calls"] == 0


def test_final_delivery_requires_native_execution_and_actual_handoff():
    docs = entry.ROOT / "docs/workload-data"
    final = json.loads(
        (
            docs / "validation/scienceagentbench-fixed3-native-delivery-20261101.json"
        ).read_text(encoding="utf-8")
    )
    native = json.loads(
        (entry.ROOT / final["native_evidence"]).read_text(encoding="utf-8")
    )
    handoff = json.loads(
        (entry.ROOT / final["research_handoff_evidence"]).read_text(encoding="utf-8")
    )
    assert final["source_checkbox_eligible"] is True
    assert final["qualified_source_commit"] == handoff["source_commit"]
    assert final["qualified_cpu_smoke_run"] == handoff["run_id"]
    assert final["original_instance_ids"] == native["original_case_ids"] == [1, 2, 3]
    assert final["original_codebert_scores"] == native["original_codebert_scores"]
    assert final["original_success"] == native["original_success"] == [0, 0, 0]
    assert (
        final["original_rows_roundtripped_and_public_research_readbacks"]
        == handoff["public_research_readbacks"]
        == 102
    )
    assert final["remaining_modes"] and not final["quality_or_engine_performance_claim"]
    code = (entry.ROOT / final["scientific_entry"]).read_text(encoding="utf-8")
    assert (
        hashlib.sha256(code.encode()).hexdigest()
        == final["public_scientific_orchestrator_git_sha256"]
    )
