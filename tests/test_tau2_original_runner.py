"""CPU interface/safety checks, never native task execution evidence."""

from pathlib import Path
import importlib.util
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_telecom_delivery_uses_new_native_branch_and_fresh_client_evidence():
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )
    row = ledger["delivery_inventory"]["TAUBENCH"]
    assert row["checked"] and row["status"] == "AVAILABLE"
    native = json.loads((ROOT / row["action_branch_native_audit"]).read_text("utf-8"))
    clean = json.loads((ROOT / row["clean_installation_audit"]).read_text("utf-8"))
    delivery = json.loads((ROOT / row["delivery_receipt"]).read_text("utf-8"))
    assert native["verified_unique_original_tasks"] == 6
    assert native["historical_unique_original_tasks_reused"] == 3
    assert native["new_unique_original_tasks"] == 3
    assert all(
        task["official_full_reward_info_equal"] and task["original_task_equal"]
        for task in native["tasks"]
    )
    assert native["original_max_steps"] == 200 and native["original_max_errors"] == 10
    assert native["evaluation_type"] == "ALL" and native["both_native_roles_retained"]
    assert clean["installed_dependency_packages_equal"] == 125
    assert clean["original_python_files_equal"] == 89
    assert clean["new_clean_environment_install_verified"] and clean["pip_check_passed"]
    assert clean["fresh_client_saved_native_full_reward_info_rescore_equal"]
    assert clean["fresh_client_rescored_original_episodes"] == 3
    assert clean["fresh_client_rescore_model_calls"] == 0
    assert not clean["native_generation_rerun_in_fresh_client"]
    assert not delivery["full2285_native_execution_claim"]
    assert not delivery["other_tau2_domains_verified"]
    assert not delivery["model_quality_or_performance_claim"]


def test_all_native_reference_handoff_is_not_full_task_execution():
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )
    row = ledger["delivery_inventory"]["TAUBENCH"]
    receipt = json.loads(
        (ROOT / row["research_exact_artifact_handoff_receipt"]).read_text("utf-8")
    )
    assert receipt["original_native_references"] == 2285
    assert receipt["installed_provider_payload_files_equal"] == 32
    assert receipt["research_prepare_and_verify_both_used"]
    assert receipt["original_task_ids_order_positions_and_source_digests_equal"]
    assert receipt["new_native_tasks_or_model_calls"] == 0
    assert not receipt["full_telecom_native_execution_claim"]
    assert not receipt["research_observation_runtime_accepted"]
    assert not receipt["other_domains_verified"]


def test_full_source_preflight_and_snapshot_are_not_new_native_tasks():
    receipt = json.loads(
        (
            ROOT / "docs/workload-data/validation/tau2-readiness-cpu-20261024.json"
        ).read_text("utf-8")
    )
    assert receipt["original_telecom_tasks_preflighted"] == 2285
    assert receipt["original_native_references_prepared"] == 2285
    assert receipt["explicit_action_reference_preflight_tasks"] == 3
    assert receipt["explicit_action_reference_handoff_verified"]
    assert receipt["original_reward_basis_counts"] == {
        "ENV_ASSERTION": 2253,
        "ENV_ASSERTION,ACTION": 32,
    }
    assert receipt["dependency_snapshot_packages_equal"] == 125
    assert receipt["existing_environment_pip_check_passed"]
    assert not receipt["new_clean_environment_install_verified"]
    assert receipt["model_calls"] == receipt["new_native_tasks"] == 0
    assert not receipt["full_telecom_native_completed"]
    recipe = ROOT / "configs/workload-native/tau2-v020-py312-dependencies.txt"
    assert (
        hashlib.sha256(recipe.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        == receipt["dependency_snapshot_sha256"]
    )
    assert (
        hashlib.sha256(
            (ROOT / "scripts/run_tau2_original_telecom.py")
            .read_bytes()
            .replace(b"\r\n", b"\n")
        ).hexdigest()
        == receipt["current_runner_sha256"]
    )


def test_original_selection_keeps_official_order_and_never_invents_ids():
    spec = importlib.util.spec_from_file_location(
        "tau2_entry", ROOT / "scripts/run_tau2_original_telecom.py"
    )
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    tasks = [SimpleNamespace(id=str(index)) for index in range(5)]
    assert entry.select_official_tasks(tasks) == tasks[:3]
    assert entry.select_official_tasks(tasks, full=True) is tasks
    selected = entry.select_official_tasks(tasks, ["4", "1"])
    assert selected == [tasks[1], tasks[4]]
    assert selected[0] is tasks[1]
    for ids in (["1", "1"], ["unknown"]):
        with pytest.raises(ValueError):
            entry.select_official_tasks(tasks, ids)
    with pytest.raises(ValueError):
        entry.select_official_tasks(tasks, ["1"], full=True)


def test_original_action_basis_is_supported_without_changing_official_scorer():
    spec = importlib.util.spec_from_file_location(
        "tau2_reward_entry", ROOT / "scripts/run_tau2_original_telecom.py"
    )
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)

    def task(*values):
        return SimpleNamespace(
            evaluation_criteria=SimpleNamespace(
                reward_basis=[SimpleNamespace(value=value) for value in values]
            )
        )

    assert entry.supported_reward_bases(
        [task("ENV_ASSERTION"), task("ENV_ASSERTION", "ACTION")]
    ) == {"ENV_ASSERTION": 1, "ENV_ASSERTION,ACTION": 1}
    with pytest.raises(ValueError, match="do not drop"):
        entry.supported_reward_bases([task("NL_ASSERTION")])


def test_tau2_help_does_not_load_optional_environment():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_tau2_original_telecom.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--official-archive" in result.stdout


def test_tau2_rejects_remote_model_endpoint_before_dependency_or_data_access(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_tau2_original_telecom.py"),
            "--data-root",
            str(tmp_path / "missing-data"),
            "--official-archive",
            str(tmp_path / "missing.zip"),
            "--output",
            str(tmp_path / "output"),
            "--endpoint",
            "https://api.openai.com/v1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "only permits local HTTP endpoints" in result.stderr
    assert not (tmp_path / "output").exists()
