"""Preparation evidence must never promote ScienceAgentBench to delivery."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/workload-data"


def test_dependency_audit_preserves_original_scoring_and_pending_delivery():
    audit_path = "validation/scienceagentbench-original-library-audit-20261101.json"
    audit = json.loads((DOCS / audit_path).read_text(encoding="utf-8"))
    assert audit["source_id"] == "SCIENCEAGENTBENCH"
    assert audit["source_checkbox_eligible"] is False
    assert audit["original_tasks_completed"] == audit["model_calls"] == 0
    assert audit["generated_code_executed"] is False
    prompt = audit["original_baseline_prompt_control"]
    assert prompt["original_rows"] == 102
    assert prompt["maximum_chat_tokens"] < prompt["input_budget"]
    assert prompt["over_budget_cases"] == []
    assert prompt["identity_trim_control_not_original_litellm_execution"] is True
    assert prompt["model_calls"] == 0
    metric = audit["code_bert_score"]
    assert metric["version"] == "0.4.1"
    assert metric["python_model"] == "neulab/codebert-python"
    assert metric["python_num_layers"] == 11
    assets = metric["model_assets"]
    assert assets["files_verified"] == 8
    assert assets["official_git_and_lfs_identities_verified"] is True
    assert assets["bytes_verified"] == 502184777
    assert assets["model_loaded"] is False
    assert assets["neural_execution_verified"] is False
    assert assets["raw_windows_disk_written"] is False
    assert metric["npu_model_binding_verified"] is False
    runtime = metric["isolated_runtime"]
    assert runtime["distribution_version"] == "0.4.1"
    assert runtime["original_internal_version"] == "0.3.5"
    assert runtime["source_files_verified_against_original_sdist"] == 4
    assert runtime["pip_check_exit_code"] == runtime["import_exit_code"] == 0
    assert runtime["npu_initialized"] is False
    assert runtime["model_loaded"] is False
    assert metric["scoring_metric_omitted"] is False
    assert audit["litellm"]["substitute_installed"] is False
    assert audit["litellm"]["fabricated_token_prices"] is False
    ledger = json.loads(
        (DOCS / "dataset-validation-status.json").read_text(encoding="utf-8")
    )
    entry = ledger["delivery_inventory"]["SCIENCEAGENTBENCH"]
    assert (
        entry["current_dependency_executor_audit"] == "docs/workload-data/" + audit_path
    )
    final = json.loads((ROOT / entry["delivery_receipt"]).read_text(encoding="utf-8"))
    assert entry["checked"] is True and final["source_checkbox_eligible"] is True
    assert final["native_evidence"] != "docs/workload-data/" + audit_path


def test_owned_executor_control_does_not_claim_scientific_environment():
    audit = json.loads(
        (
            DOCS / "validation/scienceagentbench-original-library-audit-20261101.json"
        ).read_text(encoding="utf-8")
    )
    executor = audit["owned_cpu_executor"]
    assert executor["isolated_read_only_network_none_control_exit_code"] == 0
    assert executor["persistent_volume_capacity_gib"] == 80
    for field in (
        "persistent_volume_expanded",
        "independent_docker_resized_or_reformatted",
        "default_docker_modified_or_restarted",
        "native_scientific_environment_built",
        "existing_data_deleted",
    ):
        assert executor[field] is False
