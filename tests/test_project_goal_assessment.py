"""Milestone attainment cannot silently become project-final attainment."""

import importlib.util
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "project_goals", ROOT / "scripts/project_goal_assessment.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_final90_is_independent_of_milestone50_and_retains_bound_scope():
    source = json.loads(
        (ROOT / "reports/m01-qualified-pair-20261103/summary.json").read_bytes()
    )
    data = json.loads(
        (
            ROOT / "reports/five-datasets-standard-20261113/summary-public-v2.json"
        ).read_bytes()
    )
    final = MODULE.assess(source, data["datasets"])
    target = final["targets"]["3.4.1"]
    assert target["threshold_percent"] == 90
    assert target["milestone_threshold_percent"] == 50
    assert target["attainment"] == "not_demonstrated"
    assert not final["acceptance_established"]


def test_all_four_final_criteria_and_confirmed_interpretations_are_explicit():
    data = json.loads(
        (
            ROOT / "reports/five-datasets-standard-20261113/summary-public-v2.json"
        ).read_bytes()
    )
    assert data["milestone_assessment"]["targets"] == data["formal_targets"]
    final = data["project_final_assessment"]
    assert set(final["targets"]) == set(data["formal_targets"])
    assert final["fully_specified"]
    assert final["targets"]["3.4.2"]["throughput_ratio_min"] == 2
    assert final["targets"]["3.4.2"]["mean_ttft_reduction_percent_min"] == 20
    assert final["targets"]["3.4.3"]["context_tokens_min"] == 32768
    assert final["targets"]["3.4.4"]["unit_token_cost_reduction_percent_min"] == 30
    interpretation = final["confirmed_interpretation"]
    assert interpretation["vendor_peak_is_not_used"]
    assert interpretation["model"] == "Qwen3.5-35B-A3B"
    assert interpretation["model_scope_explanation_already_confirmed"]
    assert data["merge_requirements"]["both_sections_present"]
    assert data["merge_requirements"]["final_goal_sources_complete"]
    assert not data["merge_requirements"]["ready_to_merge"]


def test_numeric_subset_success_does_not_hide_hle_or_claim_final_acceptance():
    data = json.loads(
        (
            ROOT / "reports/five-datasets-standard-20261113/summary-public-v2.json"
        ).read_bytes()
    )
    final = data["project_final_assessment"]
    performance = {
        r["dataset"]: r for r in final["targets"]["3.4.2"]["numeric_results_by_profile"]
    }
    assert performance["mmlu"]["numeric_threshold_met"]
    assert not performance["hle"]["numeric_threshold_met"]
    assert not final["acceptance_established"]
    assert not final["targets"]["3.4.3"]["continuous_32k_stability_verified"]
    assert not final["targets"]["3.4.4"]["multi_tenant_utilization_verified"]


def test_goal_source_fidelity_and_public_copy_are_explicit():
    path = ROOT / "reports/project-final-goals-20261113/final-goals-v1.json"
    profile = json.loads(path.read_bytes())
    source = profile["source"]
    assert not source["source_document_binary_available"]
    assert (
        hashlib.sha256((ROOT / source["source_excerpt"]).read_bytes()).hexdigest()
        == source["source_excerpt_sha256"]
    )
    assert (
        path.read_bytes()
        == (
            ROOT
            / "website-publication/data/five-datasets-standard-20261113/project-final-goals-v1.json"
        ).read_bytes()
    )


def test_new_numeric_results_do_not_inherit_old_hle_and_cost_diagnoses():
    source = json.loads(
        (ROOT / "reports/m01-qualified-pair-20261103/summary.json").read_bytes()
    )
    data = json.loads(
        (
            ROOT / "reports/five-datasets-standard-20261113/summary-public-v2.json"
        ).read_bytes()
    )
    for dataset in data["datasets"]:
        dataset["arms"]["B1"]["output_tokens_per_second"] = (
            3 * dataset["arms"]["B0"]["output_tokens_per_second"]
        )
        dataset["changes_percent"].update(
            mean_ttft_ms=-30, mean_tpot_ms=-30, unit_cost_change_percent=-60
        )
    final = MODULE.assess(source, data["datasets"])
    assert final["targets"]["3.4.2"]["attainment"] == "demonstrated"
    assert final["targets"]["3.4.2"]["all_profile_numeric_thresholds_met"]
    assert "50.24" not in final["targets"]["3.4.4"]["evidence_summary"]
    assert not final["acceptance_established"]  # 32K/multi-tenant remain separate.


def test_merge_readiness_uses_current_evidence_not_historical_failure_text():
    assert not MODULE.merge_requirements()["ready_to_merge"]
    ready = MODULE.merge_requirements(
        dict(
            execution_complete=True,
            raw_sources_verified=True,
            quality_denominators_accounted=True,
        )
    )
    assert ready["ready_to_merge"] and not ready["blockers"]
    assert ready["project_final_attainment_is_independent"]
