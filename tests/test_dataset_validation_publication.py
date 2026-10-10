from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import jsonschema
import pytest

from vllm_hust_benchmark.dataset_validation import (
    DatasetValidationError,
    validate_artifact,
    validate_index,
    validate_publication,
)


ROOT = Path(__file__).resolve().parents[1]
PUBLICATION = ROOT / "leaderboard-data" / "dataset-validation"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_checked_in_publication_is_valid_and_complete() -> None:
    assert validate_publication(PUBLICATION) == {"scenarios": 7, "results": 729}


def test_dataset_program_separates_primary_plan_from_measured_scenarios() -> None:
    index = load_json(PUBLICATION / "dataset_validation_index_v1.json")
    program = load_json(PUBLICATION / index["program_file"])
    assert program["test_plan"]["test_plan_version"] == "V5.4"
    assert program["test_plan"]["mandatory_model"] == "Qwen/Qwen3.5-35B-A3B"
    assert program["test_plan"]["formal_precision"] == "BF16"
    assert program["test_plan"]["baseline_roles"] == ["B0", "B1"]
    assert "NOT_EXECUTED_OPTIONAL" in program["test_plan"]["optional_status"]
    assert "BF16" in program["test_plan"]["evidence_boundary"]
    assert [dataset["label"] for dataset in program["primary_datasets"]] == [
        "MMLU-Pro",
        "HLE-Verified",
        "SWE-bench-Pro",
        "FrontierScience",
        "Terminal-Bench 2.1",
    ]
    assert all(dataset["primary_metric_zh"] for dataset in program["primary_datasets"])
    assert program["designation"]["label_zh"] == "浦江指定数据集"
    assert program["designation"]["scope_status"] == "names-only"
    assert program["designation"]["historical_b0_boundary"] == {
        "workbook_dataset_count": 21,
        "covered_designated_dataset_ids": ["mmlu-pro"],
        "not_covered_designated_dataset_ids": [
            "hle-verified",
            "swe-bench-pro",
            "frontierscience",
            "terminal-bench-2.1",
        ],
        "note": "The historical 21-dataset B0 workbook contains serving telemetry for MMLU-Pro only. It does not cover the other four designated datasets and does not provide a task-quality score for MMLU-Pro.",
        "note_zh": "历史 21 数据集 B0 工作簿只包含 MMLU-Pro 的服务遥测；另外四项不在其中，且该 MMLU-Pro 单元也不是任务质量成绩。",
    }
    readiness = {
        dataset["id"]: dataset["readiness"]["status"]
        for dataset in program["primary_datasets"]
    }
    assert readiness == {
        "mmlu-pro": "asset-frozen",
        "hle-verified": "asset-frozen",
        "swe-bench-pro": "asset-frozen",
        "frontierscience": "asset-frozen",
        "terminal-bench-2.1": "asset-frozen",
    }
    mmlu = program["primary_datasets"][0]["readiness"]
    assert mmlu["dataset_revision"] == (
        "b189ec765aa7ed75c8acfea42df31fdae71f97be"  # pragma: allowlist secret
    )
    assert mmlu["sample_count"] is None
    assert (
        mmlu["manifest_sha256"]
        == (
            "2666f482a6f6c974cde19b6549f72ee37df82a93c9dc032754005af3a839299b"  # pragma: allowlist secret
        )
    )
    assert mmlu["scorer"] is None
    assert (
        {
            dataset["id"]: dataset["readiness"]["dataset_revision"]
            for dataset in program["primary_datasets"]
        }
        == {
            "mmlu-pro": "b189ec765aa7ed75c8acfea42df31fdae71f97be",  # pragma: allowlist secret
            "hle-verified": "b705e0fb541c025a1532ce0d60d70ae2f53b00e0",  # pragma: allowlist secret
            "swe-bench-pro": "2d52cb3df914a3fcf80c7f66738b3a88ae37fc50",  # pragma: allowlist secret
            "frontierscience": "25ed67db7da8f4591484e764008ff585544f5a30",  # pragma: allowlist secret
            "terminal-bench-2.1": "7131e4375048a0e408a8fb404b5f499d726b695b",  # pragma: allowlist secret
        }
    )
    assert program["supplementary_material"]["default_tier"] == "supplementary"
    assert program["supplementary_material"]["classification_rule"] == (
        "all-other-registered-or-planned-datasets"
    )

    scenario_datasets = set()
    for scenario in index["scenarios"]:
        artifact = load_json(PUBLICATION / scenario["data_file"])
        scenario_datasets.update(dataset["id"] for dataset in artifact["datasets"])
    assert "mmlu-pro" in scenario_datasets
    assert {
        "hle-verified",
        "swe-bench-pro",
        "frontierscience",
        "terminal-bench-2.1",
    }.isdisjoint(scenario_datasets)


def test_default_scenario_is_paired_qwen35_frontier_data() -> None:
    index = load_json(PUBLICATION / "dataset_validation_index_v1.json")
    scenario = next(
        item
        for item in index["scenarios"]
        if item["id"] == index["default_scenario_id"]
    )
    assert scenario["data_file"] == (
        "dataset_validation_qwen35_frontier_unified_900s.json"
    )

    artifact = load_json(PUBLICATION / scenario["data_file"])
    assert len(artifact["results"]) == 10
    assert all(result["status"] == "passed" for result in artifact["results"])
    assert all(result["baseline_value"] is not None for result in artifact["results"])
    assert all(result["value"] is not None for result in artifact["results"])
    assert all(result["candidate_values"] for result in artifact["results"])


def test_selector_prioritizes_results_and_hides_coverage_planning() -> None:
    index = load_json(PUBLICATION / "dataset_validation_index_v1.json")
    visible = [
        scenario
        for scenario in index["scenarios"]
        if scenario.get("selector_visible", True)
    ]
    assert [scenario["id"] for scenario in visible[:4]] == [
        "qwen35-35b-a3b-bf16-tp2-pp1-dp1-ep-off-ctx262k-apc-on-mtp2-full-piecewise-sweprefix-900s",
        "qwen35-35b-a3b-bf16-tp2-pp1-dp1-ep-off-ctx262k-apc-on-mtp2-native-full-piecewise-vs-betterscale-full-e16-r20-sweprefix-900s",
        "qwen35-35b-a3b-bf16-tp2-pp2-dp1-ep-off-ctx262k-apc-on-mtp2-full-piecewise-pipeline-microbatch-sweprefix-900s",
        "qwen35-35b-a3b-bf16-tp4-c12-kv512m",
    ]
    assert all(scenario["label"].startswith("Paired B0/B1") for scenario in visible[:4])
    planning = next(
        scenario
        for scenario in index["scenarios"]
        if scenario["id"] == "qwen35-35b-a3b-bf16-tp2-dataset-matrix-v1"
    )
    assert planning["selector_visible"] is False


def test_qwen35_frontier_b1_retains_all_admitted_candidates() -> None:
    artifact = load_json(
        PUBLICATION / "dataset_validation_qwen35_frontier_unified_900s.json"
    )
    assert len(artifact["datasets"]) == 5
    assert len(artifact["metrics"]) == 2
    assert len(artifact["results"]) == 10
    assert artifact["scenario"]["expert_parallel"] is False
    assert artifact["scenario"]["prefix_caching"] is True
    assert artifact["scenario"]["measurement_seconds"] == 900
    assert artifact["baseline"]["id"] == "swe-unified-native-20260927"
    assert all(len(result["candidate_values"]) == 6 for result in artifact["results"])
    assert {
        candidate["candidate_id"]
        for candidate in artifact["results"][0]["candidate_values"]
    } == {
        "bidkv",
        "dla",
        "kv-materialization-arrival-control",
        "kv-tiering-migration",
        "kvcompress-ascend",
        "pegaflow-vllm-connectors",
    }
    c1_output = next(
        result
        for result in artifact["results"]
        if result["dataset_id"] == "swe-prefix-reuse-c1"
        and result["metric_id"] == "output_token_throughput"
    )
    assert c1_output["selected_candidate_id"] == "kvcompress-ascend"
    assert c1_output["value"] == 94.57222222222222
    c16_output = next(
        result
        for result in artifact["results"]
        if result["dataset_id"] == "swe-prefix-reuse-c16"
        and result["metric_id"] == "output_token_throughput"
    )
    assert c16_output["selected_candidate_id"] == "pegaflow-vllm-connectors"
    assert c16_output["value"] == 459.71555555555557


def test_qwen35_frontier_b1_rebuild_is_reproducible() -> None:
    script = ROOT / "scripts" / "build_qwen35_frontier_b1.py"
    spec = importlib.util.spec_from_file_location("frontier_b1_builder", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checked_in = load_json(
        PUBLICATION / "dataset_validation_qwen35_frontier_unified_900s.json"
    )
    assert module.build() == checked_in
    checked_in_betterscale = load_json(
        PUBLICATION / "dataset_validation_qwen35_frontier_betterscale_900s.json"
    )
    assert module.build_betterscale() == checked_in_betterscale
    checked_in_pipeline = load_json(
        PUBLICATION / "dataset_validation_qwen35_frontier_pipeline_pp2_900s.json"
    )
    assert module.build_pipeline() == checked_in_pipeline


def test_remaining_frontier_pairs_preserve_gains_and_regressions() -> None:
    betterscale = load_json(
        PUBLICATION / "dataset_validation_qwen35_frontier_betterscale_900s.json"
    )
    assert betterscale["baseline"]["id"] == "swe-capacity16-native"
    assert betterscale["scenario"]["baseline_graph_mode"] == "FULL_AND_PIECEWISE"
    assert betterscale["scenario"]["candidate_graph_mode"] == "FULL"
    assert {cell["selected_candidate_id"] for cell in betterscale["results"]} == {
        "betterscale"
    }
    assert {cell["comparison"]["trend"] for cell in betterscale["results"]} == {
        "improved"
    }

    pipeline = load_json(
        PUBLICATION / "dataset_validation_qwen35_frontier_pipeline_pp2_900s.json"
    )
    assert pipeline["baseline"]["id"] == "swe-k8s-pp2-20260925-nativepp-r1"
    assert pipeline["scenario"]["pipeline_parallel_size"] == 2
    assert pipeline["scenario"]["hardware"] == "4× Ascend 910B2"
    assert {cell["selected_candidate_id"] for cell in pipeline["results"]} == {
        "pipeline-microbatch-migration"
    }
    assert {cell["comparison"]["trend"] for cell in pipeline["results"]} == {
        "improved",
        "regressed",
    }
    assert all(
        cell["candidate_values"][0]["runtime_effectiveness"] == "exercised"
        for cell in pipeline["results"]
    )


def test_qwen35_workbook_b0_is_published_with_repaired_metadata() -> None:
    artifact = load_json(
        PUBLICATION / "dataset_validation_qwen35_tp2_ep_ctx32k_apcoff_inf_out256.json"
    )
    assert len(artifact["datasets"]) == 21
    assert len(artifact["metrics"]) == 6
    assert len(artifact["results"]) == 126
    assert {result["status"] for result in artifact["results"]} == {"baseline_only"}
    assert all(result["baseline_value"] is not None for result in artifact["results"])
    assert artifact["scenario"]["model"] == "Qwen3.5-35B-A3B"
    assert artifact["scenario"]["expert_parallel"] is True
    assert artifact["scenario"]["prefix_caching"] is False
    assert artifact["scenario"]["request_rate"] == "inf"

    cells = {
        (result["dataset_id"], result["metric_id"]): result
        for result in artifact["results"]
    }
    assert cells[("jsonschemabench", "request_throughput")]["baseline_value"] == 1.8
    assert cells[("longbench", "request_throughput")]["baseline_value"] == 1.21
    assert cells[("jsonschemabench", "request_success_rate")]["baseline_value"] == 99.5
    assert cells[("longbench-v2", "request_success_rate")]["baseline_value"] == 99.0
    assert all(
        result["provenance"]["result_json_sha256"] for result in artifact["results"]
    )


def test_qwen35_workbook_b0_rebuild_is_reproducible() -> None:
    script = ROOT / "scripts" / "build_dataset_matrix_coverage.py"
    spec = importlib.util.spec_from_file_location(
        "dataset_matrix_workbook_builder", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    checked_in = load_json(
        PUBLICATION / "dataset_validation_qwen35_tp2_ep_ctx32k_apcoff_inf_out256.json"
    )
    assert module.build_qwen35_workbook() == checked_in


def test_publication_documents_match_json_schemas() -> None:
    index_schema = load_json(
        ROOT / "schemas" / "dataset_validation_index_v1.schema.json"
    )
    artifact_schema = load_json(ROOT / "schemas" / "dataset_validation_v1.schema.json")
    program_schema = load_json(ROOT / "schemas" / "dataset_program_v1.schema.json")
    index = load_json(PUBLICATION / "dataset_validation_index_v1.json")
    jsonschema.Draft202012Validator(index_schema).validate(index)
    jsonschema.Draft202012Validator(program_schema).validate(
        load_json(PUBLICATION / index["program_file"])
    )
    for scenario in index["scenarios"]:
        jsonschema.Draft202012Validator(artifact_schema).validate(
            load_json(PUBLICATION / scenario["data_file"])
        )


def test_dataset_program_never_publishes_host_absolute_paths() -> None:
    program = load_json(PUBLICATION / "dataset_program_v1.json")
    paths = [
        material["path"]
        for dataset in program["primary_datasets"]
        for material in dataset["readiness"]["observed_materials"]
    ]
    assert paths
    assert all(not path.startswith("/") for path in paths)


def test_index_rejects_duplicate_scenarios_and_consumer_urls() -> None:
    index = load_json(PUBLICATION / "dataset_validation_index_v1.json")
    duplicate = copy.deepcopy(index)
    duplicate["scenarios"].append(copy.deepcopy(duplicate["scenarios"][0]))
    with pytest.raises(DatasetValidationError, match="duplicate scenario"):
        validate_index(duplicate)

    website_path = copy.deepcopy(index)
    website_path["scenarios"][0]["data_file"] = "./data/result.json"
    with pytest.raises(DatasetValidationError, match="data_file"):
        validate_index(website_path)


def test_artifact_rejects_scenario_drift_duplicate_cells_and_missing_evidence() -> None:
    artifact = load_json(PUBLICATION / "dataset_validation_qwen35_bidkv.json")
    with pytest.raises(DatasetValidationError, match="does not match"):
        validate_artifact(artifact, expected_scenario_id="another-scenario")

    duplicate = copy.deepcopy(artifact)
    duplicate["results"].append(copy.deepcopy(duplicate["results"][0]))
    with pytest.raises(DatasetValidationError, match="duplicate result cell"):
        validate_artifact(duplicate, expected_scenario_id=artifact["scenario"]["id"])

    missing_evidence = copy.deepcopy(artifact)
    missing_evidence["results"][0]["provenance"].pop("artifact")
    missing_evidence["results"][0]["provenance"].pop("report_url")
    with pytest.raises(DatasetValidationError, match="evidence URL"):
        validate_artifact(
            missing_evidence, expected_scenario_id=artifact["scenario"]["id"]
        )

    nullable_current_value = copy.deepcopy(artifact)
    nullable_current_value["results"][0]["current_value"] = None
    nullable_current_value["results"][0].pop("provenance")
    with pytest.raises(DatasetValidationError, match="lacks provenance"):
        validate_artifact(
            nullable_current_value,
            expected_scenario_id=artifact["scenario"]["id"],
        )


def test_artifact_requires_explicit_applicability_states_and_matched_b0() -> None:
    artifact = load_json(PUBLICATION / "dataset_validation_qwen35_bidkv.json")

    missing_applicability = copy.deepcopy(artifact)
    missing_applicability["datasets"][0].pop("applicable_metric_ids")
    with pytest.raises(DatasetValidationError, match="applicable_metric_ids"):
        validate_artifact(
            missing_applicability,
            expected_scenario_id=artifact["scenario"]["id"],
        )

    unmatched_b1 = copy.deepcopy(artifact)
    unmatched_b1["results"][0]["baseline_value"] = None
    with pytest.raises(DatasetValidationError, match="matched B0"):
        validate_artifact(unmatched_b1, expected_scenario_id=artifact["scenario"]["id"])

    missing_cell = copy.deepcopy(artifact)
    missing_cell["results"].clear()
    with pytest.raises(DatasetValidationError, match="explicitly cover"):
        validate_artifact(missing_cell, expected_scenario_id=artifact["scenario"]["id"])


def test_artifact_validates_full_candidate_sets() -> None:
    artifact = load_json(
        PUBLICATION / "dataset_validation_qwen35_frontier_unified_900s.json"
    )
    duplicate = copy.deepcopy(artifact)
    duplicate["results"][0]["candidate_values"].append(
        copy.deepcopy(duplicate["results"][0]["candidate_values"][0])
    )
    with pytest.raises(DatasetValidationError, match="duplicate candidate_id"):
        validate_artifact(duplicate, expected_scenario_id=artifact["scenario"]["id"])

    drift = copy.deepcopy(artifact)
    drift["results"][0]["value"] += 1
    with pytest.raises(DatasetValidationError, match="differs from selected candidate"):
        validate_artifact(drift, expected_scenario_id=artifact["scenario"]["id"])

    non_maximum = copy.deepcopy(artifact)
    cell = non_maximum["results"][0]
    lower_candidate = min(cell["candidate_values"], key=lambda item: item["value"])
    cell["selected_candidate_id"] = lower_candidate["candidate_id"]
    cell["value"] = lower_candidate["value"]
    cell["provenance"] = copy.deepcopy(lower_candidate["provenance"])
    with pytest.raises(DatasetValidationError, match="not the maximum candidate value"):
        validate_artifact(
            non_maximum,
            expected_scenario_id=artifact["scenario"]["id"],
        )

    reversed_candidates = copy.deepcopy(artifact)
    for result in reversed_candidates["results"]:
        result["candidate_values"].reverse()
    validate_artifact(
        reversed_candidates,
        expected_scenario_id=artifact["scenario"]["id"],
    )


def test_matrix_rebuild_preserves_measured_szyn_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = ROOT / "scripts" / "build_dataset_matrix_coverage.py"
    spec = importlib.util.spec_from_file_location("dataset_matrix_builder", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    existing = load_json(PUBLICATION / "dataset_validation_qwen35_tp2_matrix.json")
    qwen25 = load_json(PUBLICATION / "dataset_validation_v1.b0.json")
    cell = next(
        result
        for result in existing["results"]
        if result["dataset_id"] == "szyn-opencode-swebench-verified-500"
        and result["metric_id"] == "agent_resolve_rate"
    )
    cell.update(
        status="baseline_only",
        baseline_value=42.0,
        value=None,
        provenance={"repository": "vLLM-HUST/vllm-hust-dev-hub"},
    )
    existing["generated_at"] = "2026-10-06T00:00:00Z"
    existing["status"] = "partial_measurements"
    existing["baseline"]["commit"] = "measured-commit"

    monkeypatch.setattr(
        module, "load", lambda name: existing if name == module.QWEN35_FILE else None
    )
    rebuilt = module.build_qwen35(qwen25)
    rebuilt_cell = next(
        result
        for result in rebuilt["results"]
        if result["dataset_id"] == "szyn-opencode-swebench-verified-500"
        and result["metric_id"] == "agent_resolve_rate"
    )
    assert rebuilt_cell == cell
    assert rebuilt["generated_at"] == existing["generated_at"]
    assert rebuilt["status"] == existing["status"]
    assert rebuilt["baseline"] == existing["baseline"]


def test_szyn_publication_and_builder_do_not_revive_superseded_partial_progress() -> (
    None
):
    publication = load_json(PUBLICATION / "dataset_validation_qwen35_tp2_matrix.json")
    publication_text = json.dumps(publication, ensure_ascii=False)
    builder_text = (ROOT / "scripts" / "build_dataset_matrix_coverage.py").read_text(
        encoding="utf-8"
    )

    for stale_marker in ("1/500", "remaining 499", "其余 499"):
        assert stale_marker not in publication_text
        assert stale_marker not in builder_text

    szyn = next(
        result
        for result in publication["results"]
        if result["dataset_id"] == "szyn-opencode-swebench-verified-500"
        and result["metric_id"] == "agent_resolve_rate"
    )
    assert szyn["status"] == "baseline_only"
    assert szyn["baseline_value"] == 46.4
    assert szyn["provenance"]["attempted_tasks"] == 500
    assert szyn["provenance"]["resolved_tasks"] == 232
