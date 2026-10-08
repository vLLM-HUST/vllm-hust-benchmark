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
    index = load_json(PUBLICATION / "dataset_validation_index_v1.json")
    jsonschema.Draft202012Validator(index_schema).validate(index)
    for scenario in index["scenarios"]:
        jsonschema.Draft202012Validator(artifact_schema).validate(
            load_json(PUBLICATION / scenario["data_file"])
        )


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
