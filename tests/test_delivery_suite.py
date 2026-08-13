from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from vllm_hust_benchmark.delivery_suite import REQUIRED_WORKLOAD_IDS
from vllm_hust_benchmark.delivery_suite import delivery_suite_entries
from vllm_hust_benchmark.delivery_suite import load_delivery_suite_registry
from vllm_hust_benchmark.delivery_suite import resolve_delivery_workload
from vllm_hust_benchmark.delivery_suite import validate_delivery_suite_registry
from vllm_hust_benchmark.enterprise_replay import enterprise_case_rows


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_delivery_suite_registry_matches_json_schema() -> None:
    schema = json.loads(
        (REPO_ROOT / "schemas" / "delivery_suite_registry_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(
        load_delivery_suite_registry()
    )


def test_delivery_suite_covers_exactly_nine_application_scenarios() -> None:
    entries = delivery_suite_entries()

    assert {entry.workload_id for entry in entries} == REQUIRED_WORKLOAD_IDS
    assert len(entries) == 9
    assert all(entry.public_asset_ids for entry in entries)


def test_delivery_suite_enterprise_cases_exist_in_enterprise_registry() -> None:
    known_cases = {row["case_id"] for row in enterprise_case_rows()}

    for entry in delivery_suite_entries():
        assert set(entry.enterprise_case_ids) <= known_cases


def test_delivery_suite_pins_academic_descriptor_contract() -> None:
    contract = load_delivery_suite_registry()["academic_workload_contract"]

    assert contract == {
        "descriptor_schema_version": "workload-descriptor/v1",
        "family_generator_version": "family-generator/v1",
    }
    assert all(entry.workload_family_ids for entry in delivery_suite_entries())


def test_main_suite_excludes_customer_multinode_models() -> None:
    payload = load_delivery_suite_registry()

    assert all(
        entry["model_artifact"]["deployment_tier"] != "customer_multinode_extension"
        for entry in payload["entries"]
    )
    assert {row["model_class"] for row in payload["customer_multinode_extensions"]} == {
        "GLM-5-class",
        "Kimi-K2-or-K3-class",
    }


def test_workload_bindings_preserve_key_model_choices() -> None:
    assert resolve_delivery_workload("reasoning")["model_artifact"]["repo_id"] == (
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    )
    assert resolve_delivery_workload("agent")["model_artifact"]["repo_id"] == "zai-org/GLM-4.5-Air"
    assert resolve_delivery_workload("long_context")["model_artifact"]["repo_id"] == "Qwen/Qwen3.6-27B"
    assert all(len(entry.model_revision) == 40 for entry in delivery_suite_entries())


def test_validator_rejects_customer_multinode_fixed_target() -> None:
    payload = json.loads(json.dumps(load_delivery_suite_registry()))
    payload["entries"][0]["model_artifact"]["deployment_tier"] = "customer_multinode_extension"

    with pytest.raises(ValueError, match="cannot be main acceptance targets"):
        validate_delivery_suite_registry(payload)


def test_acceptance_policy_has_numeric_statistics_and_causal_baselines() -> None:
    policy = load_delivery_suite_registry()["acceptance_policy"]

    assert policy["experimental_design"]["paired_blocks"] >= 5
    assert policy["experimental_design"]["bootstrap_resamples"] >= 10000
    assert set(policy["causal_baselines"]) == {
        "upstream_stock",
        "hust_feature_off",
        "hust_feature_on",
        "minus_one_ablation",
    }
    assert policy["portfolio_gate"]["minimum_primary_pass_count"] == 7


def test_evaluation_asset_types_are_not_conflated() -> None:
    assets = {row["asset_id"]: row["asset_type"] for row in load_delivery_suite_registry()["evaluation_assets"]}

    assert assets["livecodebench"] == "quality_benchmark"
    assert assets["burstgpt-arrival-trace"] == "serving_trace"
    assert assets["vllm-prefix-repetition"] == "microbenchmark"
    assert assets["enterprise-request-replay"] == "enterprise_replay"
    assert assets["naturebench-design-reference"] == "design_reference"
