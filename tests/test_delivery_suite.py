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


def test_delivery_suite_covers_exactly_nine_distinct_workloads_and_benchmarks() -> None:
    entries = delivery_suite_entries()

    assert {entry.workload_id for entry in entries} == REQUIRED_WORKLOAD_IDS
    assert len(entries) == 9
    assert len({entry.benchmark_id for entry in entries}) == 9


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
        entry["model"]["deployment_tier"] != "customer_multinode_extension"
        for entry in payload["entries"]
    )
    assert {row["model_id"] for row in payload["customer_multinode_extensions"]} == {
        "zai-org/GLM-5",
        "moonshotai/Kimi-K2",
        "moonshotai/Kimi-K3",
    }


def test_workload_bindings_preserve_key_model_choices() -> None:
    assert resolve_delivery_workload("reasoning")["model"]["id"] == (
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    )
    assert resolve_delivery_workload("agent")["model"]["id"] == "zai-org/GLM-4.5-Air"
    assert resolve_delivery_workload("long_context")["model"]["id"] == "Qwen/Qwen3.5-27B"


def test_validator_rejects_customer_multinode_fixed_target() -> None:
    payload = json.loads(json.dumps(load_delivery_suite_registry()))
    payload["entries"][0]["model"]["deployment_tier"] = "customer_multinode_extension"

    with pytest.raises(ValueError, match="cannot be fixed acceptance targets"):
        validate_delivery_suite_registry(payload)
