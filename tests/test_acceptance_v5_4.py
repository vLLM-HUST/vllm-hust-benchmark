import json

from vllm_hust_benchmark.acceptance_v5_4 import (
    build_measurement_queue,
    load_and_validate,
)


def test_v54_contract_and_semantic_mirror_are_pinned() -> None:
    contract = load_and_validate()
    assert contract["test_plan_version"] == "V5.4"
    assert contract["source_fidelity"]["status"] == "semantic-mirror"
    assert contract["source_fidelity"]["byte_identical_to_canonical_final"] is False
    assert contract["mandatory_delivery"]["model"] == "Qwen/Qwen3.5-35B-A3B"
    assert contract["baseline_roles"]["allowed"] == ["B0", "B1"]
    assert contract["baseline_roles"]["B1"]["core_id"] is None
    assert contract["baseline_roles"]["B1"]["plugin_id"] is None
    assert contract["baseline_roles"]["B1"]["image_id"] is None


def test_v54_fixed_slo_cost_and_prompt_contract() -> None:
    contract = load_and_validate()
    assert contract["a3"]["slo"]["ttft_seconds"] == {
        "mean_max": 30,
        "p95_max": 45,
        "p99_max": 60,
    }
    assert contract["a3"]["slo"]["tpot_ms_per_token"] == {
        "mean_max": 40,
        "p95_max": 60,
        "p99_max": 80,
    }
    assert (
        contract["a3"]["prefill_throughput"]["may_generate_acceptance_threshold"]
        is False
    )
    assert contract["a4"]["cost_formula"].startswith("complete_lifecycle_cost /")
    assert contract["prompt_contract"]["natural_language_prompts_embedded"] is False


def test_v54_inactive_optional_deliveries_never_generate_queue_entries() -> None:
    contract = load_and_validate()
    assert [item["id"] for item in contract["optional_deliveries"]] == [
        "E1",
        "E2-G",
        "E2-D",
    ]
    assert all(item["active"] is False for item in contract["optional_deliveries"])
    assert all(
        item["verdict"] == "NOT_EXECUTED_OPTIONAL"
        for item in contract["optional_deliveries"]
    )
    queue = build_measurement_queue(contract)
    assert len(queue) == 10
    assert {item["baseline_role"] for item in queue} == {"B0", "B1"}
    serialized = json.dumps(queue)
    assert "E1" not in serialized
    assert "E2-G" not in serialized
    assert "E2-D" not in serialized
    assert "B2" not in serialized


def test_v54_aligns_the_contract_with_measured_bf16_evidence() -> None:
    contract = load_and_validate()
    alignment = contract["measured_precision_alignment"]
    assert contract["mandatory_delivery"]["precision"] == "BF16"
    assert contract["mandatory_delivery"]["dtype"] == "bfloat16"
    assert alignment["required_precision"] == "BF16"
    assert alignment["status"] == "aligned"
    assert len(alignment["evidence"]) == 2
    assert alignment["remaining_unconfirmed_b0_fields"]
