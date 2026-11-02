"""Synthetic arithmetic and rejection tests; fixtures are not measurements."""

import copy
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def calculator(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("recompute_full_answer_compute")


def inputs():
    config = {
        "text_config": dict(
            hidden_size=4,
            num_hidden_layers=2,
            num_attention_heads=1,
            head_dim=2,
            num_key_value_heads=1,
            linear_num_key_heads=1,
            linear_key_head_dim=2,
            linear_num_value_heads=1,
            linear_value_head_dim=2,
            num_experts_per_tok=2,
            moe_intermediate_size=3,
            shared_expert_intermediate_size=2,
            num_experts=4,
            vocab_size=10,
            layer_types=["full_attention", "linear_attention"],
            attn_output_gate=True,
            attention_bias=False,
        )
    }
    arm = dict(
        role="B0",
        run="synthetic-0",
        completed=1,
        failed=0,
        phase_seconds=2.0,
        successful_output_tokens=3,
        rows=[dict(case_id="synthetic", prompt_tokens=2, output_tokens=3)],
    )
    other = copy.deepcopy(arm)
    other.update(role="B1", run="synthetic-1", phase_seconds=1.0)
    return dict(
        protocol=dict(reserved_cards=2, tensor_parallel=2), arms=[arm, other]
    ), config


def test_hand_enumerated_nonpadding_work(calculator):
    pair, config = inputs()
    result = calculator.calculate(pair, config)
    # Four necessary positions, ten causal pairs; 80+96+464 FLOPs/position.
    arm = result["arms"][0]
    assert arm["necessary_target_positions"] == 4
    assert arm["causal_pairs"] == 10
    assert arm["matrix_flops_parts"] == dict(
        target_projections=2560, target_causal_qk_av=80, target_vocabulary=240
    )
    assert arm["matrix_subtotal_tflops_per_chip"] == 2880 / (2 * 2 * 1e12)
    assert result["matrix_subtotal_rate_gain_percent"] == 100
    for item in result["arms"]:
        for key in (
            "effective_flops_total",
            "effective_tflops_per_chip",
            "reference_seconds",
            "efficiency_percent",
            "outline_pass",
        ):
            assert item[key] is None


def test_natural_output_lengths_count_separately(calculator):
    pair, config = inputs()
    pair["arms"][1]["rows"][0]["output_tokens"] = 1
    pair["arms"][1]["successful_output_tokens"] = 1
    result = calculator.calculate(pair, config)
    assert result["arms"][1]["matrix_flops_subtotal"] == 2 * 640 + 3 * 8 + 80


@pytest.mark.parametrize(
    "failure", ["chips", "input", "output", "duplicate", "time", "failed"]
)
def test_reject_mismatched_or_invalid_measurements(calculator, failure):
    pair, config = inputs()
    arm = pair["arms"][1]
    if failure == "chips":
        pair["protocol"]["reserved_cards"] = 4
    elif failure == "input":
        arm["rows"][0]["prompt_tokens"] = 3
    elif failure == "output":
        arm["rows"][0]["output_tokens"] = 0
    elif failure == "duplicate":
        arm["rows"] *= 2
        arm["completed"] = 2
    elif failure == "time":
        arm["phase_seconds"] = float("nan")
    else:
        arm["failed"] = 1
    with pytest.raises(ValueError):
        calculator.calculate(pair, config)


def test_terminal_clipping_keeps_accepted_roots_before_discarded_bonus(calculator):
    def row(output):
        return dict(
            prompt_tokens=4,
            output_tokens=output,
            per_step_accepted=[2],
            per_step_drafted=[2],
        )

    both = calculator.draft_dependencies([row(3)])
    assert both["retained_vocab_rows"] == 2
    assert both["kv_rows"] == 5
    assert both["attention_pairs"] == 9
    one = calculator.draft_dependencies([row(2)])
    assert one["retained_vocab_rows"] == 1
    assert one["kv_rows"] == 4
    assert one["attention_pairs"] == 4


def test_dependency_context_includes_prior_unaccepted_step(calculator):
    row = dict(
        prompt_tokens=4,
        output_tokens=4,
        per_step_accepted=[0, 1],
        per_step_drafted=[2, 2],
    )
    result = calculator.draft_dependencies([row])
    assert result["kv_rows"] == 5
    assert result["full_query_rows"] == 1
    assert result["attention_pairs"] == 5


def test_partition_extrema_are_not_confidence_intervals(calculator):
    initial = {n: n * n for n in range(1, 65)}
    continuation = {n: 3 * n for n in range(1, 65)}
    assert calculator.partition_bounds([3, 4], initial, continuation) == [17, 25]
