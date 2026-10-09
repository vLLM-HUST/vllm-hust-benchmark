"""CPU checks for the named MMLU-Pro subset contract; no model scores simulated."""

import json
from types import SimpleNamespace

import pytest

from vllm_hust_benchmark import mmlu_pro_pyramidkv as m


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The answer is (C).", "C"),
        ("answer is J", "J"),
        ("Answer: B", "B"),
        ("First A, finally D", "D"),
        ("No choice given", None),
        ("answer is (K)", None),
    ],
)
def test_reference_extraction(text, expected):
    assert m.extract_answer(text) == expected


def test_target_prompt_does_not_leak_answer():
    row = {
        "question": "Which?",
        "options": ["one", "two"],
        "answer": "B",
        "answer_index": 1,
        "cot_content": "PRIVATE GOLD RATIONALE",
    }
    assert "PRIVATE" not in m.format_example(row, answered=False)
    assert "PRIVATE" in m.format_example(row, answered=True)
    row["answer"] = "A"
    with pytest.raises(ValueError, match="letter/index"):
        m.format_example(row, answered=False)


def test_selection_adds_only_real_eligible_task():
    inventory = [
        {"category": "subject", "question_id": i, "prompt_tokens": t}
        for i, t in [(3, 4097), (1, 4096), (2, 20), (4, 5000)]
    ]
    assert m.choose_tasks(inventory, 2) == {1, 2, 3}
    assert m.choose_tasks(inventory, 3) == {1, 2, 3}
    with pytest.raises(ValueError):
        m.choose_tasks(inventory, 0)


def test_tokenization_counts_ids_not_mapping_keys():
    def render(messages, **kwargs):
        assert kwargs["tokenize"] is False
        assert kwargs["enable_thinking"] is False
        return "rendered"

    tokenizer = SimpleNamespace(
        apply_chat_template=render, encode=lambda *a, **kw: list(range(40))
    )
    assert len(m.encode_prompt(tokenizer, "question")) == 40
    tokenizer.encode = lambda *a, **kw: {
        "input_ids": list(range(40)),
        "attention_mask": [1] * 40,
    }
    with pytest.raises(ValueError, match="actual prompt"):
        m.encode_prompt(tokenizer, "question")


def pair(case_id, output, success=True):
    case = {
        "case_id": case_id,
        "prompt_sha256": "hash",
        "prompt_tokens": 4096,
        "max_tokens": 2048,
        "ignore_eos": False,
        "category": "math",
        "answer": "B",
        "option_count": 2,
    }
    result = {**case, "output": output, "success": success}
    return case, result


def test_failures_remain_in_accuracy_denominator():
    pairs = [
        pair("correct", "answer is B"),
        pair("outside", "answer is J"),
        pair("invalid", "no answer"),
        pair("failed", "answer is B", False),
    ]
    cases, results = map(list, zip(*pairs))
    score = m.score_results(results, cases)
    assert score["count"] == 4
    assert score["correct"] == 1
    assert score["accuracy"] == 0.25
    assert score["invalid_answers"] == 3
    assert score["transport_failures"] == 1
    assert score["eligible"]["count"] == 0
    results[0]["prompt_tokens"] += 1
    with pytest.raises(ValueError, match="Unmatched"):
        m.score_results(results, cases)


def test_incomplete_or_duplicate_outcomes_rejected():
    a, ar = pair("a", "A")
    b, br = pair("b", "B")
    for results in ([ar], [ar, ar], [ar, br, br]):
        with pytest.raises(ValueError):
            m.score_results(results, [a, b])
    with pytest.raises(ValueError):
        m.score_results([], [])


def test_frozen_plan_detects_tampering(tmp_path):
    case, _ = pair("a", "B")
    case["prompt"] = list(range(40))
    case["prompt_tokens"] = 40
    case["prompt_sha256"] = m.sha(
        json.dumps(case["prompt"], separators=(",", ":")).encode()
    )
    m.save(tmp_path / "cases.json", [case])
    contract = {
        "tasks": 1,
        "scorer_source_sha256": m.sha(m.Path(m.__file__).read_bytes()),
        "files_sha256": {"cases.json": m.sha((tmp_path / "cases.json").read_bytes())},
    }
    m.save(tmp_path / "contract.json", contract)
    assert m.load_plan(tmp_path)[1] == [case]
    (tmp_path / "cases.json").write_text("[]\n")
    with pytest.raises(ValueError, match="SHA-256"):
        m.load_plan(tmp_path)


def test_arm_and_non_treatment_environment_are_bound():
    identity = {
        "environment": {
            "VLLM_ASCEND_KVCOMPRESS_ENABLED": "0",
            "VLLMHUST_EXT_ENABLED_BUNDLES": "",
            "VLLM_VERSION": "pinned",
        },
        "command": ["python", "serve"],
        "method_config_sha256": "hash",
    }
    contract = {"runtime": {**identity, "environment": dict(identity["environment"])}}
    m.validate_activation(identity, contract, "B0")
    with pytest.raises(ValueError, match="activation"):
        m.validate_activation(identity, contract, "B1")
    identity["environment"]["VLLM_VERSION"] = "different"
    with pytest.raises(ValueError, match="environment"):
        m.validate_activation(identity, contract, "B0")
