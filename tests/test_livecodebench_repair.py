"""Inert feedback fixtures; no original tasks, model calls or code execution."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
repair_spec = importlib.util.spec_from_file_location(
    "repair_contract", SCRIPTS / "livecodebench_repair.py"
)
repair = importlib.util.module_from_spec(repair_spec)
repair_spec.loader.exec_module(repair)
build_repair_requests, repair_identities = (
    repair.build_repair_requests,
    repair.repair_identities,
)

spec = importlib.util.spec_from_file_location(
    "repair_generation", SCRIPTS / "run_livecodebench_original_generation.py"
)
generation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generation)


def test_approved_runtime_compat_uses_only_existing_error_message_without_mutation():
    metadata = json.dumps(
        {"error_code": -4, "error_message": "Error during testing: inert exception"}
    )

    def original(*args):
        raise KeyError("inputs")

    result = repair.compatible_check_prompt(original, "inert question", False, metadata)
    assert "Error during testing: inert exception" in result
    assert "Expected:" not in result and "Input:" not in result
    assert json.loads(metadata) == {
        "error_code": -4,
        "error_message": "Error during testing: inert exception",
    }


@pytest.mark.parametrize(
    "metadata",
    [
        {"error_code": -2, "error_message": "inert wrong answer"},
        {"error_code": -4},
        {"error_code": -4, "error_message": "inert missing unrelated field"},
        {
            "error_code": -4,
            "error_message": "inert",
            "inputs": "actual",
            "expected": "actual",
            "error": "actual",
        },
    ],
)
def test_compatibility_does_not_mask_other_formatter_failures(metadata):
    def original(*args):
        raise KeyError("unrelated missing field")

    with pytest.raises(KeyError):
        repair.compatible_check_prompt(
            original, "inert question", False, json.dumps(metadata)
        )


def test_compatible_wrapper_keeps_normal_author_formatting_unchanged():
    assert (
        repair.compatible_check_prompt(
            lambda *args: "unchanged native feedback", "q", False, "{}"
        )
        == "unchanged native feedback"
    )


def test_accepted_original_output_is_carried_without_another_model_request():
    previous = evaluated()
    requests = [
        {
            "question_id": "inert-original-id",
            "candidate_index": i,
            "request_id": f"inert-original-id::candidate:{i}",
        }
        for i in range(1, 10)
    ]
    calls = []

    def read_output(position, request):
        calls.append((position, request["candidate_index"]))
        return [f"inert repair {request['candidate_index']}"]

    results = repair.repaired_outputs(previous, requests, read_output)
    assert results[0][0] == previous[0]["output_list"][0]
    assert results[0][1:] == [[f"inert repair {i}"] for i in range(1, 10)]
    assert calls == list(enumerate(range(1, 10)))


def test_all_accepted_candidates_require_zero_new_model_outputs():
    previous = evaluated()
    previous[0]["graded_list"] = [True] * 10
    assert repair_identities(previous) == []

    def forbidden(*args):
        raise AssertionError("accepted original candidates must not call model")

    outputs = repair.repaired_outputs(previous, [], forbidden)
    assert outputs == [previous[0]["output_list"]]


def evaluated():
    return [
        {
            "question_id": "inert-original-id",
            "question_content": "inert question",
            "output_list": [f"inert original {i}" for i in range(10)],
            "code_list": [f"inert code {i}" for i in range(10)],
            "graded_list": [True] + [False] * 9,
            "metadata": ["{}"]
            + [
                json.dumps(
                    {
                        "error_code": -2,
                        "inputs": f"fixture {i}",
                        "output": "wrong fixture",
                        "expected": "original fixture",
                    }
                )
                for i in range(1, 10)
            ],
        }
    ]


def kwargs():
    return {
        "model": "inert-local-model",
        "n": 1,
        "temperature": 0.2,
        "max_tokens": 2000,
        "top_p": 0.95,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "timeout": 90,
    }


def test_all_failed_candidates_receive_exact_original_feedback_correct_one_carried():
    previous = evaluated()
    unchanged = deepcopy(previous)
    calls = []

    def formatter(question, style, code, grade, metadata):
        calls.append((question, style, code, grade, metadata))
        return "" if grade else [{"role": "user", "content": metadata}]

    requests = build_repair_requests(
        previous, formatter, "inert-original-style", kwargs()
    )
    assert previous == unchanged
    assert len(calls) == 10 and len(requests) == 9
    assert [r["candidate_index"] for r in requests] == list(range(1, 10))
    assert all(r["question_id"] == "inert-original-id" for r in requests)
    assert all(
        r["request"]["n"] == 1 and "timeout" not in r["request"] for r in requests
    )
    assert [r["request"]["messages"][0]["content"] for r in requests] == previous[0][
        "metadata"
    ][1:]


@pytest.mark.parametrize(
    "mutation",
    ["missing_candidate", "non_boolean", "rewritten_metadata", "duplicate_question"],
)
def test_incomplete_or_ambiguous_prior_evaluation_rejected(mutation):
    previous = evaluated()
    if mutation == "missing_candidate":
        previous[0]["code_list"].pop()
    elif mutation == "non_boolean":
        previous[0]["graded_list"][0] = 1
    elif mutation == "rewritten_metadata":
        previous[0]["metadata"][1] = {"error_code": -2}
    else:
        previous += deepcopy(previous)
    with pytest.raises(ValueError):
        repair_identities(previous)


def test_native_n1_repair_requests_keep_original_question_and_candidate_relation(
    tmp_path,
):
    rows = [
        {
            "question_id": "inert-original-id",
            "candidate_index": i,
            "request_id": f"inert-original-id::candidate:{i}",
            "request": {
                **kwargs(),
                "messages": [{"role": "user", "content": "inert feedback"}],
            },
        }
        for i in (1, 2)
    ]
    for row in rows:
        row["request"].pop("timeout")
    path = tmp_path / "requests.json"
    path.write_text(json.dumps(rows))
    assert generation.requests_from(path, scenario="selfrepair") == rows
    with pytest.raises(ValueError, match="defaults"):
        generation.requests_from(path)
    calls = []

    def transport(*args):
        calls.append(args)
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {"content": "inert repaired output"},
                    "finish_reason": "stop",
                }
            ]
        }

    result = generation.run(
        path,
        tmp_path / "results",
        endpoint="http://localhost:19466/v1",
        scenario="selfrepair",
        transport=transport,
    )
    assert len(calls) == 2 and result["model_samples_per_request"] == 1
    assert result["completed_question_ids"] == ["inert-original-id"] * 2
    assert result["completed_request_ids"] == [row["request_id"] for row in rows]
    assert (
        result["status"]
        == "ORIGINAL_SELF_REPAIR_GENERATION_RETURNED_NATIVE_SCORING_PENDING"
    )
