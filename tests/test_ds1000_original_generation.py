"""CPU-only invented fixtures: never import/execute publisher or generated code."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "ds_runner", Path(__file__).parents[1] / "scripts/run_ds1000_original_generation.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

PROFILE = 'generation_config = {"do_sample": False, "max_new_tokens": 1024, "num_beams": 1}\nstop_words = ["</code>", "# SOLUTION END"]\n'


def test_publisher_config_is_literal_data_not_executed(tmp_path):
    sentinel = tmp_path / "must-not-exist"
    source = f'open({str(sentinel)!r}, "w").write("bad")\n' + PROFILE
    assert (
        runner.parse_publisher_profile(source)["generation_config"]["max_new_tokens"]
        == 1024
    )
    assert not sentinel.exists()


@pytest.mark.parametrize(
    "source",
    [
        PROFILE + 'stop_words = ["other"]\n',
        PROFILE.replace("1024", "2048"),
        PROFILE.replace("1024", "1024.0"),
        PROFILE.replace("False", "0"),
        "generation_config = dangerous_call()\n",
    ],
)
def test_changed_or_dynamic_profiles_rejected(source):
    with pytest.raises(ValueError):
        runner.parse_publisher_profile(source)


def test_original_raw_prompt_no_stop_or_reference_leak():
    task = {
        "prompt": "\n original fixture prompt\n",
        "code_context": "PRIVATE_REFERENCE_SENTINEL",
    }
    before = deepcopy(task)
    body = runner.request_body(task["prompt"], "fixture-model")
    assert body["prompt"] == task["prompt"]
    assert "messages" not in body and "stop" not in body
    assert "PRIVATE_REFERENCE_SENTINEL" not in str(body)
    assert body["max_tokens"] == 1024 and body["temperature"] == 0 and body["n"] == 1
    assert body["add_special_tokens"] is True and body["skip_special_tokens"] is True
    assert task == before


def response(finish="stop", text="fixture-code</code>keep raw trailing text"):
    return {
        "choices": [{"text": text, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def test_postprocessing_is_after_generation_and_not_functional_scoring():
    value = response("length")
    before = deepcopy(value)
    summary, candidate = runner.response_summary(
        value, 10, ["</code>", "# SOLUTION END"]
    )
    assert candidate == "fixture-code"
    assert summary["finish_reason"] == "length"
    assert summary["postprocess_only_after_full_generation"] is True
    assert summary["status"] == "GENERATED_NOT_FUNCTIONALLY_SCORED"
    assert value == before
    assert "pass_at_1" not in summary


def test_empty_postprocessed_candidate_is_not_relabelled_as_transport_failure():
    summary, candidate = runner.response_summary(
        response(text="</code>"), 10, ["</code>"]
    )
    assert candidate == "" and summary["postprocessed_solution_nonempty"] is False
    assert summary["status"] == "GENERATED_NOT_FUNCTIONALLY_SCORED"


@pytest.mark.parametrize("kind", ["prompt", "budget", "total", "multiple"])
def test_inconsistent_usage_or_choices_rejected(kind):
    value = response()
    if kind == "prompt":
        value["usage"]["prompt_tokens"] = 9
    elif kind == "budget":
        value["usage"]["completion_tokens"] = 1025
    elif kind == "total":
        value["usage"]["total_tokens"] = 14
    else:
        value["choices"].append(deepcopy(value["choices"][0]))
    with pytest.raises(ValueError):
        runner.response_summary(value, 10, ["</code>"])


def test_budget_exit_is_distinct_from_execution_failure():
    rows = [
        {"status": "GENERATED_NOT_FUNCTIONALLY_SCORED", "finish_reason": "stop"}
        for _ in range(3)
    ]
    assert runner.profile_exit_code(rows) == 0
    rows[1]["finish_reason"] = "length"
    assert runner.profile_exit_code(rows) == 2
    rows[2]["status"] = "REQUEST_OR_PROTOCOL_FAILED"
    assert runner.profile_exit_code(rows) == 1
    assert runner.profile_exit_code([]) == 1


@pytest.mark.parametrize(
    "value",
    ["https://example.com", "http://localhost/v1", "http://localhost/?key=fixture"],
)
def test_only_local_origins_without_credentials(value):
    with pytest.raises(ValueError):
        runner.origin(value)


def test_handoff_keeps_original_budget_limit_and_no_score():
    root = Path(__file__).parents[1]
    receipt = json.loads(
        (
            root
            / "docs/workload-data/validation/ds1000-original-generation-handoff-20261023.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["unique_tasks"] == 3
    assert [r["finish_reason"] for r in receipt["samples"]] == [
        "stop",
        "length",
        "stop",
    ]
    assert receipt["profile_exit_code"] == 2 and receipt["max_new_tokens"] == 1024
    # Frozen historical runner identity is not the hash of later range support.
    assert (
        receipt["public_runner_sha256"]
        == "3b4bcfd07d9c7fa288d3ccdc3d88287ce92d7bb7a9e573e6f84bf6a52d282675"  # pragma: allowlist secret - historical public runner checksum
    )
    assert receipt["postprocess_only_after_full_generation"] is True
    assert (
        receipt["generated_code_executed"] is False
        and receipt["code_context_executed"] is False
    )
    assert receipt["scorer_runs"] == 0 and receipt["source_checkbox_eligible"] is False


def test_original_selection_supports_sparse_ids_ranges_and_full():
    assert runner.select_indices(1000) == [0, 1, 2]
    assert runner.select_indices(1000, indices=[291, 511, 932]) == [291, 511, 932]
    assert runner.select_indices(1000, start=291, count=3) == [291, 292, 293]
    assert runner.select_indices(1000, full=True) == list(range(1000))


@pytest.mark.parametrize(
    "options",
    [
        {"indices": [2, 1, 0]},
        {"indices": [0, 1, 1]},
        {"indices": [0, 1, 1000]},
        {"indices": [0, 1]},
        {"indices": [0, 1, 2], "start": 0},
        {"start": 999, "count": 3},
    ],
)
def test_bad_or_conflicting_selection_rejected(options):
    with pytest.raises(ValueError):
        runner.select_indices(1000, **options)
