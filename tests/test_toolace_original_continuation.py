"""Inert fixtures and frozen receipts; no live model or tool calls in tests."""

import importlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
runner = importlib.import_module("run_toolace_original_continuation")


def original():
    return {
        "system": "Original tool descriptions.",
        "conversations": [
            {"from": "user", "value": "Original question"},
            {"from": "assistant", "value": "Original textual function call"},
            {"from": "tool", "value": "Original historical tool response"},
            {"from": "assistant", "value": "Original second reply"},
            {"from": "user", "value": "Original followup"},
            {"from": "assistant", "value": "Original third reply"},
        ],
    }


def test_every_original_assistant_position_keeps_tool_history_without_ids():
    row = original()
    views = runner.original_prefixes(row)
    assert [view["original_assistant_turn_index"] for view in views] == [1, 3, 5]
    assert views[0]["messages"] == [
        {"role": "system", "content": row["system"]},
        {"role": "user", "content": "Original question"},
    ]
    assert views[1]["messages"][-1] == {
        "role": "tool",
        "content": "Original historical tool response",
    }
    assert views[1]["original_tool_turn_indices"] == [2]
    assert views[2]["historical_reference"] == "Original third reply"
    assert "Original third reply" not in json.dumps(views[2]["messages"])
    assert all(
        set(turn) == {"role", "content"} for view in views for turn in view["messages"]
    )
    views[0]["messages"][0]["content"] = "changed view"
    assert row["system"] == "Original tool descriptions."


@pytest.mark.parametrize(
    "failure",
    [
        "role",
        "extra-field",
        "content-type",
        "no-system",
        "first-role",
        "consecutive-assistant",
    ],
)
def test_unsupported_original_shape_fails_before_model_call(failure):
    row = original()
    if failure == "role":
        row["conversations"][-1]["from"] = "invented"
    elif failure == "extra-field":
        row["conversations"][-1]["tool_call_id"] = "invented"
    elif failure == "content-type":
        row["conversations"][-1]["value"] = ["parts"]
    elif failure == "no-system":
        row.pop("system")
    elif failure == "first-role":
        row["conversations"][0]["from"] = "tool"
    else:
        row["conversations"][2]["from"] = "assistant"
    with pytest.raises(ValueError):
        runner.original_prefixes(row)


def test_fixed_range_full_and_no_small_or_invalid_corpus():
    rows = [original()] * 11300
    assert [i for i, _ in runner.select(rows)] == [0, 1, 2]
    assert [i for i, _ in runner.select(rows, start=3)] == [3, 4, 5]
    assert len(runner.select(rows, full=True)) == 11300
    for count in (0, 1, 2):
        with pytest.raises(ValueError):
            runner.select(rows, count=count)
    with pytest.raises(ValueError):
        runner.select(rows[:3])
    with pytest.raises(ValueError):
        runner.select(rows, full=True, start=1)


def result(text="new text", finish="stop", tokens=3):
    return {
        "choices": [{"text": text, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": tokens},
    }


def invoke(tmp_path, transport, **kwargs):
    return runner.run(
        [(0, original())],
        render=lambda messages: (json.dumps(messages), 10),
        output=tmp_path / "private",
        endpoint="http://127.0.0.1:19466/v1",
        model="fixture",
        max_tokens=32,
        context=128,
        transport=transport,
        **kwargs,
    )


def test_original_raw_rendered_requests_complete_without_tool_execution(tmp_path):
    calls = []

    def transport(url, body):
        calls.append((url, body))
        return {"count": 10} if url.endswith("/tokenize") else result()

    receipt = invoke(tmp_path, transport)
    assert receipt["conversations"] == 1 and len(receipt["requests"]) == 3
    assert (
        receipt["tools_executed"]
        == receipt["dynamic_observations"]
        == receipt["invented_tool_call_ids"]
        == 0
    )
    assert not receipt["quality_scored"] and not receipt["source_checkbox_eligible"]
    requests = [body for url, body in calls if url.endswith("/completions")]
    assert len(requests) == 3 and "tools" not in requests[0]
    assert json.loads(requests[1]["prompt"])[-1]["role"] == "tool"


@pytest.mark.parametrize(
    "text,finish,tokens", [("", "stop", 0), ("unfinished", "length", 32)]
)
def test_empty_or_budget_outputs_retained_without_quality_claim(
    tmp_path, text, finish, tokens
):
    def transport(url, body):
        return (
            {"count": 10} if url.endswith("/tokenize") else result(text, finish, tokens)
        )

    receipt = invoke(tmp_path, transport)
    assert [r["finish_reason"] for r in receipt["requests"]] == [finish] * 3
    saved = json.loads(
        (tmp_path / "private/request-000000.private.json").read_text("utf-8")
    )
    assert saved["response"]["choices"][0]["text"] == text
    assert not receipt["quality_scored"]


@pytest.mark.parametrize(
    "failure", ["tokenizer", "capacity", "response", "transport", "final-identity"]
)
def test_failure_preserves_original_attempt_and_never_retries_or_replaces(
    tmp_path, failure
):
    calls = []

    def transport(url, body):
        calls.append(url)
        if failure == "transport":
            raise RuntimeError("inert failure")
        if url.endswith("/tokenize"):
            return {
                "count": 11
                if failure == "tokenizer"
                else 120
                if failure == "capacity"
                else 10
            }
        response = result()
        if failure == "response":
            response["choices"] = []
        return response

    def verify():
        if failure == "final-identity":
            raise ValueError("source changed")

    with pytest.raises((RuntimeError, ValueError)):
        invoke(tmp_path, transport, final_verify=verify)
    receipt = json.loads((tmp_path / "private/receipt-private.json").read_text("utf-8"))
    assert receipt["status"] == "FAILED_NOT_COMPLETE"
    assert (tmp_path / "private/original-row-0.private.json").exists()
    assert not receipt["source_checkbox_eligible"]
    assert len(calls) <= 6


def test_remote_endpoint_refused_without_creating_results(tmp_path):
    with pytest.raises(ValueError):
        runner.run(
            [(0, original())],
            render=lambda _: ("fixture", 10),
            output=tmp_path / "never",
            endpoint="https://external.example/v1",
            model="fixture",
            max_tokens=32,
            context=128,
        )
    assert not (tmp_path / "never").exists()


def test_external_jinja_and_vocabulary_are_part_of_tokenizer_binding(tmp_path):
    for name in (
        "tokenizer_config.json",
        "tokenizer.json",
        "config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
    ):
        (tmp_path / name).write_text("inert fixture " + name, encoding="utf-8")
    expected = runner.local_tokenizer_files(tmp_path)
    assert set(expected) == {
        "tokenizer_config.json",
        "tokenizer.json",
        "config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
    }
    runner.verify_tokenizer_files(tmp_path, expected)
    (tmp_path / "chat_template.jinja").write_text(
        "changed external template", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="template changed"):
        runner.verify_tokenizer_files(tmp_path, expected)


def test_adding_a_new_external_template_does_not_evade_identity_check(tmp_path):
    for name in ("tokenizer_config.json", "tokenizer.json", "config.json"):
        (tmp_path / name).write_text("inert fixture", encoding="utf-8")
    expected = runner.local_tokenizer_files(tmp_path)
    (tmp_path / "chat_templates").mkdir()
    (tmp_path / "chat_templates/tool.jinja").write_text(
        "new template", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="template changed"):
        runner.verify_tokenizer_files(tmp_path, expected)


def test_frozen_native_receipt_retains_original_tool_history_not_online_claim():
    path = "docs/workload-data/validation/toolace-original-continuation-20261026.json"
    receipt = json.loads((ROOT / path).read_text("utf-8"))
    native, consumer = receipt["native_profile"], receipt["fixed_consumer"]
    assert receipt["source_checkbox_eligible"] is True
    assert native["selected_original_rows"] == [0, 1, 2]
    assert native["every_original_assistant_position_requested"] == 15
    assert native["original_assistant_turn_indices"] == [1, 3, 5, 7, 9] * 3
    assert native["prefixes_with_original_tool_return_context"] == 12
    assert native["finish_reasons"] == {"stop": 15}
    assert (
        native["local_prompts_independently_reproduced_and_api_token_counts_equal"]
        == 15
    )
    assert (
        native["tools_executed"]
        == native["invented_tool_call_ids"]
        == native["dynamic_observations"]
        == 0
    )
    assert not native["quality_scored"] and not native["online_agent_episode"]
    assert consumer["original_rows_and_index"] == 11300
    assert consumer["exact_installed_payload_files"] == 41
    assert consumer["public_research_roundtrip_indices"] == [
        0,
        1,
        2,
        11297,
        11298,
        11299,
    ]
    assert not consumer["research_default_lock_changed"]
    assert not consumer["research_observations_or_features_executed"]
    binding = receipt["latest_tokenizer_binding_recheck"]
    assert binding["runtime_files_match_prior_deployable_artifact"] == 6
    assert "chat_template.jinja" in binding["files"]
    assert binding["saved_prompts_reproduced"] == 15 and binding["new_model_calls"] == 0
    additional = receipt["additional_latest_entry_native_validation"]
    assert additional["selected_original_rows"] == [3, 4, 5]
    assert additional["native_requests"] == 15
    assert additional["finish_reasons"] == {"stop": 14, "length": 1}
    assert additional["actual_tokenizer_files_guarded_before_and_after"] == 6
    assert additional["original_rows0to2_not_rerun"] is True
    assert additional["budget_terminated_output_retained_without_retry"] is True
    assert additional["source_count_increment"] == 0
    for key in (
        "quality_or_function_correctness_claim",
        "publisher_model_or_bfcl_replication_claim",
        "full11300_model_execution_claim",
        "original_api_request_capture_claim",
        "raw_dataset_or_prediction_published",
    ):
        assert receipt[key] is False
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )
    source = ledger["delivery_inventory"]["TOOLACE"]
    assert source["status"] == "AVAILABLE" and source["checked"] is True
    assert source["delivery_receipt"] == path
