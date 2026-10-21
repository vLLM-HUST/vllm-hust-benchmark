"""Invented response fixtures only, never real model or task evidence."""

from copy import deepcopy
import json

import pytest

from vllm_hust_benchmark.workload_response_validation import check_response


def response(text="fixture", finish="stop", **message):
    return {
        "choices": [
            {
                "index": 0,
                "message": {"content": text, **message},
                "finish_reason": finish,
            }
        ]
    }


@pytest.mark.parametrize("body", [{}, {"response_format": {"type": "text"}}])
def test_plain_response_is_not_semantic_quality(body):
    before = deepcopy(body)
    result = check_response(response(), body)
    assert result["status"] == "RESPONSE_INTEGRITY_VERIFIED"
    assert result["semantic_quality"] == "NOT_EVALUATED_NO_INDEPENDENT_ORACLE"
    assert "fixture" not in json.dumps(result)
    assert body == before


@pytest.mark.parametrize(
    "text,status",
    [
        ("{}", "RESPONSE_INTEGRITY_VERIFIED"),
        ("[]", "OUTPUT_NOT_JSON_OBJECT"),
        ("```json\n{}\n```", "INVALID_JSON_OUTPUT"),
        ('{"x":NaN}', "INVALID_JSON_OUTPUT"),
    ],
)
def test_original_json_object_contract(text, status):
    assert (
        check_response(response(text), {"response_format": {"type": "json_object"}})[
            "status"
        ]
        == status
    )


def test_original_json_schema_not_an_invented_gold():
    body = {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {"n": {"type": "integer"}},
                    "required": ["n"],
                    "additionalProperties": False,
                }
            },
        }
    }
    assert (
        check_response(response('{"n":1}'), body)["status"]
        == "RESPONSE_INTEGRITY_VERIFIED"
    )
    assert (
        check_response(response('{"n":"one"}'), body)["status"]
        == "OUTPUT_SCHEMA_VIOLATION"
    )
    assert (
        check_response(response('{"n":1,"extra":2}'), body)["status"]
        == "OUTPUT_SCHEMA_VIOLATION"
    )


def test_stream_requires_done_and_finish_and_content():
    body = {"stream": True}
    stream = {
        "stream": True,
        "terminated": True,
        "termination": "[DONE]",
        "events": [
            {
                "choices": [
                    {"index": 0, "delta": {"content": "fixture"}, "finish_reason": None}
                ]
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"completion_tokens": 1}},
        ],
    }
    assert check_response(stream, body)["status"] == "RESPONSE_INTEGRITY_VERIFIED"
    stream["terminated"] = False
    assert check_response(stream, body)["status"] == "STREAM_NOT_TERMINATED"


@pytest.mark.parametrize(
    "value,status",
    [
        (response(finish="length"), "BUDGET_EXHAUSTED_QUALITY_UNVERIFIED"),
        (response(finish=None), "MISSING_OR_UNSUPPORTED_TERMINATION"),
        (response(""), "EMPTY_OUTPUT"),
        (response(tool_calls=[{}]), "NATIVE_TOOL_LOOP_REQUIRED"),
        ({"choices": []}, "CHOICE_COUNT_MISMATCH"),
    ],
)
def test_fail_closed(value, status):
    assert check_response(value, {})["status"] == status


def test_all_requested_choices_need_validation():
    value = response()
    assert check_response(value, {"n": 2})["status"] == "CHOICE_COUNT_MISMATCH"
    value["choices"].append(
        {"index": 1, "message": {"content": "fixture"}, "finish_reason": "length"}
    )
    assert (
        check_response(value, {"n": 2})["status"]
        == "BUDGET_EXHAUSTED_QUALITY_UNVERIFIED"
    )


def test_invalid_schema_and_unavailable_validator_fail_closed(monkeypatch):
    import sys

    body = {
        "response_format": {
            "type": "json_schema",
            "json_schema": {"schema": {"type": "not-a-json-schema-type"}},
        }
    }
    assert check_response(response("{}"), body)["status"] == "INVALID_SOURCE_SCHEMA"
    monkeypatch.setitem(sys.modules, "jsonschema", None)
    body["response_format"]["json_schema"]["schema"] = {}
    assert (
        check_response(response("{}"), body)["status"] == "SCHEMA_VALIDATOR_UNAVAILABLE"
    )


@pytest.mark.parametrize("format_value", [False, {}, {"type": []}])
def test_malformed_source_format_is_not_repaired(format_value):
    assert (
        check_response(response(), {"response_format": format_value})["status"]
        == "INVALID_SOURCE_FORMAT"
    )


def test_remote_schema_references_are_not_fetched_and_internal_refs_work():
    body = {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "schema": {"$ref": "https://example.invalid/private-schema"}
            },
        }
    }
    assert (
        check_response(response("{}"), body)["status"]
        == "SCHEMA_VALIDATION_UNAVAILABLE"
    )
    body["response_format"]["json_schema"]["schema"] = {
        "$defs": {"value": {"type": "integer"}},
        "$ref": "#/$defs/value",
    }
    assert (
        check_response(response("1"), body)["status"] == "RESPONSE_INTEGRITY_VERIFIED"
    )


def test_unknown_dialect_is_not_silently_reinterpreted():
    body = {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "schema": {"$schema": "https://example.invalid/custom-dialect"}
            },
        }
    }
    assert (
        check_response(response("{}"), body)["status"] == "UNSUPPORTED_SCHEMA_DIALECT"
    )


def test_huge_requested_choice_count_does_not_allocate_an_index_set():
    assert (
        check_response(response(), {"n": 10**12})["status"] == "CHOICE_COUNT_MISMATCH"
    )
