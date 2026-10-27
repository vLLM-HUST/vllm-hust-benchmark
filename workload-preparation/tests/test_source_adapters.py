"""Tiny invented parser fixtures only; these are not workload evidence."""

from copy import deepcopy
import gzip
import hashlib
import json

import pytest

from vllm_hust_workload_data import iter_workload
from vllm_hust_workload_data.adapters import adapt_row
from vllm_hust_workload_data.assets import asset_path, iter_rows
from vllm_hust_workload_data.registry import adapter_spec, read_catalog


def source(sid):
    return {
        "source_id": sid,
        "revision": "unit-fixture",
        "file_sha256": "f" * 64,
        "file": "fixture.jsonl",
    }


def adapt(sid, row):
    return adapt_row(sid, row, source=source(sid), spec=adapter_spec(sid), index=0)


@pytest.mark.parametrize(
    "sid,input_field,reference_field",
    [
        ("DOLLY", "instruction", "response"),
        ("GSM8K", "question", "answer"),
        ("MATH500", "problem", "answer"),
        ("MMLU_PRO", "question", "cot_content"),
        ("AI2_ARC", "question", "answerKey"),
        ("BIGCODEBENCH", "complete_prompt", "canonical_solution"),
        ("DS1000", "prompt", "reference_code"),
        ("SCIENCEAGENTBENCH", "task_inst", "gold_program_name"),
        ("STRUCTEVAL", "query", "raw_output_metric"),
        ("LONGBENCH_V2", "question", "answer"),
        ("SWE_VERIFIED", "problem_statement", "patch"),
    ],
)
def test_explicit_projection_never_puts_reference_into_input(
    sid, input_field, reference_field
):
    spec = adapter_spec(sid)
    row = {key: "fixture" for key in spec["required_fields"]}
    row[input_field] = "test input"
    if sid == "MATH500":
        row["level"] = 3
        row["unique_id"] = "fixture/math"
    if sid == "MMLU_PRO":
        row["options"] = ["one", "two"]
    if sid == "AI2_ARC":
        row["choices"] = {"text": ["one", "two"], "label": ["A", "B"]}
    row[reference_field] = "PRIVATE_REFERENCE_SENTINEL"
    row["unknown_answer"] = "UNMAPPED_SENTINEL"
    before = deepcopy(row)
    record = adapt(sid, row)
    assert record.task["input"][input_field] == row[input_field]
    assert "PRIVATE_REFERENCE_SENTINEL" not in json.dumps(record.input_record())
    assert "UNMAPPED_SENTINEL" not in json.dumps(record.input_record())
    assert record.outcome_record()[reference_field] == row[reference_field]
    assert row == before
    with pytest.raises(ValueError, match="not a ready model request"):
        record.request_body()


@pytest.mark.parametrize(
    "sid", list(read_catalog("adapter_registry_v1.json")["adapters"])
)
def test_missing_required_fields_fail_instead_of_generating(sid):
    spec = adapter_spec(sid)
    if spec["kind"] in {"external_runtime_reference", "official_task_suite"}:
        with pytest.raises(NotImplementedError, match="official native runner"):
            adapt(sid, {})
        return
    with pytest.raises(ValueError, match="missing required fields"):
        adapt(sid, {})


def test_enterprise_wrapped_request_preserves_entire_api_body_and_no_private_repr():
    body = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call1",
                        "function": {"name": "lookup", "arguments": '{"x":1}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call1", "content": "PRIVATE_TOOL_RESULT"},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "local-image-ref"}}
                ],
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "test", "schema": {"type": "object"}},
        },
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "max_completion_tokens": 64,
        "stream": True,
    }
    record = adapt(
        "PAIO-JSON-500",
        {"request_body": json.dumps(body), "secret_wrapper": "PRIVATE_WRAPPER"},
    )
    assert record.request_body() == body
    assert record.outcome_record() is None
    assert "PRIVATE" not in repr(record)
    record.request_body()["messages"].clear()
    assert record.request_body() == body


def test_enterprise_estimates_and_creation_time_are_not_runtime_measurements():
    row = {
        "messages": [{"role": "user", "content": "fixture"}],
        "prompt": "fixture",
        "metadata": {"output_tokens_target": 88},
        "created_at_ms": 1,
        "group_id": "constructed",
    }
    record = adapt("PAIO-LONG-PREFILL-5000", row)
    assert "max_tokens" not in record.request_body()
    assert "timestamp" not in record.replay
    assert record.replay["annotations"]["created_at_ms"] == 1
    assert record.replay["annotations"]["metadata"]["output_tokens_target"] == 88


def test_oasst_preserves_edges_and_role_not_fake_sequence():
    row = {
        "message_id": "child",
        "parent_id": "parent",
        "text": "observed response",
        "role": "assistant",
        "message_tree_id": "root",
        "rank": 0,
    }
    record = adapt("OASST1", row)
    assert record.task["parent_task_id"] == "parent"
    assert record.task["session_id"] == "root"
    assert "not_conversation_order" in record.replay["order"]
    assert record.runtime["kind"] == "historical_message"


def test_longwriter_target_is_separate_not_entire_training_dialogue_as_input():
    record = adapt(
        "LONGWRITER",
        {
            "messages": [
                {"role": "user", "content": "write"},
                {"role": "assistant", "content": "TARGET_SENTINEL"},
            ]
        },
    )
    assert "TARGET_SENTINEL" not in json.dumps(record.input_record())
    assert record.outcome_record()["reference_message"]["content"] == "TARGET_SENTINEL"


def test_toolace_keeps_native_historical_dialogue_not_live_tool_execution():
    row = {
        "system": "tool specification",
        "conversations": [
            {"from": "human", "value": "fixture"},
            {"from": "function", "value": "observed"},
        ],
    }
    record = adapt("TOOLACE", row)
    assert record.task["input"] == row
    assert record.runtime["kind"] == "historical_dialogue"


def test_chartqa_keeps_real_media_bytes_and_separates_label():
    row = {
        "image": {"bytes": b"fixture-media-bytes", "path": None},
        "query": "fixture",
        "label": ["target"],
    }
    record = adapt("CHARTQA", row)
    assert record.task["input"]["image"]["bytes"] == row["image"]["bytes"]
    assert "label" not in record.task["input"]
    with pytest.raises(ValueError, match="real image"):
        adapt("CHARTQA", {**row, "image": {"bytes": None, "path": None}})


@pytest.mark.parametrize(
    "sid,row",
    [
        (
            "BURSTGPT",
            {
                "Timestamp": "1",
                "Model": "fixture",
                "Request tokens": "10",
                "Response tokens": "0",
                "Log Type": "failed",
            },
        ),
        (
            "TRACELAB",
            {
                "round_id": "r1",
                "round_index": 0,
                "session_id": "s1",
                "timing_events": [{"event": "start"}],
            },
        ),
    ],
)
def test_metadata_does_not_turn_into_fake_prompts_or_filter_failure(sid, row):
    record = adapt(sid, row)
    assert record.task["input"] == {}
    assert record.replay["annotations"] == row
    assert record.outcome_record() is None


def fixture_catalog(tmp_path, rows, *, expected_rows=None):
    data = tmp_path / "datafiles"
    data.mkdir()
    path = data / "fixture.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    catalog = tmp_path / "catalog"
    dest = catalog / "data/catalogs"
    dest.mkdir(parents=True)
    (dest / "adapter_registry_v1.json").write_text(
        json.dumps({"adapters": {"GSM8K": adapter_spec("GSM8K")}}), encoding="utf-8"
    )
    spec = {
        "revision": "unit-fixture",
        "unit": "tasks",
        "provenance_kind": "UNIT_TEST_NOT_EVIDENCE",
        "files": [
            {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
                "rows": expected_rows if expected_rows is not None else len(rows),
                "role": "data",
            }
        ],
    }
    (dest / "source_manifests_v1.json").write_text(
        json.dumps({"sources": {"GSM8K": spec}}), encoding="utf-8"
    )
    return data, catalog


def test_pinned_external_root_determinism_count_and_no_implicit_download(
    tmp_path, monkeypatch
):
    import socket

    monkeypatch.setattr(
        socket, "create_connection", lambda *a, **k: pytest.fail("network")
    )
    data, catalog = fixture_catalog(
        tmp_path,
        [{"question": "one", "answer": "a"}, {"question": "two", "answer": "b"}],
    )
    first = list(iter_workload("GSM8K", data_root=data, catalog_root=catalog))
    second = list(iter_workload("GSM8K", data_root=data, catalog_root=catalog))
    assert [r.input_record() for r in first] == [r.input_record() for r in second]
    assert len(first) == 2 and first[0].task["task_id"] != first[1].task["task_id"]
    assert (
        len(list(iter_workload("GSM8K", data_root=data, catalog_root=catalog, limit=1)))
        == 1
    )
    (data / "fixture.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        list(iter_workload("GSM8K", data_root=data, catalog_root=catalog))


def test_full_file_row_count_mismatch_is_not_accepted(tmp_path):
    data, catalog = fixture_catalog(
        tmp_path, [{"question": "one", "answer": "a"}], expected_rows=2
    )
    with pytest.raises(ValueError, match="row count mismatch"):
        list(iter_workload("GSM8K", data_root=data, catalog_root=catalog))


@pytest.mark.parametrize(
    "selection", [[], ["other.jsonl"], ["fixture.jsonl", "fixture.jsonl"]]
)
def test_only_explicit_registered_distinct_files(tmp_path, selection):
    data, catalog = fixture_catalog(tmp_path, [{"question": "one", "answer": "a"}])
    with pytest.raises(ValueError, match="registered data files"):
        list(
            iter_workload(
                "GSM8K", data_root=data, catalog_root=catalog, files=selection
            )
        )


@pytest.mark.parametrize(
    "relative", ["../escape", "/absolute", "C:/escape", "C:\\escape", "..\\escape"]
)
def test_asset_paths_cannot_escape(tmp_path, relative):
    with pytest.raises(ValueError):
        asset_path(tmp_path, relative)


def test_unknown_source_and_missing_data_do_not_generate(tmp_path):
    with pytest.raises(NotImplementedError):
        list(iter_workload("UNKNOWN", data_root=tmp_path))
    with pytest.raises(FileNotFoundError, match="no download or fallback"):
        list(iter_workload("GSM8K", data_root=tmp_path))
    with pytest.raises(NotImplementedError, match="no QBI fallback"):
        list(iter_workload("GAIA", data_root=tmp_path))


@pytest.mark.parametrize(
    "text", ["not json PRIVATE_SENTINEL\n", '{"x":NaN}\n', "[1]\n", "\n"]
)
def test_json_parser_errors_never_expose_payload(tmp_path, text):
    path = tmp_path / "fixture.jsonl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError) as error:
        list(iter_rows(path))
    assert "PRIVATE_SENTINEL" not in str(error.value)


def test_json_csv_gzip_readers(tmp_path):
    path = tmp_path / "fixture.json"
    path.write_text('[{"key":1}]', encoding="utf-8")
    assert list(iter_rows(path)) == [{"key": 1}]
    path = tmp_path / "fixture.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    assert list(iter_rows(path)) == [{"a": "1", "b": "2"}]
    path.write_text("a,b\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError):
        list(iter_rows(path))
    path = tmp_path / "fixture.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write('{"key":1}\n')
    assert list(iter_rows(path)) == [{"key": 1}]


def test_parquet_binary_input_is_retained(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "fixture.parquet"
    pq.write_table(pa.Table.from_pylist([{"blob": b"fixture"}]), path)
    assert list(iter_rows(path)) == [{"blob": b"fixture"}]


@pytest.mark.parametrize("value", [None, "", "   ", 12, []])
def test_malformed_required_text_is_not_an_adapted_task(value):
    with pytest.raises(ValueError):
        adapt("GSM8K", {"question": value, "answer": "fixture"})


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "1"])
def test_invalid_limits_rejected_before_asset_access(tmp_path, limit):
    with pytest.raises(ValueError, match="positive integer"):
        list(iter_workload("GSM8K", data_root=tmp_path, limit=limit))


def test_nested_nonfinite_json_request_is_rejected():
    with pytest.raises(ValueError, match="invalid request body"):
        adapt("PAIO-JSON-500", {"request_body": '{"messages": [], "temperature": NaN}'})


def test_tulu3_preserves_original_messages_and_component_provenance():
    row = {
        "id": "original-fixture-id",
        "messages": [
            {"role": "user", "content": "unchanged question"},
            {"role": "assistant", "content": "original reference"},
        ],
        "source": "original-component-fixture",
    }
    original = deepcopy(row)
    record = adapt("TULU3", row)
    assert row == original
    assert record.task["input"]["messages"] == original["messages"]
    assert record.task["task_id"] == row["id"]
    assert record.replay["annotations"]["source"] == row["source"]
    assert record.replay["annotations"]["id"] == row["id"]
    assert record.runtime["kind"] == "historical_dialogue"
    assert record.outcome is None
    for missing in ("id", "messages", "source"):
        with pytest.raises(ValueError):
            adapt("TULU3", {key: value for key, value in row.items() if key != missing})
