"""Invented field fixtures; original video/NPU execution is not implied."""

from copy import deepcopy
import json

import pytest

from vllm_hust_workload_data.adapters import adapt_row, iter_source_rows
from vllm_hust_workload_data.ovobench import restore_original_case
from vllm_hust_workload_data.registry import adapter_spec, source_spec


@pytest.mark.parametrize(
    "task",
    [
        "EPM",
        "HLD",
        "ASI",
        "STU",
        "OJR",
        "ATR",
        "FPD",
        "ACR",
        "OCR",
        "CRR",
        "SSR",
        "REC",
    ],
)
def test_complete_original_case_roundtrip_without_reference_leak(task, tmp_path):
    row = {"id": 0, "task": task, "video": "original-video.mp4"}
    if task in {"CRR", "SSR", "REC"}:
        row.update(
            test_info=[
                {"realtime": 1.5, "type": "GOLD_TYPE"},
                {"realtime": 4, "count": "GOLD_COUNT"},
            ],
            answer="GOLD_ANSWER",
        )
        if task == "SSR":
            row["test_info"][0]["step"] = "original instruction"
    else:
        row.update(
            question="original question",
            options=["first", "second"],
            realtime=1.5,
            gt=0,
            answer="GOLD_ANSWER",
        )
    original = deepcopy(row)
    path = tmp_path / "original.json"
    path.write_text(json.dumps([row]), encoding="utf-8")
    spec = adapter_spec("OVOBENCH")
    loaded = list(iter_source_rows(path, spec))
    assert loaded == [original]
    record = adapt_row(
        "OVOBENCH", loaded[0], source={"source_id": "OVOBENCH"}, spec=spec, index=0
    )
    assert "GOLD" not in json.dumps(record.input_record())
    assert restore_original_case(record) == original
    assert row == original
    if "test_info" in row:
        assert len(record.task["input"]["test_info"]) == 2
    with pytest.raises(ValueError, match="not a ready model request"):
        record.request_body()


def test_unknown_forward_stage_field_fails_instead_of_becoming_input():
    row = {
        "id": 1,
        "task": "REC",
        "video": "original.mp4",
        "test_info": [{"realtime": 1, "new_reference": "do not leak"}],
    }
    with pytest.raises(ValueError, match="unrecognized original forward stage"):
        adapt_row(
            "OVOBENCH",
            row,
            source={"source_id": "OVOBENCH"},
            spec=adapter_spec("OVOBENCH"),
            index=0,
        )


def test_publisher_annotation_pin_is_not_published_model_results():
    source = source_spec("OVOBENCH")
    assert source["files"][0]["path"] == "data/ovo_bench_new.json"
    assert source["files"][0]["rows"] == 1640
