"""Invented tiny field fixtures, not evidence of original model execution."""

from copy import deepcopy
import json

import pytest

from vllm_hust_workload_data.adapters import adapt_row, iter_source_rows
from vllm_hust_workload_data.registry import adapter_spec
from vllm_hust_workload_data.streamingbench import restore_original_case


@pytest.mark.parametrize("sequential", [False, True])
@pytest.mark.parametrize("proactive", [False, True])
def test_group_roundtrip_preserves_order_and_references(
    tmp_path, sequential, proactive
):
    question = {"question": "original question", "time_stamp": "00:01:00"}
    if proactive:
        question.update(
            ground_truth_output="GOLD_OUTPUT", ground_truth_time_stamp="00:01:04"
        )
    else:
        question.update(
            answer="GOLD_ANSWER",
            options=["A. one", "B. two"],
            cause="GOLD_CAUSE",
            effect="GOLD_EFFECT",
        )
    group = {
        "video_path": "original.mp4",
        "questions": [question, {**question, "time_stamp": "00:02:00"}],
        "captions": "GOLD_CAPTION",
    }
    original = [group, deepcopy(group)] if sequential else group
    snapshot = deepcopy(original)
    path = tmp_path / "questions.json"
    path.write_text(json.dumps([original]), encoding="utf-8")
    spec = adapter_spec("STREAMINGBENCH")
    rows = list(iter_source_rows(path, spec))
    assert rows == [{"original_group": snapshot}]
    record = adapt_row(
        "STREAMINGBENCH",
        rows[0],
        source={"source_id": "STREAMINGBENCH"},
        spec=spec,
        index=0,
    )
    assert "GOLD" not in json.dumps(record.input_record())
    assert restore_original_case(record) == snapshot
    assert original == snapshot
    assert len(record.task["input"]["groups"]) == (2 if sequential else 1)
    with pytest.raises(ValueError, match="not a ready model request"):
        record.request_body()


def test_unknown_question_field_fails_closed_instead_of_leaking_labels():
    row = {
        "original_group": {
            "video_path": "video.mp4",
            "questions": [
                {"question": "query", "time_stamp": "00:00:01", "future_label": "GOLD"}
            ],
        }
    }
    with pytest.raises(ValueError, match="unrecognized"):
        adapt_row(
            "STREAMINGBENCH",
            row,
            source={"source_id": "STREAMINGBENCH"},
            spec=adapter_spec("STREAMINGBENCH"),
            index=0,
        )


def test_restore_rejects_truncated_reference_sequence():
    row = {
        "original_group": {
            "video_path": "video.mp4",
            "questions": [
                {"question": "query", "time_stamp": "00:00:01", "answer": "A"}
            ],
        }
    }
    record = adapt_row(
        "STREAMINGBENCH",
        row,
        source={"source_id": "STREAMINGBENCH"},
        spec=adapter_spec("STREAMINGBENCH"),
        index=0,
    )
    record.outcome["references"]["groups"][0]["questions"].clear()
    with pytest.raises(ValueError):
        restore_original_case(record)
