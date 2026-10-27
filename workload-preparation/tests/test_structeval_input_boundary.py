"""Synthetic software fixture: evaluator answers never enter input envelopes."""

import json

from vllm_hust_workload_data.adapters import adapt_row
from vllm_hust_workload_data.registry import adapter_spec


def test_vqa_answers_are_scorer_only_not_replay_annotations():
    row = {
        "task_id": "fixture",
        "task_name": "Text to JSON",
        "query": "Return an object.",
        "feature_requirements": "field required",
        "VQA": [
            {"question": "fixture visual question", "answer": "SCORER_ONLY_FIXTURE"}
        ],
        "raw_output_metric": ["SCORER_PATH_FIXTURE"],
        "rendering": False,
    }
    record = adapt_row(
        "STRUCTEVAL",
        row,
        source={"source_id": "STRUCTEVAL"},
        spec=adapter_spec("STRUCTEVAL"),
        index=0,
    )
    serialized = json.dumps(record.input_record())
    assert "SCORER_ONLY_FIXTURE" not in serialized
    assert "SCORER_PATH_FIXTURE" not in serialized
    assert "VQA" not in record.replay["annotations"]
    assert record.outcome_record()["VQA"] == row["VQA"]
    assert record.task["input"]["query"] == row["query"]
