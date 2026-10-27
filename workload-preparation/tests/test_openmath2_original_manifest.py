"""Metadata and unchanged raw corpus projection, no native model evidence."""

from vllm_hust_workload_data.adapters import adapt_row
from vllm_hust_workload_data.registry import adapter_spec, source_spec


def test_canonical_train_shards_complete_and_overlap_not_counted_as_new_data():
    spec = source_spec("OPENMATH2")
    assert len(spec["files"]) == 32
    assert sum(row["bytes"] for row in spec["files"]) == 7576089560
    assert {row["path"] for row in spec["files"]} == {
        f"data/train-{number:05d}-of-00032.parquet" for number in range(32)
    }
    assert all(
        row["split"] == "train" and row["role"] == "data" for row in spec["files"]
    )
    assert spec["access"]["gated"] is False
    assert spec["access"]["license"] == "cc-by-4.0"


def test_raw_problem_is_separate_from_teacher_and_expected_answer():
    row = {
        "problem": "fixture problem",
        "expected_answer": "",
        "generated_solution": "fixture teacher solution",
        "problem_source": "math",
    }
    record = adapt_row(
        "OPENMATH2",
        row,
        source={"source_id": "OPENMATH2", "file_sha256": "fixture-file-identity"},
        spec=adapter_spec("OPENMATH2"),
        index=0,
    )
    assert record.task["input"] == {"problem": "fixture problem"}
    assert record.outcome["expected_answer"] == ""
    assert record.outcome["generated_solution"] == "fixture teacher solution"
    assert record.replay["annotations"]["problem_source"] == "math"
