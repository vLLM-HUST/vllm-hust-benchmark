"""Synthetic MATH500 contract fixtures; not original task or score evidence."""

from copy import deepcopy
import hashlib
import json

import pytest

from vllm_hust_workload_data import iter_workload
from vllm_hust_workload_data.adapters import adapt_row
from vllm_hust_workload_data.consumer import resolve_input
from vllm_hust_workload_data.pipeline import prepared_rows, run
from vllm_hust_workload_data.registry import adapter_spec, source_spec


def fixture_rows():
    return [
        {
            "unique_id": f"fixture/problem-{index}",
            "problem": f"Invented problem {index}: compute {index} + 1.",
            "solution": f"Invented derivation {index}.",
            "answer": str(index + 1),
            "subject": "algebra",
            "level": index + 1,
        }
        for index in range(3)
    ]


def test_math500_projects_original_problem_and_isolates_reference():
    row = fixture_rows()[0]
    before = deepcopy(row)
    record = adapt_row(
        "MATH500",
        row,
        source={
            "source_id": "MATH500",
            "revision": "synthetic-unit-fixture-not-source-evidence",
            "file": "fixture.jsonl",
            "file_sha256": "a" * 64,
        },
        spec=adapter_spec("MATH500"),
        index=0,
    )
    assert record.task["input"] == {"problem": row["problem"]}
    assert record.outcome_record() == {
        "answer": "1",
        "solution": "Invented derivation 0.",
    }
    assert record.replay["annotations"] == {
        "unique_id": "fixture/problem-0",
        "subject": "algebra",
        "level": 1,
    }
    assert "Invented derivation" not in json.dumps(record.input_record())
    assert '"answer"' not in json.dumps(record.input_record())
    assert row == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("unique_id", " "),
        ("problem", ""),
        ("solution", ""),
        ("answer", ""),
        ("subject", ""),
        ("level", True),
        ("level", 0),
        ("level", 6),
    ],
)
def test_invalid_math500_fields_fail_closed(field, value):
    row = fixture_rows()[0]
    row[field] = value
    with pytest.raises(ValueError, match="MATH500 requires"):
        adapt_row(
            "MATH500",
            row,
            source={"source_id": "MATH500", "file_sha256": "a" * 64},
            spec=adapter_spec("MATH500"),
            index=0,
        )


def test_three_synthetic_rows_round_trip_through_preparation_and_consumer(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    path = data / "fixture.jsonl"
    raw_rows = fixture_rows()
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in raw_rows), encoding="utf-8"
    )
    manifest = deepcopy(source_spec("MATH500"))
    manifest["revision"] = "synthetic-unit-fixture-not-source-evidence"
    manifest["files"] = [
        {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
            "rows": len(raw_rows),
            "role": "data",
        }
    ]
    original = list(
        iter_workload("MATH500", data_root=data, manifest=manifest, limit=3)
    )
    assert len(original) == 3
    assert [record.task["task_id"] for record in original] == [
        row["unique_id"] for row in raw_rows
    ]

    manifest_path = tmp_path / "binding.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    run(
        {
            "sources": [
                {
                    "source_id": "MATH500",
                    "data_root": str(data),
                    "manifest": str(manifest_path),
                    "limit": 3,
                }
            ]
        },
        output=tmp_path / "prepared",
    )
    database = tmp_path / "prepared/workload.sqlite"
    refs = list(prepared_rows(database))
    assert len(refs) == 3
    for ref, expected in zip(refs, original):
        actual = resolve_input(database, ref["record_id"], data_root=data)
        assert actual.input_record() == expected.input_record()
        assert actual.outcome_record() == expected.outcome_record()
        assert "answer" not in actual.input_record()["task"]["input"]
        assert "solution" not in actual.input_record()["task"]["input"]
