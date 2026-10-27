"""Synthetic CPU fixtures for native CSV preparation and reference isolation."""

from copy import deepcopy
import csv
import hashlib
import io
from random import Random
from zipfile import ZipFile

import pytest

from vllm_hust_workload_data.adapters import adapt_row
from vllm_hust_workload_data.gpqa import prepare_original_archive, seed_zero_choices
from vllm_hust_workload_data.registry import adapter_spec, source_spec


def example():
    return {
        "Question": "fixture question",
        "Correct Answer": "correct option",
        "Incorrect Answer 1": "wrong1",
        "Incorrect Answer 2": "wrong2",
        "Incorrect Answer 3": "wrong3",
        "Explanation": "reference only",
        "Canary String": "fixture canary",
        "Record ID": "original fixture id",
    }


def test_per_file_seed0_matches_sequential_publisher_shuffle_out_of_order():
    row = example()
    random = Random(0)
    expected = []
    for _ in range(100):
        choices = [
            row[k]
            for k in [
                "Incorrect Answer 1",
                "Incorrect Answer 2",
                "Incorrect Answer 3",
                "Correct Answer",
            ]
        ]
        random.shuffle(choices)
        expected.append(choices)
    for i in [99, 1, 0, 69, 5, 42]:
        choices, correct = seed_zero_choices(row, i)
        assert choices == expected[i] and choices[correct] == row["Correct Answer"]


def test_raw_fields_reconstruct_without_exposing_answer_keys_or_annotations():
    row = example()
    original = deepcopy(row)
    record = adapt_row(
        "GPQA",
        row,
        source={"source_id": "GPQA", "file_sha256": "fixture"},
        spec=adapter_spec("GPQA"),
        index=7,
    )
    assert row == original
    assert set(record.task["input"]) == {"Question", "choices"}
    assert record.replay["unmapped_field_names"] == []
    outcome = record.outcome_record()
    assert (
        record.task["input"]["choices"][outcome["correct_index"]]
        == row["Correct Answer"]
    )
    assert {
        "Question": record.task["input"]["Question"],
        **{k: v for k, v in outcome.items() if k != "correct_index"},
    } == row
    assert record.task["task_id"] == "fixture:7"


def test_invalid_row_or_offset_does_not_invent_choices():
    for offset in [-1, True]:
        with pytest.raises(ValueError):
            seed_zero_choices(example(), offset)
    row = example()
    row["Correct Answer"] = ""
    with pytest.raises(ValueError):
        seed_zero_choices(row, 0)


def test_manifest_has_three_question_splits_and_explicit_original_archive():
    spec = source_spec("GPQA")
    assert spec["platform"] == "github" and spec["preparation_command"]
    assert {f["rows"] for f in spec["files"] if f["role"] == "data"} == {448, 198, 546}
    assert len([f for f in spec["files"] if f["role"] == "source_archive"]) == 1


def test_prepare_preserves_exact_csv_bytes_and_ignores_other_members(
    tmp_path, monkeypatch
):
    row = example()
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)
    raw = stream.getvalue().encode()
    archive = tmp_path / "fixture.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("dataset/gpqa_main.csv", raw)
        zipped.writestr("../must-not-extract.txt", "fixture")
    source = {
        "revision": "fixture",
        "files": [
            {
                "role": "source_archive",
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            },
            {
                "role": "data",
                "path": "gpqa_main.csv",
                "archive_member": "dataset/gpqa_main.csv",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "rows": 1,
            },
        ],
    }
    monkeypatch.setattr(
        "vllm_hust_workload_data.registry.source_spec", lambda _: source
    )
    output = tmp_path / "prepared"
    receipt = prepare_original_archive(data_archive=archive, output=output)
    assert (output / "gpqa_main.csv").read_bytes() == raw and receipt[
        "model_calls"
    ] == 0
    assert not (tmp_path / "must-not-extract.txt").exists()
    with pytest.raises(ValueError, match="fresh"):
        prepare_original_archive(data_archive=archive, output=output)
