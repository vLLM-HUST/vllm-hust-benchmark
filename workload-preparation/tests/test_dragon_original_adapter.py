"""Tiny artificial four-file relations; actual native benchmark evidence is separate."""

from copy import deepcopy
import hashlib

import pytest

from vllm_hust_workload_data import dragon, iter_workload
from vllm_hust_workload_data.registry import source_spec


@pytest.fixture
def original_fixture(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows = {
        "public_questions": [
            {"id": 0, "question": "fixture question0"},
            {"id": 1, "question": "fixture question1"},
        ],
        "public_texts": [{"id": 5, "text": "fixture corpus"}],
        "reference_qa": [
            {
                "id": 11,
                "public_id": 1,
                "text_ids": "[[7]]",
                "text": "fixture reference context1",
                "url": "https://example.invalid/1",
                "question": "fixture question1",
                "answer": "fixture reference1",
                "type": "fixture",
            },
            {
                "id": 10,
                "public_id": 0,
                "text_ids": "[7]",
                "text": "fixture reference context0",
                "url": "https://example.invalid/0",
                "question": "fixture question0",
                "answer": "fixture reference0",
                "type": "fixture",
            },
        ],
        "reference_texts": [{"id": 7, "public_id": 5, "text": "fixture corpus"}],
    }
    spec = deepcopy(source_spec("DRAGON_PUBLIC"))
    spec["files"] = []

    def write():
        spec["files"].clear()
        for kind, values in rows.items():
            relative = f"{kind}/data/train-00000-of-00001.parquet"
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(values), path)
            spec["files"].append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "role": "data"
                    if kind == "public_questions"
                    else "native_environment",
                }
            )
        return tmp_path / "public_questions/data/train-00000-of-00001.parquet"

    path = write()
    monkeypatch.setattr(dragon, "source_spec", lambda _: spec)
    return path, spec, rows, write


def test_exact_public_ids_not_reference_row_order_and_no_answer_in_input(
    original_fixture,
):
    path, spec, _, _ = original_fixture
    records = list(
        iter_workload("DRAGON_PUBLIC", data_root=path.parents[2], manifest=spec)
    )
    assert len(records) == 2
    assert records[0].task["task_id"] == "0"
    assert records[0].task["input"] == {"id": 0, "question": "fixture question0"}
    assert records[0].outcome["original_reference_qa"]["id"] == 10
    assert records[0].outcome["original_reference_qa"]["answer"] == "fixture reference0"
    assert "answer" not in records[0].task["input"]
    assert records[0].runtime["execution"] == "not_executed"


@pytest.mark.parametrize("problem", ["question", "text", "reference_id", "duplicate"])
def test_invalid_full_original_relation_rejected_even_for_first_one(
    original_fixture, problem
):
    path, _, rows, write = original_fixture
    if problem == "question":
        rows["reference_qa"][0]["question"] = "different"
    elif problem == "text":
        rows["reference_texts"][0]["text"] = "different"
    elif problem == "reference_id":
        rows["reference_qa"][0]["text_ids"] = "[999]"
    else:
        rows["reference_qa"][0]["public_id"] = 0
    write()
    with pytest.raises(ValueError):
        next(dragon.original_rows(path))


def test_reference_asset_mutation_after_yield_is_detected(original_fixture):
    path, _, _, _ = original_fixture
    rows = dragon.original_rows(path)
    next(rows)
    reference = path.parents[2] / "reference_qa/data/train-00000-of-00001.parquet"
    reference.write_bytes(reference.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="asset identity mismatch"):
        rows.close()
