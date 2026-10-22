"""Invented parent/child fixtures, not original QuALITY task evidence."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from vllm_hust_workload_data import iter_workload
from vllm_hust_workload_data.bindings import bind_source
from vllm_hust_workload_data.consumer import resolve_input
from vllm_hust_workload_data.pipeline import prepared_rows, run
from vllm_hust_workload_data.quality import question_rows
from vllm_hust_workload_data.registry import source_spec


def question(index):
    return {
        "question_unique_id": f"fixture-q{index}",
        "question": f"question {index}",
        "options": ["option one", "option two", "option three", "option four"],
        "gold_label": 2,
        "writer_label": 2,
        "validation": [{"private_reference": "ANNOTATION_SENTINEL"}],
        "difficult": 1,
    }


def parents():
    return [
        {
            "article_id": "fixture-a",
            "set_unique_id": "fixture-set-a",
            "article": " complete\noriginal article ",
            "license": "UNIT_FIXTURE",
            "questions": [question(0), question(1)],
        },
        {
            "article_id": "fixture-b",
            "set_unique_id": "fixture-set-b",
            "article": "second original article",
            "license": "UNIT_FIXTURE",
            "questions": [question(2)],
        },
    ]


def test_question_expansion_retains_parent_child_identity_without_modification():
    original = parents()
    before = deepcopy(original)
    rows = list(question_rows(original))
    assert len(rows) == 3
    assert [
        (r["source_parent_row_index"], r["source_question_index"]) for r in rows
    ] == [(0, 0), (0, 1), (1, 0)]
    assert rows[0]["article"] == rows[1]["article"] == before[0]["article"]
    assert all("questions" not in row for row in rows)
    rows[0]["options"].append("mutate result")
    assert original == before


@pytest.mark.parametrize(
    "failure", ["duplicate_id", "collision", "reserved", "empty", "not_object"]
)
def test_invalid_question_relation_fails_without_payload(failure):
    rows = parents()
    if failure == "duplicate_id":
        rows[1]["questions"][0]["question_unique_id"] = "fixture-q0"
    elif failure == "collision":
        rows[0]["questions"][0]["article"] = "PRIVATE_SENTINEL"
    elif failure == "reserved":
        rows[0]["source_parent_row_index"] = 42
    elif failure == "empty":
        rows[0]["questions"] = []
    else:
        rows[0]["questions"][0] = "PRIVATE_SENTINEL"
    with pytest.raises(ValueError) as error:
        list(question_rows(rows))
    assert "PRIVATE_SENTINEL" not in str(error.value)


def test_read_bind_pipeline_and_resolve_share_question_view(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    path = data / "fixture.dev"
    path.write_text("".join(json.dumps(r) + "\n" for r in parents()), encoding="utf-8")
    manifest = deepcopy(source_spec("QUALITY"))
    manifest["revision"] = "synthetic-unit-fixture-not-original-data"
    manifest["files"] = [
        {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": 3,
            "role": "data",
        }
    ]
    original = list(iter_workload("QUALITY", data_root=data, manifest=manifest))
    assert len(original) == 3
    for record in original:
        assert set(record.task["input"]) == {"article", "question", "options"}
        assert "ANNOTATION_SENTINEL" not in json.dumps(record.input_record())
        assert record.outcome_record()["gold_label"] == 2
    binding = bind_source(
        "QUALITY", data_root=data, files=[path.name], revision=manifest["revision"]
    )
    assert binding["files"][0]["rows"] == 3  # questions, not two JSONL sets
    bound = tmp_path / "binding.json"
    bound.write_text(json.dumps(binding), encoding="utf-8")
    expected = list(iter_workload("QUALITY", data_root=data, manifest=binding, limit=3))
    run(
        {
            "sources": [
                {
                    "source_id": "QUALITY",
                    "data_root": str(data),
                    "manifest": str(bound),
                    "limit": 3,
                }
            ]
        },
        output=tmp_path / "index",
    )
    db = tmp_path / "index/workload.sqlite"
    refs = list(prepared_rows(db))
    assert len(refs) == 3
    for reference, record in zip(refs, expected):
        actual = resolve_input(db, reference["record_id"], data_root=data)
        assert actual.input_record() == record.input_record()
        assert actual.outcome_record() == record.outcome_record()


@pytest.mark.parametrize("value", [True, 0, 5, "2", None])
def test_invalid_gold_label_not_silently_coerced(tmp_path, value):
    from vllm_hust_workload_data.adapters import adapt_row
    from vllm_hust_workload_data.registry import adapter_spec

    row = next(question_rows(parents()))
    row["gold_label"] = value
    with pytest.raises(ValueError):
        adapt_row("QUALITY", row, source={}, spec=adapter_spec("QUALITY"), index=0)


def test_reader_receipt_does_not_become_native_task_acceptance():
    root = Path(__file__).resolve().parents[2]
    receipt = json.loads(
        (
            root / "docs/workload-data/validation/quality-original-reader-20261023.json"
        ).read_text(encoding="utf-8")
    )
    spec = source_spec("QUALITY")
    assert spec["unit"] == "original_questions"
    assert (
        spec["files"][0]["rows"]
        == receipt["original_questions_read_checked_and_indexed"]
        == 2086
    )
    assert (
        spec["files"][0]["parent_rows"] == receipt["original_jsonl_parent_rows"] == 230
    )
    assert spec["files"][0]["sha256"] == receipt["file_sha256"]
    assert receipt["distinct_articles"] == 115
    assert receipt["input_and_reference_roundtrips"] == 3
    assert receipt["model_calls"] == receipt["scorer_runs"] == 0
    assert receipt["source_checkbox_eligible"] is False
    assert receipt["full_native_suite_validated"] is False
    ledger = json.loads(
        (root / "docs/workload-data/dataset-validation-status.json").read_text(
            encoding="utf-8"
        )
    )
    row = next(r for r in ledger["sources"] if r["source_id"] == "QUALITY")
    assert row["reader_receipt"].endswith("quality-original-reader-20261023.json")
    assert row["receipt"].endswith("quality-original-mc-npu-20261023.json")
    native = json.loads((root / row["receipt"]).read_text(encoding="utf-8"))
    assert (
        native["unique_original_questions"]
        == 3
        < receipt["original_questions_read_checked_and_indexed"]
    )
    assert native["full_suite_validated"] is False
    assert row["checked"] is False
