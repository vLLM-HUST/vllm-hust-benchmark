"""Invented LoCoMo-shaped fixtures, never original source or execution evidence."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from vllm_hust_workload_data import iter_workload
from vllm_hust_workload_data.consumer import resolve_input
from vllm_hust_workload_data.locomo import question_rows
from vllm_hust_workload_data.pipeline import prepared_rows, run
from vllm_hust_workload_data.registry import source_spec


def parents():
    return [
        {
            "sample_id": "fixture-conversation",
            "conversation": {
                "speaker_a": "A",
                "speaker_b": "B",
                "session_1_date_time": "original local date, no timezone assertion",
                "session_1": [
                    {
                        "speaker": "A",
                        "text": " original\nturn ",
                        "dia_id": "D1:1",
                        "img_url": "https://example.invalid/image",
                        "blip_caption": "publisher caption",
                    }
                ],
            },
            "qa": [
                {
                    "question": "original question",
                    (
                        "adversarial_answer" if category == 5 else "answer"
                    ): "REFERENCE_SENTINEL",
                    "category": category,
                    "evidence": ["D1:1"],
                }
                for category in (2, 3, 5)
            ],
            "observation": {"private": "OBSERVATION_SENTINEL"},
            "session_summary": {"private": "SUMMARY_SENTINEL"},
            "event_summary": {"private": "EVENT_GOLD_SENTINEL"},
        }
    ]


def test_expansion_preserves_complete_context_original_positions_and_isolation():
    original = parents()
    frozen = deepcopy(original)
    rows = list(question_rows(original))
    assert len(rows) == 3
    assert [r["source_question_index"] for r in rows] == [0, 1, 2]
    assert all(r["source_parent_row_index"] == 0 for r in rows)
    assert all(r["conversation"] == original[0]["conversation"] for r in rows)
    assert all(
        not {"qa", "observation", "session_summary", "event_summary"} & r.keys()
        for r in rows
    )
    assert "answer" not in rows[2]  # Do not rename publisher adversarial_answer.
    assert rows[2]["adversarial_answer"] == "REFERENCE_SENTINEL"
    rows[0]["conversation"]["session_1"][0]["text"] = "mutated output"
    assert original == frozen


@pytest.mark.parametrize(
    "failure", ["duplicate_id", "date", "turn", "category", "collision", "empty"]
)
def test_malformed_relations_fail_without_exposing_payload(failure):
    data = parents()
    if failure == "duplicate_id":
        data.append(deepcopy(data[0]))
    elif failure == "date":
        data[0]["conversation"].pop("session_1_date_time")
    elif failure == "turn":
        data[0]["conversation"]["session_1"][0]["text"] = None
    elif failure == "category":
        data[0]["qa"][0]["category"] = True
    elif failure == "collision":
        data[0]["qa"][0]["conversation"] = "PRIVATE_SENTINEL"
    else:
        data[0]["qa"] = []
    with pytest.raises(ValueError) as error:
        list(question_rows(data))
    assert "SENTINEL" not in str(error.value)


def test_source_pipeline_and_resolver_use_same_question_view(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    path = data / "fixture.json"
    path.write_text(json.dumps(parents()), encoding="utf-8")
    manifest = deepcopy(source_spec("LOCOMO"))
    manifest["revision"] = "synthetic-unit-fixture-not-original"
    manifest["files"] = [
        {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
            "rows": 3,
            "role": "data",
        }
    ]
    original = list(iter_workload("LOCOMO", data_root=data, manifest=manifest, limit=3))
    for record in original:
        assert set(record.task["input"]) == {"conversation", "question"}
        assert "SENTINEL" not in json.dumps(record.input_record())
        outcome = record.outcome_record()
        assert (
            outcome.get("answer", outcome.get("adversarial_answer"))
            == "REFERENCE_SENTINEL"
        )
    binding = tmp_path / "binding.json"
    binding.write_text(json.dumps(manifest), encoding="utf-8")
    run(
        {
            "sources": [
                {
                    "source_id": "LOCOMO",
                    "data_root": str(data),
                    "manifest": str(binding),
                    "limit": 3,
                }
            ]
        },
        output=tmp_path / "prepared",
    )
    database = tmp_path / "prepared/workload.sqlite"
    references = list(prepared_rows(database))
    assert len(references) == 3
    for ref, expected in zip(references, original):
        actual = resolve_input(database, ref["record_id"], data_root=data)
        assert actual.input_record() == expected.input_record()
        assert actual.outcome_record() == expected.outcome_record()


def test_original_reader_failure_is_preserved_and_recovery_sample_is_scored():
    root = Path(__file__).resolve().parents[2]
    docs = root / "docs/workload-data"
    manifest = source_spec("LOCOMO")
    assert manifest["files"][0]["rows"] == 1986
    assert manifest["files"][0]["original_conversations"] == 10
    assert manifest["unit"] == "original_questions"
    assert manifest["access"]["license"] == "CC-BY-NC-4.0"
    reader = json.loads(
        (docs / "validation/locomo-original-reader-20261023.json").read_text("utf-8")
    )
    failure = json.loads(
        (docs / "validation/locomo-original-qa-failure-20261023.json").read_text(
            "utf-8"
        )
    )
    recovery = json.loads(
        (docs / "validation/locomo-original-qa-recovery-20261023.json").read_text(
            "utf-8"
        )
    )
    recovery_batch2 = json.loads(
        (
            docs / "validation/locomo-original-qa-recovery-batch2-20261023.json"
        ).read_text("utf-8")
    )
    sampled = json.loads(
        (docs / "validation/locomo-original-qa-sampled-20261023.json").read_text(
            "utf-8"
        )
    )
    ledger = json.loads((docs / "dataset-validation-status.json").read_text("utf-8"))
    row = next(
        source for source in ledger["sources"] if source["source_id"] == "LOCOMO"
    )
    assert reader["original_questions_field_checked"] == 1986
    assert reader["category_5_reference_only_records"] == 444
    assert reader["model_calls"] == reader["scorer_runs"] == 0
    assert failure["completed_model_generations"] == failure["scorer_runs"] == 0
    assert failure["model_requests_attempted"] == 1
    assert (
        reader["source_checkbox_eligible"]
        is failure["source_checkbox_eligible"]
        is False
    )
    assert recovery["status"] == "END_TO_END_EXECUTED_AND_SCORED"
    assert recovery["selected_samples"] == [0, 1, 2]
    assert recovery["scoring"] == "COMPLETED"
    assert [sample["finish_reason"] for sample in recovery["samples"]] == [
        "stop",
        "stop",
        "stop",
    ]
    assert [sample["publisher_f1"] for sample in recovery["samples"]] == [
        0.6666666666666666,
        0.0,
        0.28571428571428575,
    ]
    assert recovery["aggregate"]["publisher_f1_mean"] == 0.31746031746031744
    assert recovery["raw_requests_and_model_responses_published"] is False
    assert recovery_batch2["selected_samples"] == [3, 4, 5]
    assert [sample["publisher_f1"] for sample in recovery_batch2["samples"]] == [
        1.0,
        0.5,
        0.7499999999999999,
    ]
    assert sampled["selected_samples"] == list(range(6))
    assert sampled["aggregate"]["publisher_f1_mean"] == 0.5337301587301587
    assert len(sampled["samples"]) == 6
    assert row["checked"] is True
    assert row["selected_samples"] == list(range(6))
    assert (
        row["receipt"]
        == "docs/workload-data/validation/locomo-original-qa-sampled-20261023.json"
    )
    assert row["execution_attempt_receipts"] == [
        "docs/workload-data/validation/locomo-original-qa-recovery-20261023.json",
        "docs/workload-data/validation/locomo-original-qa-recovery-batch2-20261023.json",
    ]
    assert (
        row["prior_failure_receipt"]
        == "docs/workload-data/validation/locomo-original-qa-failure-20261023.json"
    )
    assert row["blocker"].find("第5类") >= 0
