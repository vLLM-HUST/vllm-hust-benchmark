"""Synthetic software fixtures; no publisher data or generation in CI."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from vllm_hust_workload_data import iter_workload
from vllm_hust_workload_data.adapters import adapt_row
from vllm_hust_workload_data.registry import adapter_spec, source_spec
from vllm_hust_workload_data.catalog import download_plan
from vllm_hust_workload_data import bindings


def fixture_row():
    return {
        "index": 17,
        "input": "Complete fixture context and original question",
        "answer_prefix": " Answer:",
        "outputs": ["REFERENCE_SENTINEL"],
        "length": 40,
        "length_w_model_temp": 40,
        "token_position_answer": 13,
        "truncation": -1,
        "others": {"id": "original-relation"},
    }


def test_ruler_reference_isolated_and_original_index_not_task_identity():
    row = fixture_row()
    source = {"source_id": "RULER", "file_sha256": "fixture-digest"}
    record = adapt_row("RULER", row, source=source, spec=adapter_spec("RULER"), index=3)
    assert record.task["task_id"] == "fixture-digest:3"
    assert record.task["identity_kind"] == "file_sha256_row_index"
    assert record.task["input"] == {
        "input": row["input"],
        "answer_prefix": row["answer_prefix"],
    }
    assert "REFERENCE_SENTINEL" not in json.dumps(record.input_record())
    assert record.outcome_record() == {
        "outputs": row["outputs"],
        "index": 17,
        "token_position_answer": 13,
        "others": row["others"],
    }
    assert "token_position_answer" not in json.dumps(record.input_record())
    assert record.replay["annotations"]["length_w_model_temp"] == 40
    assert record.replay["unmapped_field_names"] == []
    assert record.outcome_record()["others"] == row["others"]
    with pytest.raises(ValueError, match="not a ready model request"):
        record.request_body()


def test_full_reader_duplicate_indices_no_drop_no_mutation(tmp_path):
    rows = [fixture_row(), fixture_row(), fixture_row()]
    rows[1].pop("answer_prefix")
    path = tmp_path / "validation.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    raw = path.read_bytes()
    manifest = deepcopy(source_spec("RULER"))
    manifest["files"] = [
        {
            "path": path.name,
            "role": "data",
            "rows": 3,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "generation": {**manifest["files"][0]["generation"], "num_samples": 3},
        }
    ]
    records = list(iter_workload("RULER", data_root=tmp_path, manifest=manifest))
    assert len(records) == 3
    assert len({r.task["task_id"] for r in records}) == 3
    assert [r.outcome_record()["index"] for r in records] == [17] * 3
    assert "answer_prefix" not in records[1].task["input"]
    assert path.read_bytes() == raw
    assert all(
        record.source["generation"] == manifest["files"][0]["generation"]
        for record in records
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("index", True),
        ("input", ""),
        ("answer_prefix", 0),
        ("outputs", []),
        ("outputs", [None]),
    ],
)
def test_invalid_generated_contract_rejected(key, value):
    row = fixture_row()
    row[key] = value
    with pytest.raises(ValueError):
        adapt_row(
            "RULER",
            row,
            source={"source_id": "RULER", "file_sha256": "fixture"},
            spec=adapter_spec("RULER"),
            index=0,
        )


def test_default_profile_provenance_not_natural_request_or_complete_suite():
    source = source_spec("RULER")
    assert source["provenance_kind"] == "synthetic_long_context_probe"
    assert len(source["files"]) == 1
    profile = source["files"][0]
    assert len(profile["sha256"]) == 64
    assert profile["generation"]["num_samples"] == profile["rows"] == 500
    assert profile["generation"]["task"] == "niah_single_1"
    assert profile["generation"]["max_seq_length"] == 4096
    plan = download_plan("RULER")
    assert plan["method"] == "ORIGINAL_PUBLISHER_GENERATOR"
    assert plan["files"][0]["generation"] == profile["generation"]
    assert "url" not in plan["files"][0]


def test_bind_preserves_known_generation_and_rejects_guessing_new_profiles(
    tmp_path, monkeypatch
):
    path = tmp_path / "fixture.jsonl"
    path.write_text(json.dumps(fixture_row()), encoding="utf-8")
    manifest = deepcopy(source_spec("RULER"))
    profile = manifest["files"][0]
    profile.update(
        path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=1
    )
    profile["generation"]["num_samples"] = 1
    monkeypatch.setattr(bindings, "source_spec", lambda *a: manifest)
    actual = bindings.bind_source(
        "RULER", data_root=tmp_path, files=[path.name], revision=manifest["revision"]
    )
    assert actual["files"][0]["generation"] == profile["generation"]
    assert len(list(iter_workload("RULER", data_root=tmp_path, manifest=actual))) == 1
    with pytest.raises(ValueError, match="explicit manifest"):
        bindings.bind_source(
            "RULER", data_root=tmp_path, files=[path.name], revision="different-profile"
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("tokenizer_revision", ""),
        ("random_seed", True),
        ("max_seq_length", 0),
        ("num_samples", 500),
        ("subset", None),
    ],
)
def test_reader_rejects_incomplete_or_mislabelled_generation(tmp_path, field, value):
    path = tmp_path / "fixture.jsonl"
    path.write_text(json.dumps(fixture_row()), encoding="utf-8")
    manifest = deepcopy(source_spec("RULER"))
    profile = manifest["files"][0]
    profile.update(
        path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=1
    )
    profile["generation"]["num_samples"] = 1
    profile["generation"][field] = value
    with pytest.raises(ValueError, match="generation provenance"):
        list(iter_workload("RULER", data_root=tmp_path, manifest=manifest))


def test_fixed_ci_handoff_does_not_upgrade_reading_to_native_completion():
    root = Path(__file__).resolve().parents[2]
    receipt = json.loads(
        (
            root
            / "docs/workload-data/validation/ruler-ci-research-handoff-20261025.json"
        ).read_text()
    )
    assert receipt["source_id"] == "RULER" and len(receipt["data_commit"]) == 40
    assert (
        len(receipt["data_wheel_sha256"]) == 64
        and receipt["data_cpu_ci_run"] == 40123881426
    )
    assert (
        receipt["public_prepare_resolve_rows"]
        == receipt["original_generated_rows_checked"]
        == 500
    )
    assert (
        receipt["research_source_index_import_rows"]
        == receipt["research_direct_prepare_resolve_rows"]
        == 3
    )
    assert not receipt["candidate_not_final_ci_artifact"]
    assert not receipt["research_default_dependency_lock_updated"]
    assert receipt["new_model_calls"] == 0 and not receipt["source_checkbox_eligible"]
