"""Synthetic software fixtures only; no original data download or model calls."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from vllm_hust_workload_data import iter_workload
from vllm_hust_workload_data.registry import source_spec
from vllm_hust_workload_data.pipeline import prepared_rows, run
from vllm_hust_workload_data.consumer import resolve_input


SPEC = importlib.util.spec_from_file_location(
    "ruler_binding",
    Path(__file__).resolve().parents[2] / "scripts/bind_ruler_original_profile.py",
)
binding = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(binding)


def fixture(tmp_path):
    rows = [
        {
            "index": 17,
            "input": "complete fixture question",
            "outputs": ["REFERENCE_SENTINEL"],
        }
    ] * 2
    (tmp_path / "data").mkdir()
    path = tmp_path / "data/fixture.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    generation = deepcopy(source_spec("RULER")["files"][0]["generation"])
    generation["num_samples"] = 2
    return path, generation


def test_complete_binding_prepare_and_resolve_with_duplicate_indices(tmp_path):
    path, generation = fixture(tmp_path)
    raw = path.read_bytes()
    manifest = binding.bind(path.parent, path.name, generation)
    assert manifest["raw_data_copied"] is False
    assert not manifest["generation_independently_reexecuted_by_binding"]
    assert not manifest["native_execution_or_scoring_verified_by_binding"]
    assert (
        manifest["files"][0]["identity_assurance"]
        == "caller_selected_local_generated_snapshot_not_publisher_file_attestation"
    )
    records = list(iter_workload("RULER", data_root=path.parent, manifest=manifest))
    assert (
        len(records) == 2 and records[0].task["task_id"] != records[1].task["task_id"]
    )
    target = tmp_path / "manifest.json"
    target.write_text(json.dumps(manifest), encoding="utf-8")
    run(
        {
            "sources": [
                {
                    "source_id": "RULER",
                    "data_root": str(path.parent),
                    "manifest": str(target),
                }
            ]
        },
        output=tmp_path / "prepared",
    )
    db = tmp_path / "prepared/workload.sqlite"
    for ref, record in zip(prepared_rows(db), records, strict=True):
        actual = resolve_input(db, ref["record_id"], data_root=path.parent)
        assert actual.input_record() == record.input_record()
        assert actual.outcome_record() == record.outcome_record()
        assert "REFERENCE_SENTINEL" not in json.dumps(actual.input_record())
    assert path.read_bytes() == raw


@pytest.mark.parametrize(
    "key,value",
    [
        ("num_samples", 3),
        ("task", "made-up-task"),
        ("task", []),
        ("tokenizer_type", {}),
        ("random_seed", True),
        ("max_seq_length", 0),
        ("tokenizer_revision", ""),
        ("subset", "made-up-split"),
        ("source_commit", "wrong-public-commit"),
    ],
)
def test_malformed_or_mislabelled_profile_rejected(tmp_path, key, value):
    path, generation = fixture(tmp_path)
    generation[key] = value
    with pytest.raises(ValueError):
        binding.bind(path.parent, path.name, generation)


def test_missing_provenance_rejected(tmp_path):
    path, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="complete original generator"):
        binding.bind(path.parent, path.name, {"task": "niah_single_1"})


def test_known_identical_bytes_cannot_be_relabelled(tmp_path, monkeypatch):
    path, generation = fixture(tmp_path)
    existing = binding.bind(path.parent, path.name, generation)
    monkeypatch.setattr(binding, "source_spec", lambda *args: existing)
    generation["task"] = "vt"
    with pytest.raises(ValueError, match="conflicts"):
        binding.bind(path.parent, path.name, generation)


def test_full_reader_count_check_is_not_optional(tmp_path, monkeypatch):
    path, generation = fixture(tmp_path)
    monkeypatch.setattr(binding, "iter_workload", lambda *a, **kw: iter([]))
    with pytest.raises(ValueError, match="reader count mismatch"):
        binding.bind(path.parent, path.name, generation)


def test_cli_writes_only_fresh_manifest_and_never_original_data(tmp_path, monkeypatch):
    path, generation = fixture(tmp_path)
    original = path.read_bytes()
    metadata = tmp_path / "generation.json"
    metadata.write_text(json.dumps(generation), encoding="utf-8")
    target = tmp_path / "manifest.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bind-ruler",
            "--data-root",
            str(path.parent),
            "--file",
            path.name,
            "--generation",
            str(metadata),
            "--output",
            str(target),
        ],
    )
    binding.main()
    saved = target.read_bytes()
    assert json.loads(saved)["files"][0]["rows"] == 2
    with pytest.raises(FileExistsError):
        binding.main()
    assert target.read_bytes() == saved and path.read_bytes() == original


def test_undeclared_original_field_is_not_silently_dropped(tmp_path):
    path, generation = fixture(tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["unreviewed_original_field"] = "fixture value"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="undeclared original source fields"):
        binding.bind(path.parent, path.name, generation)


def test_reviewed_generator_compatibility_metadata_is_preserved(tmp_path):
    path, generation = fixture(tmp_path)
    metadata = {
        "patch_id": "ruler-qa-required-doc-floor/v1",
        "target_relative_path": "scripts/data/synthetic/qa.py",
        "original_source_sha256": "a" * 64,
        "runtime_source_sha256": "b" * 64,
        "explicit_opt_in": True,
        "question_renderer_ast_preserved": True,
    }
    generation["generator_compatibility"] = metadata
    result = binding.bind(path.parent, path.name, generation)
    assert result["files"][0]["generation"]["generator_compatibility"] == metadata
    metadata["explicit_opt_in"] = False
    with pytest.raises(ValueError, match="compatibility metadata"):
        binding.bind(path.parent, path.name, generation)
