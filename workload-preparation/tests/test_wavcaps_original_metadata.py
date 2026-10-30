"""Invented parser fixtures, not audio, native inference or quality evidence."""

from copy import deepcopy
import hashlib
import json
import importlib.util
from pathlib import Path

import pytest

from vllm_hust_workload_data.adapters import adapt_row, iter_source_rows, iter_workload
from vllm_hust_workload_data.registry import adapter_spec, source_spec


def original_fixture():
    return {
        "id": "fixture-id",
        "audio": "unchanged-original.flac",
        "duration": 1.25,
        "caption": "WEAK_REFERENCE",
        "description": "RAW_REFERENCE",
        "title": "TITLE_REFERENCE",
        "category": "CATEGORY_REFERENCE",
        "tags": ["TAG_REFERENCE"],
        "file_name": "original-file-name.wav",
        "author": "fixture-credit",
        "href": "https://example.org/fixture",
        "download_link": "https://example.org/fixture.flac",
    }


def test_original_envelope_input_and_weak_reference_are_separate(tmp_path):
    row = original_fixture()
    payload = {"num_captions_per_audio": 1, "data": [row]}
    path = tmp_path / "original.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    original = deepcopy(payload)
    spec = adapter_spec("WAVCAPS")
    rows = list(iter_source_rows(path, spec))
    assert rows == [{"num_captions_per_audio": 1, **row}]
    source = {"source_id": "WAVCAPS", "file_sha256": "fixture", "file": path.name}
    record = adapt_row("WAVCAPS", rows[0], source=source, spec=spec, index=0)
    assert record.task["input"] == {"audio": row["audio"], "duration": 1.25}
    assert record.outcome_record() == {key: row[key] for key in spec["outcome_fields"]}
    assert "REFERENCE" not in json.dumps(record.input_record())
    assert record.replay["annotations"]["num_captions_per_audio"] == 1
    assert record.replay["unmapped_field_names"] == []
    assert json.loads(path.read_text()) == original
    with pytest.raises(ValueError, match="not a ready model request"):
        record.request_body()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("audio", ""),
        ("caption", None),
        ("duration", True),
        ("duration", -1),
        ("duration", float("inf")),
        ("num_captions_per_audio", 5),
    ],
)
def test_invalid_metadata_cannot_become_a_model_request(field, value):
    row = {"num_captions_per_audio": 1, **original_fixture(), field: value}
    with pytest.raises(ValueError):
        adapt_row(
            "WAVCAPS",
            row,
            source={"source_id": "WAVCAPS"},
            spec=adapter_spec("WAVCAPS"),
            index=0,
        )


def test_fixed_file_selection_remains_hash_verified(tmp_path):
    path = tmp_path / "original.json"
    path.write_text(
        json.dumps({"num_captions_per_audio": 1, "data": [original_fixture()]})
    )
    manifest = {
        "source_id": "WAVCAPS",
        "revision": "unit-fixture",
        "unit": "source_rows",
        "provenance_kind": "audio_caption_corpus",
        "files": [
            {
                "path": path.name,
                "role": "data",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ],
    }
    assert (
        len(list(iter_workload("WAVCAPS", data_root=tmp_path, manifest=manifest))) == 1
    )
    manifest["files"].append(
        {"path": "missing-original.json", "role": "data", "sha256": "f" * 64}
    )
    assert (
        len(
            list(
                iter_workload(
                    "WAVCAPS", data_root=tmp_path, files=[path.name], manifest=manifest
                )
            )
        )
        == 1
    )
    with pytest.raises(FileNotFoundError, match="asset missing"):
        list(iter_workload("WAVCAPS", data_root=tmp_path, manifest=manifest))
    path.write_text("{}")
    with pytest.raises(ValueError, match="identity mismatch"):
        list(iter_workload("WAVCAPS", data_root=tmp_path, manifest=manifest))


def test_catalog_pins_all_four_metadata_components_not_audio():
    spec = source_spec("WAVCAPS")
    assert spec["platform"] == "github"
    assert len(spec["files"]) == 4
    assert all(
        item["role"] == "data" and item["path"].endswith(".json")
        for item in spec["files"]
    )
    assert {item["rows"] for item in spec["files"]} == {1232, 31201, 108317, 262300}
    assert "academic-only" in spec["access"]["license"]


def test_existing_public_getter_reuses_the_explicit_metadata_profile():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/prepare_reused_research_assets.py"
    )
    module_spec = importlib.util.spec_from_file_location("wavcaps_getter", path)
    getter = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(getter)
    source, files = getter.selection("WAVCAPS")
    assert len(files) == 4
    assert source["publisher_metadata_acquisition"]["registered_file_roles"] == ["data"]
    selected = [item["path"] for item in files if "/FreeSound/" not in item["path"]]
    _, files = getter.selection("WAVCAPS", selected)
    assert len(files) == 3 and sum(item["bytes"] for item in files) == 27482307


def test_metadata_ci_receipt_cannot_promote_native_audio_delivery():
    root = Path(__file__).resolve().parents[2]
    receipt = json.loads(
        (
            root
            / "docs/workload-data/validation/wavcaps-original-metadata-ci-handoff-20261031.json"
        ).read_text()
    )
    assert receipt["source_checkbox_eligible"] is False
    assert receipt["native_audio_model_execution"] is False
    assert receipt["official_audio_quality_scoring"] is False
    assert receipt["model_calls"] == 0
    assert receipt["original_metadata_rows"] == sum(
        item["rows"] for item in source_spec("WAVCAPS")["files"]
    )
    assert receipt["public_and_unchanged_research_complete_record_readbacks"] == 12
