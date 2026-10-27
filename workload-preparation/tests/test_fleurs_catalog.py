"""Original metadata/reader fixtures only; no real audio or model validation."""

from copy import deepcopy
import json
import re

import pytest

from vllm_hust_workload_data.adapters import adapt_row
from vllm_hust_workload_data.catalog import download_plan
from vllm_hust_workload_data.registry import adapter_spec, read_catalog, source_spec


def test_pinned_original_three_splits_all_real_locales_no_duplicate_all_view():
    source = source_spec("FLEURS")
    files = source["files"]
    assert len(files) == 308
    assert len({item["locale"] for item in files}) == 102
    assert "all" not in {item["locale"] for item in files}
    assert sum(item["bytes"] for item in files) == 315248828965
    tests = [item for item in files if item["split"] == "test"]
    assert len(tests) == 102 and sum(item["rows"] for item in tests) == 77810
    assert sum(item["bytes"] for item in tests) == 64464134486
    assert all(re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) for item in files)
    assert all(source["revision"] in item["url"] for item in files)
    plan = download_plan("FLEURS")
    assert len(plan["files"]) == 308
    assert plan["files"][0]["locale"] and plan["files"][0]["split"]


def test_audio_and_every_original_label_remain_in_correct_sides():
    audio = {"bytes": b"inert audio fixture, not real WAV", "path": "original-name.wav"}
    row = {
        "audio": audio,
        "id": 7,
        "transcription": "normalized original",
        "raw_transcription": "Raw original.",
        "num_samples": 4,
        "path": "original-name.wav",
        "gender": 0,
        "lang_id": 13,
        "language": "Mandarin Chinese",
        "lang_group_id": 6,
    }
    record = adapt_row(
        "FLEURS",
        row,
        source={"source_id": "FLEURS", "file_sha256": "fixture"},
        spec=adapter_spec("FLEURS"),
        index=0,
    )
    assert record.task["input"] == {"audio": audio}
    assert record.outcome == {
        "transcription": row["transcription"],
        "raw_transcription": row["raw_transcription"],
    }
    assert record.replay["annotations"] == {
        key: value
        for key, value in row.items()
        if key not in {"audio", "transcription", "raw_transcription"}
    }
    assert record.replay["unmapped_field_names"] == []


@pytest.mark.parametrize(
    "mismatch", ["source_id", "repo_id", "revision", "duplicate_path"]
)
def test_external_resource_mismatch_rejected_without_changing_source(
    tmp_path, mismatch
):
    sources = deepcopy(read_catalog("source_manifests_v1.json"))
    manifest = deepcopy(read_catalog("fleurs_source_files_v1.json"))
    if mismatch == "duplicate_path":
        manifest["files"].append(deepcopy(manifest["files"][0]))
    else:
        manifest[mismatch] = "wrong publisher identity"
    target = tmp_path / "data/catalogs"
    target.mkdir(parents=True)
    (target / "source_manifests_v1.json").write_text(
        json.dumps(sources), encoding="utf-8"
    )
    (target / "fleurs_source_files_v1.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="resource identity"):
        source_spec("FLEURS", tmp_path)
