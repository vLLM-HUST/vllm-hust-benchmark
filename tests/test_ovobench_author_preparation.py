"""Offline source-plan checks; no video, model execution or published results."""

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_original_software_plan_excludes_raw_annotations_and_results(tmp_path):
    pins = json.loads(
        (ROOT / "configs/workload-native/ovobench-author-source.json").read_text()
    )
    paths = {item["path"] for item in pins["files"]}
    assert len(paths) == 45
    assert {
        "inference.py",
        "score.py",
        "utils/OVOBench.py",
        "utils/OVOBenchScore.py",
        "models/QWen2VL.py",
        "LICENSE",
    } <= paths
    assert not any(path.startswith(("data/", "results/", "images/")) for path in paths)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_jsonschemabench_author_source.py"),
            "--source",
            "ovobench",
            "--author-root",
            str(tmp_path / "original-author"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "OVOBENCH" in result.stdout
    inventory = json.loads(result.stdout)["media_archive_inventory"]
    assert len(inventory["parts"]) == 20
    assert inventory["complete_src_archive_bytes"] == 46341744640
    assert inventory["complete_chunked_archive_bytes"] == 153277153280
    assert inventory["dataset_license_conflict_unresolved"] is True
    assert inventory["dataset_readme_license"] != inventory["hub_card_license"]
    media = inventory["qualified_original_members"]
    assert [item["case_id"] for item in media if item["task"] == "ASI"] == [
        483,
        484,
        485,
    ]
    assert sum(item["bytes"] for item in media if item["task"] == "ASI") == 37469641
    assert not (tmp_path / "original-author").exists()


def test_registered_fixed_media_matches_ledger_without_implying_native_execution():
    profile = json.loads(
        (ROOT / "configs/workload-native/ovobench-author-source.json").read_text()
    )["media_archive_inventory"]
    state = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )["delivery_inventory"]["OVOBENCH"]["media_capacity_and_license_review"]
    media = state["qualified_fixed_original_media"]
    members = profile["qualified_original_members"]
    assert (
        len(members)
        == len({item["path"] for item in members})
        == media["registered_native_clips"]
    )
    assert (
        sum(item["bytes"] for item in members)
        == media["current_registered_original_bytes_total"]
    )
    assert (
        sum(item["video_frames"] for item in members)
        == media["current_registered_video_frames_total"]
    )
    if media["all_fixed_media_registered"]:
        assert len(members) == media["required_fixed_native_clips"] == 103
        assert len({item["case_id"] for item in members}) == 36
        assert len({item["task"] for item in members}) == 12
        assert media["public_require_all_fixed103_cli_passed"] is True
    assert media["source_checkbox_eligible"] is False
    assert (
        state["fixed_original_media_acquisition_queue"][
            "native_model_execution_started"
        ]
        is False
    )
