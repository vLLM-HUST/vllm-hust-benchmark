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
