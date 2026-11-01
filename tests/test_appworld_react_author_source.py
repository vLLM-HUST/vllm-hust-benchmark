"""Pinned software-selection/plan contracts, not official tasks or Agent runs."""

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_original_react_software_selection_is_complete_and_bounded():
    pins = json.loads(
        (ROOT / "configs/workload-native/appworld-react-author-source.json").read_text()
    )
    assert pins["source_id"] == "APPWORLD"
    expected_revision = (
        "66ad8099e12188ece0d3fe45e661dbc01880813b"  # pragma: allowlist secret
    )
    assert pins["revision"] == expected_revision
    assert len(pins["files"]) == 10
    assert sum(item["bytes"] for item in pins["files"]) == 34698
    paths = {item["path"] for item in pins["files"]}
    assert len(paths) == 10
    assert len([p for p in paths if p.endswith(".py")]) == 8
    assert "experiments/prompts/react.txt" in paths
    assert "experiments/configs/react_llama3_test_normal.jsonnet" in paths
    assert all("data/tasks" not in path for path in paths)


def test_public_preparer_plan_does_not_acquire_or_run(tmp_path):
    target = tmp_path / "not-created"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_jsonschemabench_author_source.py"),
            "--source",
            "appworld-react",
            "--author-root",
            str(target),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["source_id"] == "APPWORLD"
    assert report["status"] == "PLAN_ONLY_NO_DOWNLOAD"
    assert report["model_calls"] == 0
    assert not target.exists()
