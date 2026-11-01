"""Offline source preparation checks do not certify native Agent execution."""

import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/workload-native/mlagentbench-author-source.json"


def test_pinned_original_source_inventory_and_scope():
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    assert profile["repository"] == "snap-stanford/MLAgentBench"
    files = profile["files"]
    assert len(files) == 138 and sum(item["bytes"] for item in files) == 670271
    assert len({item["path"] for item in files}) == len(files)
    assert sum(item["path"].endswith(".py") for item in files) == 79
    for item in files:
        path = PurePosixPath(item["path"])
        assert not path.is_absolute() and ".." not in path.parts
        assert re.fullmatch("[0-9a-f]{40}", item["git_blob"])
        assert re.fullmatch("[0-9a-f]{64}", item["sha256"])
        assert "payload" not in item
        assert (
            "Auto-GPT/" not in item["path"] and item["path"] != "MLAgentBench/plot.py"
        )
    paths = {item["path"] for item in files}
    assert {
        "LICENSE",
        "MLAgentBench/runner.py",
        "MLAgentBench/eval.py",
        "MLAgentBench/prepare_task.py",
    }.issubset(paths)


def test_public_plan_neither_creates_directory_nor_executes_source(tmp_path):
    target = tmp_path / "not-created"
    raw = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_jsonschemabench_author_source.py"),
            "--source",
            "mlagentbench",
            "--author-root",
            str(target),
            "--action",
            "plan",
        ]
    )
    report = json.loads(raw)
    assert report["files"] == 138 and report["bytes"] == 670271
    assert report["model_calls"] == 0 and report["source_checkbox_eligible"] is False
    assert not target.exists()


def test_source_preparation_does_not_promote_delivery():
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text(
            encoding="utf-8"
        )
    )
    source = ledger["delivery_inventory"]["MLAGENTBENCH"]
    assert not source["checked"] and source["delivery_receipt"] is None
