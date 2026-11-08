"""Software preparation is independent from gated live-instance execution."""

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_plan_has_no_files_network_or_instance_side_effects(tmp_path):
    root = tmp_path / "not-created"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_workarena_author_source.py"),
            "--author-root",
            str(root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    assert report["status"] == "PLAN_ONLY"
    assert report["files"] == 174 and report["bytes"] == 148527353
    assert report["model_calls"] == 0 and not report["source_checkbox_eligible"]
    assert not root.exists()


def test_full_public_repository_pins_do_not_include_instance_credentials():
    pins = json.loads(
        (ROOT / "configs/workload-native/workarena-author-source.json").read_text(
            "utf-8"
        )
    )
    names = {row["path"] for row in pins["files"]}
    assert len(names) == 174
    assert {
        "README.md",
        "LICENSE",
        "requirements.txt",
        "pyproject.toml",
        "src/browsergym/workarena/instance.py",
        "src/browsergym/workarena/install.py",
    } <= names
    assert "instances_v2.json" not in {Path(name).name for name in names}
    assert all(
        len(row["sha256"]) == 64 and len(row["git_blob"]) == 40 for row in pins["files"]
    )
    assert not pins["native_execution_complete"]
