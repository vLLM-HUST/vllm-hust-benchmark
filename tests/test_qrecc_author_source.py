"""Original author code identity and offline source plan, not retrieval/QA."""

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_author_source_includes_both_scorers_and_exact_dependency_declaration():
    pins = json.loads(
        (ROOT / "configs/workload-native/qrecc-author-source.json").read_text("utf-8")
    )
    files = {x["path"]: x for x in pins["files"]}
    assert len(files) == len(pins["files"]) == 11
    assert sum(x["bytes"] for x in files.values()) == 53913
    assert sum(name.endswith(".py") for name in files) == 6
    assert {
        "requirements.txt",
        "utils/evaluate_qa.py",
        "utils/evaluate_retrieval.py",
        "utils/span_heuristic.py",
        "collection/README.md",
        "LICENSE",
    } <= files.keys()
    assert not any(
        name.startswith("dataset/") or name.endswith(".zip") for name in files
    )
    assert pins["declared_requirements"]["pyserini"] == "0.10.0.1"
    assert (
        files["requirements.txt"]["git_blob"]
        == "30045b7721768448f15a6ddac0ed4193c9134575"  # pragma: allowlist secret - public original requirements identity
    )


def test_offline_plan_has_no_download_or_environment_side_effect(tmp_path):
    root = tmp_path / "absent"
    report = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-I",
                str(ROOT / "scripts/prepare_jsonschemabench_author_source.py"),
                "--source",
                "qrecc",
                "--author-root",
                str(root),
            ],
            text=True,
        )
    )
    assert report["source_id"] == "QRECC" and report["files"] == 11
    assert report["bytes"] == 53913 and report["status"] == "PLAN_ONLY_NO_DOWNLOAD"
    assert report["model_calls"] == 0 and not report["source_checkbox_eligible"]
    assert not root.exists()
