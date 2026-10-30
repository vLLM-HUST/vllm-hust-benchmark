"""Pinned software preparation only; original runtime defect stays disclosed."""

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "configs/workload-native/marble-author-source.json"


def test_original_source_selection_preserves_lock_and_known_defect_identity():
    pins = json.loads(PINS.read_text("utf-8"))
    files = {item["path"]: item for item in pins["files"]}
    assert len(files) == len(pins["files"]) == 100
    assert sum(item["bytes"] for item in files.values()) == 1420988
    assert sum(name.endswith(".py") for name in files) == 94
    assert {
        "LICENSE",
        "poetry.lock",
        "pyproject.toml",
        "marble/main.py",
        "multiagentbench/runjsonl2yaml.sh",
    } <= files.keys()
    assert (
        files["marble/evaluator/evaluator.py"]["sha256"]
        == "cc9234a9003792538acc5e01579f484d16446da696319cb0c34a7f3466990ad8"  # pragma: allowlist secret - verified public publisher software digest
    )
    assert not any(name.endswith("_main.jsonl") for name in files)


def test_default_source_plan_does_not_download_or_create_environment(tmp_path):
    target = tmp_path / "absent"
    report = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-I",
                str(ROOT / "scripts/prepare_jsonschemabench_author_source.py"),
                "--source",
                "marble",
                "--author-root",
                str(target),
            ],
            text=True,
        )
    )
    assert report["source_id"] == "MULTIAGENTBENCH"
    assert report["status"] == "PLAN_ONLY_NO_DOWNLOAD"
    assert report["files"] == 100
    assert report["model_calls"] == 0 and report["source_checkbox_eligible"] is False
    assert not target.exists()


def test_supplemental_client_dependencies_are_disclosed_not_provider_defaults():
    pins = [
        line
        for line in (ROOT / "configs/workload-native/marble-client-supplements.txt")
        .read_text("utf-8")
        .splitlines()
        if line and not line.startswith("#")
    ]
    assert pins == ["ruamel.yaml==0.18.6", "ruamel.yaml.clib==0.2.12"]
    assert "ruamel" not in (ROOT / "workload-preparation/pyproject.toml").read_text(
        "utf-8"
    )
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )
    row = ledger["delivery_inventory"]["MULTIAGENTBENCH"]
    preflight = row["original_software_preparation"]["compat_client_entry_preflight"]
    assert preflight["exit_code"] == 0 and preflight["total_client_distributions"] == 95
    assert preflight["original93_locked_distribution_versions_unchanged"]
    assert preflight["model_calls"] == 0
    assert (
        preflight["native_execution"] is False
        and preflight["official_scoring"] is False
    )
    assert preflight["source_checkbox_eligible"] is False and row["checked"] is False
