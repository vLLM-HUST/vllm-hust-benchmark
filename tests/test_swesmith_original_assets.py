"""SWE-smith source identity is distinct from historical Agent trajectories."""

import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
wrapper = importlib.import_module("prepare_swesmith_original_assets")


def test_original_task_profile_does_not_claim_native_execution():
    profile = json.loads(wrapper.PINS.read_text("utf-8"))
    dataset = profile["dataset"]
    assert dataset["repo_id"] == "SWE-bench/SWE-smith"
    assert len(dataset["revision"]) == 40
    files = dataset["files"]
    assert len(files) == len({row["path"] for row in files}) == 11
    assert all(row["bytes"] > 0 and len(row["sha256"]) == 64 for row in files)
    assert not profile["source_checkbox_eligible"]
    assert profile["legacy_history_is_not_native_execution"]


def test_historical_reuse_is_not_new_execution():
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/swesmith-author-history-reuse-20261102.json"
        ).read_text("utf-8")
    )
    assert receipt["complete_author_rows"] == 24
    assert receipt["new_downloads"] == receipt["new_model_calls"] == 0
    assert not receipt["research_analysis_copied"]
    assert not receipt["source_checkbox_eligible"]


def test_original_patch_direction_and_architecture_remain_explicit():
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/swesmith-original-semantics-audit-20261102.json"
        ).read_text("utf-8")
    )
    assert "--reverse" in receipt["official_gold_protocol"]
    assert "forward" in receipt["official_candidate_protocol"]
    assert not receipt["architecture_compatibility_qualified"]
    assert not receipt["source_checkbox_eligible"]


def test_fixed_first3_preserves_original_empty_issue_without_completion_claim():
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/swesmith-original-first3-preparation-20261102.json"
        ).read_text("utf-8")
    )
    assert receipt["original_issue_empty"] == [False, False, True]
    assert len(receipt["fixed_instance_ids"]) == 3
    assert receipt["acquired_release_files"] == 11
    assert receipt["pinned_release_files"] == 11
    assert receipt["research_consumer_qualified"]
    assert receipt["public_research_readbacks"] == 3
    assert receipt["public_installed_wheel_qualified"]
    assert receipt["actual_ci_wheel_package_payloads_verified"] == 82
    assert not receipt["native_scoring_executed"]
    assert not receipt["source_checkbox_eligible"]
