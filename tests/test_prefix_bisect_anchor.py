from __future__ import annotations

import json
from pathlib import Path

from vllm_hust_benchmark import integration
from vllm_hust_benchmark.workload_config_contract import (
    requires_workload_config_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BISECT_ANCHOR_DIR = (
    REPO_ROOT
    / "submissions"
    / "historical-pr-prefix-bisect-20260706-runtime-compat-prefix-repetition-online-103d3aa344-bae528a400"
)


def _load_bisect_anchor_entry() -> dict:
    return json.loads(
        (BISECT_ANCHOR_DIR / "run_leaderboard.json").read_text(encoding="utf-8")
    )


def test_prefix_bisect_anchor_passes_contract() -> None:
    entry = _load_bisect_anchor_entry()

    # The bisect anchor was submitted on 2026-07-20, before the
    # workload_config_contract activation date of 2026-07-24, so it is
    # grandfathered and not subject to strict contract validation.
    assert not requires_workload_config_contract(entry)
    assert integration._validate_entry_workload_contract(
        entry, source="prefix-bisect-anchor", require_official=False
    )


def test_prefix_bisect_anchor_not_rejected_as_pr_preview() -> None:
    entry = _load_bisect_anchor_entry()

    # github_pr_number is null and github_event_name is null, so the
    # artifact must not be falsely classified as a PR-preview submission.
    assert not integration._artifact_has_pr_preview_metadata(entry)
