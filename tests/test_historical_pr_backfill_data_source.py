"""Defensive contract: every historical-PR-backfill submission MUST carry
``metadata.data_source == "real-online-historical-pr-backfill"``.

The website-side aggregator
(:file:`vllm-hust-website/scripts/aggregate_results.py`) exempts these
entries from the canonical-plugin-commit rejection rule so that cross-PR
comparison runs (PR#66 / PR#70 / PR#77 each testing a different plugin
commit against the same vllm-hust engine commit) can coexist on the public
snapshot and feed the compare cards.

If a historical-pr-backfill submission is ever written without the correct
``data_source`` marker, the website-side rejector will silently drop it
from the public snapshot, and the cross-PR compare cards will lose data
— exactly the regression that prompted this test.

This test scans the real ``submissions/`` directory and fails if any
directory whose name matches the historical-PR-backfill naming pattern
contains a ``run_leaderboard.json`` whose ``metadata.data_source`` is not
``"real-online-historical-pr-backfill"``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS_DIR = REPO_ROOT / "submissions"

# historical-PR-backfill directories follow the naming convention
# ``historical-pr-<pr-ref>-<workload>-<engine_commit>-<plugin_commit>``.
# We match on the ``historical-pr-`` prefix to stay robust against ref-name
# variations (pr-77, ascend-pr66, etc.).
HISTORICAL_PR_DIR_PATTERN = re.compile(r"^historical-pr-", re.IGNORECASE)
EXPECTED_DATA_SOURCE = "real-online-historical-pr-backfill"


def _historical_pr_submission_dirs() -> list[Path]:
    if not SUBMISSIONS_DIR.is_dir():
        return []
    return [
        child
        for child in sorted(SUBMISSIONS_DIR.iterdir())
        if child.is_dir() and HISTORICAL_PR_DIR_PATTERN.match(child.name)
    ]


def _load_data_source(submission_dir: Path) -> str | None:
    artifact = submission_dir / "run_leaderboard.json"
    if not artifact.is_file():
        return None
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("data_source")
    return str(value) if value is not None else None


@pytest.mark.skipif(
    not SUBMISSIONS_DIR.is_dir(),
    reason=f"submissions/ not present at {SUBMISSIONS_DIR}",
)
def test_every_historical_pr_backfill_submission_carries_expected_data_source() -> None:
    """Every ``historical-pr-*`` submission directory MUST set
    ``metadata.data_source`` to
    ``"real-online-historical-pr-backfill"`` so the website-side aggregator
    exempts it from the canonical-plugin-commit rejection rule.
    """
    bad: list[tuple[str, str | None]] = []
    for sub_dir in _historical_pr_submission_dirs():
        data_source = _load_data_source(sub_dir)
        if data_source != EXPECTED_DATA_SOURCE:
            bad.append((sub_dir.name, data_source))

    if bad:
        details = "\n".join(
            f"  - {name}: data_source={value!r} (expected {EXPECTED_DATA_SOURCE!r})"
            for name, value in bad
        )
        pytest.fail(
            "Found historical-PR-backfill submissions missing the required "
            f"data_source marker ({EXPECTED_DATA_SOURCE!r}). These entries "
            "will be silently dropped by the website-side "
            "plugin_commit_mismatch_rejection_reason filter, breaking "
            "cross-PR compare cards. Fix metadata.data_source on:\n" + details
        )
