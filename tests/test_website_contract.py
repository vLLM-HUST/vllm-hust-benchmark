"""Contract tests for website fixture consumption.

These tests verify that:
1. Each fixture category produces the expected trend_status classification
2. The website can stably distinguish all six fixture categories
3. Invalid/experimental entries are never shown as default formal trends
4. The filtering contract is correctly enforced

Test paths:
- Status classification tests go through the full validator pipeline
  (load_json_entries → validate_entries) to verify the adjudicated status.
- Filtering tests use the raw fixture load path (load_json_entries only)
  to simulate the website consumer perspective, separate from the
  validator pipeline. A fixture's trend_status is either set statically
  (for pre-classified entries) or set by the validator; the website
  only reads trend_status, not how it was produced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vllm_hust_benchmark.trend_validator import load_json_entries, validate_entries

FIXTURES = Path(__file__).parent / "fixtures" / "trend_coverage"

# ── Category definitions ──────────────────────────────────────────────
# Each category lists the fixture file(s) and the expected trend_status
# that the validator must assign for the website to consume.

CATEGORIES: dict[str, dict[str, Any]] = {
    "full-matrix": {
        "files": ["valid/full-matrix.json"],
        "expected_status": "blocked",
        "desc": "Single full-matrix checkpoint without raw repeats → blocked (needs aggregate entries)",
    },
    "complete-targeted-pair": {
        "files": ["valid/complete-pair.json"],
        "expected_status": "default",
        "desc": "Complete targeted pair with both baseline/head and raw repeats → default",
    },
    "blocked-half-pair": {
        "files": ["valid/blocked-half-pair.json"],
        "expected_status": "blocked",
        "desc": "Targeted pair with only one side present → blocked (PAIR_HALF_MISSING)",
    },
    "experimental": {
        "files": ["valid/experimental.json"],
        "expected_status": "experimental",
        "desc": "W8A8/INT8 entry outside formal support → experimental",
    },
    "invalid-metric": {
        "files": ["valid/invalid.json"],
        "expected_status": "blocked",
        "desc": "Entry with invalid metrics → blocked (latency throughput not applicable)",
    },
    "repeat-aggregate": {
        "files": ["valid/repeat-aggregate.json"],
        "expected_status": "default",
        "desc": "Full-matrix entry with canonical aggregate from 3 repeat runs → default",
    },
}

VALID_STATUSES = {"default", "experimental", "blocked", "invalid", "excluded"}

# Statuses that the website should NOT show as default formal trends
NON_DEFAULT_STATUSES = VALID_STATUSES - {"default"}


def _load_category_entries(name: str) -> list[dict[str, Any]]:
    """Load raw fixture entries for a category (no validator)."""
    cat = CATEGORIES[name]
    entries: list[dict[str, Any]] = []
    for path in cat["files"]:
        entries.extend(load_json_entries(FIXTURES / path))
    return entries


@pytest.fixture(scope="module")
def reports():
    """Cache validated reports per category, shared across all tests in this module."""
    cache: dict[str, Any] = {}
    for name in CATEGORIES:
        entries = _load_category_entries(name)
        cache[name] = validate_entries(entries)
    return cache


# ── Contract: Each category produces the expected status ──────────────


def test_all_six_categories_produce_expected_status(reports) -> None:
    """Each fixture category must produce the expected trend_status after validation."""
    statuses = {}
    for name in CATEGORIES:
        unique = {d.status for d in reports[name].decisions}
        statuses[name] = unique
    assert statuses["full-matrix"] == {"blocked"}
    assert statuses["complete-targeted-pair"] == {"default"}
    assert statuses["blocked-half-pair"] == {"blocked"}
    assert statuses["experimental"] == {"experimental"}
    assert statuses["invalid-metric"] == {"blocked"}
    assert statuses["repeat-aggregate"] == {"default"}


def test_full_matrix_fixture_status(reports) -> None:
    """A full-matrix checkpoint without raw repeats is blocked until raw data is provided."""
    assert reports["full-matrix"].decisions[0].status == "blocked"
    assert "MATRIX_REPEAT_INCOMPLETE" in {i.code for i in reports["full-matrix"].issues}


def test_complete_targeted_pair_status(reports) -> None:
    """A complete targeted pair (baseline + head with raw repeats) is admitted as default."""
    assert all(
        d.status == "default" for d in reports["complete-targeted-pair"].decisions
    )
    assert reports["complete-targeted-pair"].passed


def test_blocked_half_pair_status(reports) -> None:
    """A targeted pair missing its counterpart is blocked with PAIR_HALF_MISSING."""
    assert all(d.status == "blocked" for d in reports["blocked-half-pair"].decisions)
    assert "PAIR_HALF_MISSING" in {i.code for i in reports["blocked-half-pair"].issues}


def test_experimental_fixture_status(reports) -> None:
    """An experimental (W8A8/INT8) entry is classified as experimental, not default."""
    assert reports["experimental"].decisions[0].status == "experimental"
    assert reports["experimental"].passed  # experimental is not a hard error


def test_invalid_metric_fixture_status(reports) -> None:
    """An entry with invalid metrics is blocked, never default."""
    assert reports["invalid-metric"].decisions[0].status == "blocked"
    assert "LATENCY_THROUGHPUT_NOT_APPLICABLE" in {
        i.code for i in reports["invalid-metric"].issues
    }


def test_repeat_aggregate_fixture_status(reports) -> None:
    """A full-matrix entry with complete repeat aggregate is admitted as default."""
    assert all(d.status == "default" for d in reports["repeat-aggregate"].decisions)
    assert reports["repeat-aggregate"].passed


# ── Contract: Website filtering rules ──────────────────────────────────


def test_default_filter_contract_across_all_categories(reports) -> None:
    """The default-only filter excludes all non-default entries across every category.

    This is the core contract: the one-line filter rule
    ``entry.get('trend_status') == 'default'`` is the ONLY gate for formal trends.
    The website must never show experimental/blocked/invalid/excluded as formal trends.

    Uses the validator-adjudicated reports (the same pipeline that produces the
    production data the website consumes), not raw fixture files.
    """
    for name in CATEGORIES:
        report = reports[name]
        default_count = sum(1 for d in report.decisions if d.status == "default")
        cat = CATEGORIES[name]
        if cat["expected_status"] == "default":
            assert default_count == len(report.decisions), (
                f"{name}: expected all {len(report.decisions)} entries to be "
                f"default, got {default_count}"
            )
        else:
            assert default_count == 0, (
                f"{name}: expected 0 default entries (status={cat['expected_status']}), "
                f"got {default_count}. Non-default entries must be excluded from the "
                f"formal trend filter."
            )


def test_non_default_statuses_are_valid(reports) -> None:
    """All non-default statuses belong to the known set of allowed statuses."""
    for name in CATEGORIES:
        for d in reports[name].decisions:
            if d.status != "default":
                assert d.status in NON_DEFAULT_STATUSES, (
                    f"{name}/{d.entry_id[:8]}: unexpected non-default status {d.status}"
                )


def test_experimental_entries_have_diagnostic_reason(reports) -> None:
    """Experimental entries must include a reason explaining why they are experimental."""
    for d in reports["experimental"].decisions:
        if d.status == "experimental":
            assert d.reason, f"Experimental entry {d.entry_id[:8]} has no reason"
            assert "W8A8" in d.reason or "INT8" in d.reason


def test_blocked_entries_have_actionable_trend_reason(reports) -> None:
    """Blocked entries must include a trend_reason so the website can display diagnostic info."""
    for name in ("full-matrix", "blocked-half-pair", "invalid-metric"):
        for d in reports[name].decisions:
            if d.status == "blocked":
                assert d.reason, (
                    f"Blocked entry {d.entry_id[:8]} in {name} has no reason"
                )


# ── Helpers for website consumers ─────────────────────────────────────


def _filter_by_status(entries: list[dict], *statuses: str) -> list[dict]:
    """Utility: filter entries by trend_status (website consumption helper).

    Used in the filtering tests below to demonstrate the exact filter
    the website should apply when building its trend chart.
    """
    return [e for e in entries if e.get("trend_status") in statuses]


def test_website_filter_default_entries() -> None:
    """Demonstrate how the website filters entries by trend_status.

    Only 'default' entries should be plotted as formal trend lines.
    This test exercises the raw fixture load path (website consumer perspective)
    rather than the validator pipeline, to verify the fixture's on-disk contract.
    """
    entries = load_json_entries(FIXTURES / "valid" / "complete-pair.json")
    default_entries = _filter_by_status(entries, "default")
    assert len(default_entries) == 6  # all 6 entries in the fixture are default
    non_default = [e for e in entries if e.get("trend_status") != "default"]
    assert len(non_default) == 0


def test_website_excludes_non_default_from_formal_trend() -> None:
    """Non-default entries (experimental, blocked) are excluded from the default trend filter.

    The website should always filter to trend_status=default before plotting.
    This test exercises the raw fixture load path (website consumer perspective).
    """
    entries = load_json_entries(FIXTURES / "valid" / "blocked-half-pair.json")
    default_entries = _filter_by_status(entries, "default")
    assert len(default_entries) == 0  # all blocked, none for formal trend


def test_website_can_build_filterable_dataframe() -> None:
    """All fixture entries have the required fields for website filtering.

    This is a structural contract: the website can safely filter by trend_status
    on any entry without checking for field existence first.
    """
    for name in CATEGORIES:
        entries = _load_category_entries(name)
        for entry in entries:
            assert "trend_status" in entry, (
                f"{name}: entry {entry.get('entry_id', '?')[:8]} missing trend_status"
            )
            assert entry["trend_status"] in VALID_STATUSES, (
                f"{name}: unexpected trend_status={entry['trend_status']}"
            )
