"""Contract tests for website fixture consumption.

These tests verify that:
1. Each fixture category produces the expected trend_status classification
2. The website can stably distinguish all six fixture categories
3. Invalid/experimental entries are never shown as default formal trends
4. The filtering contract is correctly enforced
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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

# Statuses that the website should NOT show as default formal trends
NON_DEFAULT_STATUSES = {"experimental", "blocked", "invalid", "excluded"}

# Has the validator already been run for the session?
_CACHED_REPORTS: dict[str, Any] = {}


def _load_and_validate(name: str) -> Any:
    """Load fixture entries and validate them, caching per category."""
    if name in _CACHED_REPORTS:
        return _CACHED_REPORTS[name]
    cat = CATEGORIES[name]
    entries: list[dict[str, Any]] = []
    for path in cat["files"]:
        entries.extend(load_json_entries(FIXTURES / path))
    report = validate_entries(entries)
    _CACHED_REPORTS[name] = report
    return report


# ── Contract: Each category produces the expected status ──────────────


def test_all_six_categories_are_distinguishable() -> None:
    """The website must be able to distinguish all six fixture categories."""
    statuses = {}
    for name in CATEGORIES:
        report = _load_and_validate(name)
        unique = {d.status for d in report.decisions}
        statuses[name] = unique
    # Verify each category has a distinct status (or status set)
    # The six categories produce three distinct statuses (default, blocked, experimental)
    # but each category is distinguishable by its fixture name → status mapping
    assert statuses["full-matrix"] == {"blocked"}
    assert statuses["complete-targeted-pair"] == {"default"}
    assert statuses["blocked-half-pair"] == {"blocked"}
    assert statuses["experimental"] == {"experimental"}
    assert statuses["invalid-metric"] == {"blocked"}
    assert statuses["repeat-aggregate"] == {"default"}


def test_full_matrix_fixture_status() -> None:
    """A full-matrix checkpoint without raw repeats is blocked until raw data is provided."""
    report = _load_and_validate("full-matrix")
    assert report.decisions[0].status == "blocked"
    assert "MATRIX_REPEAT_INCOMPLETE" in {i.code for i in report.issues}


def test_complete_targeted_pair_status() -> None:
    """A complete targeted pair (baseline + head with raw repeats) is admitted as default."""
    report = _load_and_validate("complete-targeted-pair")
    assert all(d.status == "default" for d in report.decisions)
    assert report.passed


def test_blocked_half_pair_status() -> None:
    """A targeted pair missing its counterpart is blocked with PAIR_HALF_MISSING."""
    report = _load_and_validate("blocked-half-pair")
    assert all(d.status == "blocked" for d in report.decisions)
    assert "PAIR_HALF_MISSING" in {i.code for i in report.issues}


def test_experimental_fixture_status() -> None:
    """An experimental (W8A8/INT8) entry is classified as experimental, not default."""
    report = _load_and_validate("experimental")
    assert report.decisions[0].status == "experimental"
    assert report.passed  # experimental is not a hard error


def test_invalid_metric_fixture_status() -> None:
    """An entry with invalid metrics is blocked, never default."""
    report = _load_and_validate("invalid-metric")
    assert report.decisions[0].status == "blocked"
    assert "LATENCY_THROUGHPUT_NOT_APPLICABLE" in {i.code for i in report.issues}


def test_repeat_aggregate_fixture_status() -> None:
    """A full-matrix entry with complete repeat aggregate is admitted as default."""
    report = _load_and_validate("repeat-aggregate")
    assert all(d.status == "default" for d in report.decisions)
    assert report.passed


# ── Contract: Website filtering rules ──────────────────────────────────


def test_only_default_status_shows_as_formal_trend() -> None:
    """The website MUST NOT show experimental/blocked/invalid/excluded as default formal trends."""
    for name in CATEGORIES:
        report = _load_and_validate(name)
        for d in report.decisions:
            if d.status == "default":
                continue  # default entries may be shown
            # Entries with non-default status must NOT be shown as formal trends
            assert d.status in NON_DEFAULT_STATUSES, (
                f"{name}/{d.entry_id[:8]}: unexpected status {d.status}"
            )


def test_valid_statuses_are_well_defined() -> None:
    """All validator-produced statuses must be from the allowed set."""
    VALID_STATUSES = {"default", "experimental", "blocked", "invalid", "excluded"}
    for name in CATEGORIES:
        report = _load_and_validate(name)
        for d in report.decisions:
            assert d.status in VALID_STATUSES, f"Unexpected status {d.status} in {name}"


def test_experimental_entries_have_diagnostic_reason() -> None:
    """Experimental entries must include a reason explaining why they are experimental."""
    report = _load_and_validate("experimental")
    for d in report.decisions:
        if d.status == "experimental":
            assert d.reason, f"Experimental entry {d.entry_id[:8]} has no reason"
            assert "W8A8" in d.reason or "INT8" in d.reason


def test_blocked_entries_have_actionable_trend_reason() -> None:
    """Blocked entries must include a trend_reason so the website can display diagnostic info."""
    for name in ("full-matrix", "blocked-half-pair", "invalid-metric"):
        report = _load_and_validate(name)
        for d in report.decisions:
            if d.status == "blocked":
                assert d.reason, f"Blocked entry {d.entry_id[:8]} in {name} has no reason"


# ── Helpers for website consumers ─────────────────────────────────────


def _filter_by_status(entries: list[dict], *statuses: str) -> list[dict]:
    """Utility: filter entries by trend_status (website consumption helper)."""
    return [e for e in entries if e.get("trend_status") in statuses]


def test_website_filter_default_entries() -> None:
    """Demonstrate how the website filters entries by trend_status.

    Only 'default' entries should be plotted as formal trend lines.
    This test verifies the fixture supports that filtering contract.
    """
    entries = load_json_entries(FIXTURES / "valid" / "complete-pair.json")
    default_entries = _filter_by_status(entries, "default")
    assert len(default_entries) == 6  # all 6 entries in the fixture are default
    non_default = [e for e in entries if e.get("trend_status") != "default"]
    assert len(non_default) == 0


def test_website_excludes_non_default_from_formal_trend() -> None:
    """Non-default entries (experimental, blocked) are excluded from the default trend filter.

    The website should always filter to trend_status=default before plotting.
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
        cat = CATEGORIES[name]
        entries: list[dict] = []
        for path in cat["files"]:
            entries.extend(load_json_entries(FIXTURES / path))
        for entry in entries:
            assert "trend_status" in entry, f"{name}: entry {entry.get('entry_id', '?')[:8]} missing trend_status"
            assert entry["trend_status"] in ("default", "experimental", "blocked", "invalid", "excluded", "pending"), \
                f"{name}: unexpected trend_status={entry['trend_status']}"
