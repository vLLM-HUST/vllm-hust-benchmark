"""Tests for the plugin commit consistency guard in backfill_single_gpu.py.

Covers four cases enumerated in docs/HISTORICAL_PR_BACKFILL.md →

1. snapshot has a canonical plugin commit and the requested plugin commit
   agrees → pass.
2. snapshot has a canonical plugin commit and the requested plugin commit
   disagrees → raise ``PluginCommitMismatch``.
3. snapshot has no entry for this hust_commit (snapshot miss) → pass; the
   first run against a given vllm-hust commit is unconstrained because
   there is nothing to be inconsistent with.
4. ``allow_override=True`` → pass and the override is recorded in
   ``state.json`` under ``audit.plugin_override``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "backfill_single_gpu.py"
)
REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "backfill_single_gpu", SCRIPT_PATH
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_passes_when_requested_plugin_matches_snapshot_canonical(
    tmp_path: Path,
) -> None:
    """Case 1: requested commit == snapshot canonical → pass."""
    mod = load_module()
    hust = "a46abb7ae68acc13a4fc5870db98619b3f97c6e0"
    plugin = "f430530ada2c0c2ec2f925606494bc95a474d9c8"
    with patch.object(mod, "_lookup_ascend_commit_from_snapshot", return_value=plugin):
        # Should not raise.
        mod.assert_plugin_commit_consistent(hust, plugin)


def test_raises_when_requested_plugin_disagrees_with_snapshot_canonical(
    tmp_path: Path,
) -> None:
    """Case 2: snapshot has a canonical, requested is different → raise."""
    mod = load_module()
    hust = "a46abb7ae68acc13a4fc5870db98619b3f97c6e0"
    canonical = "f430530ada2c0c2ec2f925606494bc95a474d9c8"
    divergent = "03a12f9bddd944952bd029c6b62e23d68fa3a28e"
    with patch.object(mod, "_lookup_ascend_commit_from_snapshot", return_value=canonical):
        try:
            mod.assert_plugin_commit_consistent(hust, divergent)
        except mod.PluginCommitMismatch as exc:
            assert exc.hust_commit == hust
            assert exc.canonical == canonical
            assert exc.requested == divergent
            assert "force-mismatched-plugin-commit" in str(exc)
            return
    raise AssertionError("PluginCommitMismatch was not raised")


def test_passes_on_snapshot_miss_when_no_canonical_known() -> None:
    """Case 3: snapshot has no entry for this hust_commit → pass."""
    mod = load_module()
    hust = "0000000000000000000000000000000000000000"  # not in any snapshot
    plugin = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    with patch.object(mod, "_lookup_ascend_commit_from_snapshot", return_value=None):
        # Should not raise.
        mod.assert_plugin_commit_consistent(hust, plugin)


def test_allow_override_permits_mismatch_and_records_audit_entry(
    tmp_path: Path,
) -> None:
    """Case 4: ``allow_override=True`` → pass; override recorded in state.json."""
    mod = load_module()
    hust = "a46abb7ae68acc13a4fc5870db98619b3f97c6e0"
    canonical = "f430530ada2c0c2ec2f925606494bc95a474d9c8"
    override_value = "03a12f9bddd944952bd029c6b62e23d68fa3a28e"

    # Redirect STATE_DIR to tmp_path so the test does not touch the real
    # .benchmarks/ directory.
    state_dir = tmp_path / ".benchmarks" / "backfill-single-gpu"
    state_dir.mkdir(parents=True)
    state_file = state_dir / "state.json"
    state = {"hust_head": "abc", "ascend_head": "xyz", "cells": {}}
    state_file.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    with patch.object(mod, "STATE_DIR", state_dir), \
         patch.object(mod, "STATE_FILE", state_file), \
         patch.object(
             mod, "_lookup_ascend_commit_from_snapshot", return_value=canonical
         ):
        # Pass via override.
        mod.assert_plugin_commit_consistent(
            hust, override_value, allow_override=True
        )
        # Record the audit entry as cmd_run would.
        mod.record_plugin_override(
            state, hust, canonical, override_value
        )

    persisted = json.loads(state_file.read_text(encoding="utf-8"))
    overrides = persisted.get("audit", {}).get("plugin_override", [])
    assert len(overrides) == 1
    entry = overrides[0]
    assert entry["hust_commit"] == hust
    assert entry["canonical_plugin_commit"] == canonical
    assert entry["override_plugin_commit"] == override_value
    assert "timestamp" in entry


def test_plugin_commit_mismatch_message_includes_short_shas() -> None:
    """The exception message exposes 9-char prefixes for easy reading."""
    mod = load_module()
    hust = "a46abb7ae68acc13a4fc5870db98619b3f97c6e0"
    canonical = "f430530ada2c0c2ec2f925606494bc95a474d9c8"
    divergent = "03a12f9bddd944952bd029c6b62e23d68fa3a28e"
    exc = mod.PluginCommitMismatch(hust, canonical, divergent)
    msg = str(exc)
    assert "a46abb7ae" in msg
    assert "f430530ad" in msg
    assert "03a12f9bd" in msg
    assert "force-mismatched-plugin-commit" in msg