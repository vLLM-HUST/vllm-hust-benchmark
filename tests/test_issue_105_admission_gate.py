"""Tests for issue #105 admission gate fail-closed behavior.

Covers the two review comments on PR #139:
1. ``scripts/cleanup_leaderboard_for_issue_105.py`` must fail closed when
   ``gpu_mem``/``max_len`` is missing and must enforce the admission gate
   (verified, peak_mem, error_rate, resolved_spec_hash, runtime provenance)
   rather than admitting entries purely on profile shape.
2. ``scripts/generate_paired_evidence_for_issue_105.py`` must preserve and
   validate verified / target_id / resolved_spec_hash / runtime provenance /
   peak_mem / error_rate; any missing field sets paired_evidence_valid=false.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]

CLEANUP_PATH = REPO_ROOT / "scripts" / "cleanup_leaderboard_for_issue_105.py"
PAIRED_PATH = REPO_ROOT / "scripts" / "generate_paired_evidence_for_issue_105.py"


@pytest.fixture(scope="module")
def cleanup_mod():
    spec = importlib.util.spec_from_file_location("cleanup_issue_105", CLEANUP_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def paired_mod():
    spec = importlib.util.spec_from_file_location(
        "paired_evidence_issue_105", PAIRED_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_VALID_ENGINE_COMMIT = (
    "e4ce33646f2ef1781289e6dc651fad0d00177c55"  # pragma: allowlist secret
)
_VALID_PLUGIN_COMMIT = (
    "0f38988f47b55e2e896551bc6125fda27fae5392"  # pragma: allowlist secret
)
_VALID_RESOLVED_HASH = "f8cc8fc26b4b9bb06d50f079174894a95d2bc0f49799374a652e6e04b75c8feb"  # pragma: allowlist secret


def _valid_active_entry() -> dict:
    """A core-text-14b entry that passes both profile match and admission gate."""
    return {
        "entry_id": "11111111-2222-3333-4444-555555555555",
        "engine": "vllm-ascend",
        "engine_version": "v0.17.2.post1-3628-ge4ce33646f",
        "config_type": "single_gpu",
        "hardware": {"vendor": "ascend", "chip_model": "910B2", "chip_count": 1},
        "model": {
            "name": "Qwen/Qwen2.5-14B-Instruct",
            "precision": "FP16",
        },
        "workload": {
            "name": "random-online",
            "input_length": 1024,
            "output_length": 1024,
        },
        "metrics": {
            "ttft_ms": 235.71,
            "tbt_ms": 39.99,
            "throughput_tps": 244.89,
            "peak_mem_mb": 30000,
            "error_rate": 0.0,
        },
        "constraints": {},
        "versions": {},
        "environment": {"gpu_memory_utilization": 0.6, "max_model_len": 32768},
        "metadata": {
            "verified": True,
            "git_commit": _VALID_ENGINE_COMMIT,
            "runtime_provenance": {
                "engine": {"commit": _VALID_ENGINE_COMMIT},
                "plugin": {"commit": _VALID_PLUGIN_COMMIT},
            },
        },
        "same_spec": {
            "spec_id": "official-ascend-jan-2026-v0.18.0-random-online-qwen25-14b-910b2",
            "resolved_spec_hash": _VALID_RESOLVED_HASH,
        },
    }


# ===========================================================================
# cleanup_leaderboard_for_issue_105.py tests (comment 1)
# ===========================================================================


class TestCleanupMatchActiveProfileFailClosed:
    def test_missing_gpu_mem_fails_closed(self, cleanup_mod):
        entry = _valid_active_entry()
        # Remove gpu_memory_utilization entirely
        entry["environment"] = {"max_model_len": 32768}
        assert cleanup_mod._match_active_profile(entry) is None

    def test_missing_max_model_len_fails_closed(self, cleanup_mod):
        entry = _valid_active_entry()
        entry["environment"] = {"gpu_memory_utilization": 0.6}
        assert cleanup_mod._match_active_profile(entry) is None

    def test_present_matching_config_matches(self, cleanup_mod):
        entry = _valid_active_entry()
        prof = cleanup_mod._match_active_profile(entry)
        assert prof is not None
        assert prof["profile"] == "core-text-14b"


class TestCleanupAdmissionGate:
    def test_passes_admission_gate_for_valid_entry(self, cleanup_mod):
        entry = _valid_active_entry()
        ok, failures = cleanup_mod._passes_admission_gate(entry)
        assert ok, f"unexpected failures: {failures}"
        assert failures == []

    def test_verified_none_rejected(self, cleanup_mod):
        entry = _valid_active_entry()
        entry["metadata"]["verified"] = None
        ok, failures = cleanup_mod._passes_admission_gate(entry)
        assert not ok
        assert "metadata.verified" in failures

    def test_verified_false_rejected(self, cleanup_mod):
        entry = _valid_active_entry()
        entry["metadata"]["verified"] = False
        ok, failures = cleanup_mod._passes_admission_gate(entry)
        assert not ok
        assert "metadata.verified" in failures

    def test_peak_mem_zero_rejected(self, cleanup_mod):
        entry = _valid_active_entry()
        entry["metrics"]["peak_mem_mb"] = 0
        ok, failures = cleanup_mod._passes_admission_gate(entry)
        assert not ok
        assert "metrics.peak_mem_mb" in failures

    def test_error_rate_one_rejected(self, cleanup_mod):
        entry = _valid_active_entry()
        entry["metrics"]["error_rate"] = 1.0
        ok, failures = cleanup_mod._passes_admission_gate(entry)
        assert not ok
        assert "metrics.error_rate" in failures

    def test_missing_resolved_spec_hash_rejected(self, cleanup_mod):
        entry = _valid_active_entry()
        entry["same_spec"] = {
            "spec_id": "official-ascend-jan-2026-v0.18.0-random-online-qwen25-14b-910b2"
        }
        ok, failures = cleanup_mod._passes_admission_gate(entry)
        assert not ok
        assert "same_spec.resolved_spec_hash" in failures

    def test_short_engine_commit_rejected(self, cleanup_mod):
        entry = _valid_active_entry()
        entry["metadata"]["runtime_provenance"]["engine"]["commit"] = (
            "e4ce33646f"  # 11 chars
        )
        ok, failures = cleanup_mod._passes_admission_gate(entry)
        assert not ok
        assert "metadata.runtime_provenance.engine.commit" in failures

    def test_missing_plugin_commit_rejected(self, cleanup_mod):
        entry = _valid_active_entry()
        entry["metadata"]["runtime_provenance"]["plugin"] = {}
        ok, failures = cleanup_mod._passes_admission_gate(entry)
        assert not ok
        assert "metadata.runtime_provenance.plugin.commit" in failures


class TestCleanupClassifyEntry:
    def test_valid_entry_kept(self, cleanup_mod):
        entry = _valid_active_entry()
        prof, disposition, reason, _drift, _missing = cleanup_mod._classify_entry(entry)
        assert prof is not None
        assert disposition == "keep"
        assert reason is None

    def test_verified_false_quarantined_not_kept(self, cleanup_mod):
        """The 132s TTFT entry from the review (verified=False, peak_mem=0)."""
        entry = _valid_active_entry()
        entry["metadata"]["verified"] = False
        entry["metrics"]["peak_mem_mb"] = 0
        entry["metrics"]["ttft_ms"] = 132137.44  # 132s anomaly
        _prof, disposition, reason, _drift, _missing = cleanup_mod._classify_entry(
            entry
        )
        assert disposition == "quarantine"
        assert "admission_gate_failure" in reason

    def test_verified_none_quarantined(self, cleanup_mod):
        entry = _valid_active_entry()
        entry["metadata"]["verified"] = None
        _prof, disposition, reason, _drift, missing = cleanup_mod._classify_entry(entry)
        assert disposition == "quarantine"
        assert "admission_gate_failure" in reason
        assert "metadata.verified" in missing

    def test_missing_gpu_mem_quarantined_not_kept(self, cleanup_mod):
        entry = _valid_active_entry()
        entry["environment"] = {"max_model_len": 32768}  # no gpu_memory_utilization
        _prof, disposition, _reason, _drift, missing = cleanup_mod._classify_entry(
            entry
        )
        # Profile match fails closed; entry falls through to quarantine path
        assert disposition == "quarantine"
        assert "gpu_memory_utilization" in missing


# ===========================================================================
# generate_paired_evidence_for_issue_105.py tests (comment 2)
# ===========================================================================


def _valid_paired_metrics() -> dict:
    """Metrics dict with all admission-critical fields present and valid."""
    return {
        "engine_version": "v0.17.2.post1-3628-ge4ce33646f",
        "git_commit": _VALID_ENGINE_COMMIT,
        "ttft_ms": 235.71,
        "tbt_ms": 39.99,
        "throughput_tps": 244.89,
        "error_rate": 0.0,
        "peak_mem_mb": 30000,
        "repetitions": 3,
        "has_measurement_block": True,
        "measurement_strategy": "median",
        "selected_run_index": 1,
        "verified": True,
        "target_id": "official-ascend-jan-2026-v0.18.0-random-online-qwen25-14b-910b2",
        "resolved_spec_hash": _VALID_RESOLVED_HASH,
        "engine_commit": _VALID_ENGINE_COMMIT,
        "plugin_commit": _VALID_PLUGIN_COMMIT,
    }


class TestPairedExtractMetrics:
    def test_extract_preserves_admission_fields(self, paired_mod):
        entry = {
            "engine_version": "v0.17.2.post1-3628-ge4ce33646f",
            "metrics": {
                "ttft_ms": 235.71,
                "tbt_ms": 39.99,
                "throughput_tps": 244.89,
                "peak_mem_mb": 30000,
                "error_rate": 0.0,
            },
            "measurement": {
                "per_run": [{"a": 1}, {"b": 2}, {"c": 3}],
                "strategy": "median",
                "selection": {"selected_run_index": 1},
            },
            "metadata": {
                "verified": True,
                "git_commit": _VALID_ENGINE_COMMIT,
                "runtime_provenance": {
                    "engine": {"commit": _VALID_ENGINE_COMMIT},
                    "plugin": {"commit": _VALID_PLUGIN_COMMIT},
                },
            },
            "same_spec": {
                "spec_id": "official-ascend-jan-2026-v0.18.0-random-online-qwen25-14b-910b2",
                "resolved_spec_hash": _VALID_RESOLVED_HASH,
            },
        }
        m = paired_mod._extract_metrics(entry)
        assert m["verified"] is True
        assert (
            m["target_id"]
            == "official-ascend-jan-2026-v0.18.0-random-online-qwen25-14b-910b2"
        )
        assert m["resolved_spec_hash"] == _VALID_RESOLVED_HASH
        assert m["engine_commit"] == _VALID_ENGINE_COMMIT
        assert m["plugin_commit"] == _VALID_PLUGIN_COMMIT
        assert m["peak_mem_mb"] == 30000
        assert m["error_rate"] == 0.0
        assert m["repetitions"] == 3
        assert m["has_measurement_block"] is True


class TestPairedIsValidSide:
    def test_valid_side_passes(self, paired_mod):
        ok, reason = paired_mod._is_valid_paired_side(_valid_paired_metrics())
        assert ok, f"unexpected reason: {reason}"

    def test_verified_none_rejected(self, paired_mod):
        m = _valid_paired_metrics()
        m["verified"] = None
        ok, reason = paired_mod._is_valid_paired_side(m)
        assert not ok
        assert "verified" in reason

    def test_verified_false_rejected(self, paired_mod):
        m = _valid_paired_metrics()
        m["verified"] = False
        ok, reason = paired_mod._is_valid_paired_side(m)
        assert not ok
        assert "verified" in reason

    def test_missing_target_id_rejected(self, paired_mod):
        m = _valid_paired_metrics()
        m["target_id"] = ""
        ok, reason = paired_mod._is_valid_paired_side(m)
        assert not ok
        assert "target_id" in reason

    def test_target_id_misaligned_rejected(self, paired_mod):
        m = _valid_paired_metrics()
        m["target_id"] = "some-other-target-random-online"
        ok, reason = paired_mod._is_valid_paired_side(
            m, expected_target_id="official-ascend-jan-2026-v0.18.0"
        )
        assert not ok
        assert "does not align" in reason

    def test_missing_resolved_spec_hash_rejected(self, paired_mod):
        m = _valid_paired_metrics()
        m["resolved_spec_hash"] = ""
        ok, reason = paired_mod._is_valid_paired_side(m)
        assert not ok
        assert "resolved_spec_hash" in reason

    def test_short_engine_commit_rejected(self, paired_mod):
        m = _valid_paired_metrics()
        m["engine_commit"] = "e4ce33646f"
        ok, reason = paired_mod._is_valid_paired_side(m)
        assert not ok
        assert "engine_commit" in reason

    def test_short_plugin_commit_rejected(self, paired_mod):
        m = _valid_paired_metrics()
        m["plugin_commit"] = "0f38988f47"
        ok, reason = paired_mod._is_valid_paired_side(m)
        assert not ok
        assert "plugin_commit" in reason

    def test_peak_mem_zero_rejected(self, paired_mod):
        m = _valid_paired_metrics()
        m["peak_mem_mb"] = 0
        ok, reason = paired_mod._is_valid_paired_side(m)
        assert not ok
        assert "peak_mem_mb" in reason

    def test_error_rate_one_rejected(self, paired_mod):
        m = _valid_paired_metrics()
        m["error_rate"] = 1.0
        ok, reason = paired_mod._is_valid_paired_side(m)
        assert not ok
        assert "error_rate" in reason

    def test_anomalous_ttft_rejected(self, paired_mod):
        """The 132s TTFT entry from the review summary."""
        m = _valid_paired_metrics()
        m["ttft_ms"] = 132137.44  # 132s
        ok, reason = paired_mod._is_valid_paired_side(m)
        assert not ok
        assert "anomalous" in reason


class TestPairedIsValidPairedEvidence:
    def test_both_valid_passes(self, paired_mod):
        base = _valid_paired_metrics()
        head = _valid_paired_metrics()
        ok, reason = paired_mod._is_valid_paired_evidence(base, head)
        assert ok, f"unexpected reason: {reason}"

    def test_base_verified_false_rejects_pair(self, paired_mod):
        base = _valid_paired_metrics()
        base["verified"] = False
        head = _valid_paired_metrics()
        ok, reason = paired_mod._is_valid_paired_evidence(base, head)
        assert not ok
        assert "base" in reason and "verified" in reason

    def test_head_peak_mem_zero_rejects_pair(self, paired_mod):
        base = _valid_paired_metrics()
        head = _valid_paired_metrics()
        head["peak_mem_mb"] = 0
        ok, reason = paired_mod._is_valid_paired_evidence(base, head)
        assert not ok
        assert "head" in reason and "peak_mem_mb" in reason


# ===========================================================================
# Idempotency guard tests (reviewer round 3: reverse test for second run)
# ===========================================================================


class TestCleanupIdempotencyGuard:
    """Reverse tests for the idempotency guard (issue #105 reviewer round 3).

    Per reviewer: '重复运行脚本还可能用已经清空的 leaderboard 覆盖第一次生成的
    quarantine。请从干净的 4d549c3 输入重新生成，确保 0 keep、21 quarantine，
    并让 cleanup 重跑保持既有 quarantine 或明确拒绝对已清理输入覆写；补一个
    幂等/二次运行反向测试'.

    These tests verify that:
    1. A second run on an already-cleaned input (0 entries) is rejected when
       pre_cleanup_freeze.json shows a prior run with > 0 entries.
    2. The --force flag bypasses the guard.
    3. The guard does NOT trigger when the input has entries (first run).
    4. The guard does NOT trigger when pre_cleanup_freeze.json is absent.
    """

    def test_second_run_rejected_when_input_cleaned(
        self, cleanup_mod, tmp_path, monkeypatch, capsys
    ):
        """main() must return exit code 3 on already-cleaned input."""
        snapshot_dir = tmp_path / "snapshots"
        snapshot_dir.mkdir()
        # Already-cleaned input (0 entries)
        (snapshot_dir / "leaderboard_single.json").write_text("[]\n")
        # Prior freeze shows 21 entries from a previous run
        freeze = {
            "schema_version": "pre-cleanup-freeze/v1",
            "entry_count": 21,
            "entry_ids": [f"id-{i}" for i in range(21)],
            "frozen_entries": [],
        }
        (snapshot_dir / "pre_cleanup_freeze.json").write_text(json.dumps(freeze) + "\n")
        monkeypatch.setattr(
            "sys.argv",
            ["cleanup", "--snapshot-dir", str(snapshot_dir)],
        )
        rc = cleanup_mod.main()
        assert rc == 3
        captured = capsys.readouterr()
        assert "Idempotency guard" in captured.err

    def test_force_flag_bypasses_guard(self, cleanup_mod, tmp_path, monkeypatch):
        """--force flag must bypass the idempotency guard on non-empty input."""
        snapshot_dir = tmp_path / "snapshots"
        snapshot_dir.mkdir()
        # Non-empty input (restored original) — required for --force
        entry = _valid_active_entry()
        (snapshot_dir / "leaderboard_single.json").write_text(
            json.dumps([entry]) + "\n"
        )
        freeze = {
            "schema_version": "pre-cleanup-freeze/v1",
            "entry_count": 21,
            "entry_ids": [],
            "frozen_entries": [],
        }
        (snapshot_dir / "pre_cleanup_freeze.json").write_text(json.dumps(freeze) + "\n")
        monkeypatch.setattr(
            "sys.argv",
            [
                "cleanup",
                "--snapshot-dir",
                str(snapshot_dir),
                "--force",
            ],
        )
        # Should NOT raise — force bypasses guard on restored non-empty input
        rc = cleanup_mod.main()
        assert rc == 0

    def test_first_run_not_blocked(self, cleanup_mod, tmp_path):
        """First run on input with entries must NOT trigger the guard."""
        snapshot_dir = tmp_path / "snapshots"
        snapshot_dir.mkdir()
        entry = _valid_active_entry()
        (snapshot_dir / "leaderboard_single.json").write_text(
            json.dumps([entry]) + "\n"
        )
        # No pre_cleanup_freeze.json — first run
        kept, removed = cleanup_mod.cleanup_snapshot(
            snapshot_dir / "leaderboard_single.json",
            snapshot_dir / "leaderboard_single.json",
            [],
            [],
        )
        # Entry has verified=True, peak_mem>0, etc. — should be kept
        assert kept == 1
        assert removed == 0

    def test_guard_not_triggered_when_freeze_absent(self, cleanup_mod, tmp_path):
        """0-entry input without pre_cleanup_freeze.json must NOT trigger guard."""
        snapshot_dir = tmp_path / "snapshots"
        snapshot_dir.mkdir()
        (snapshot_dir / "leaderboard_single.json").write_text("[]\n")
        # No pre_cleanup_freeze.json — guard should not trigger
        kept, removed = cleanup_mod.cleanup_snapshot(
            snapshot_dir / "leaderboard_single.json",
            snapshot_dir / "leaderboard_single.json",
            [],
            [],
        )
        assert kept == 0
        assert removed == 0

    def test_guard_not_triggered_when_freeze_has_zero_entries(
        self, cleanup_mod, tmp_path
    ):
        """0-entry input with freeze also showing 0 must NOT trigger guard."""
        snapshot_dir = tmp_path / "snapshots"
        snapshot_dir.mkdir()
        (snapshot_dir / "leaderboard_single.json").write_text("[]\n")
        freeze = {
            "schema_version": "pre-cleanup-freeze/v1",
            "entry_count": 0,
            "entry_ids": [],
            "frozen_entries": [],
        }
        (snapshot_dir / "pre_cleanup_freeze.json").write_text(json.dumps(freeze) + "\n")
        # Both input and freeze show 0 — not a re-run scenario
        kept, removed = cleanup_mod.cleanup_snapshot(
            snapshot_dir / "leaderboard_single.json",
            snapshot_dir / "leaderboard_single.json",
            [],
            [],
        )
        assert kept == 0
        assert removed == 0

    # ------------------------------------------------------------------
    # Round 4 reverse tests: fail-closed on corrupted/missing/wrong-type
    # freeze fields, and --force requires non-empty input.
    # ------------------------------------------------------------------

    def test_corrupted_freeze_json_rejected(
        self, cleanup_mod, tmp_path, monkeypatch, capsys
    ):
        """Corrupted pre_cleanup_freeze.json must trigger guard (rc=3)."""
        snapshot_dir = tmp_path / "snapshots"
        snapshot_dir.mkdir()
        (snapshot_dir / "leaderboard_single.json").write_text("[]\n")
        # Corrupted JSON — not parseable
        (snapshot_dir / "pre_cleanup_freeze.json").write_text(
            "{this is not valid json}\n"
        )
        monkeypatch.setattr(
            "sys.argv",
            ["cleanup", "--snapshot-dir", str(snapshot_dir)],
        )
        rc = cleanup_mod.main()
        assert rc == 3
        captured = capsys.readouterr()
        assert "cannot be parsed" in captured.err

    def test_freeze_missing_entry_count_rejected(
        self, cleanup_mod, tmp_path, monkeypatch, capsys
    ):
        """Freeze missing 'entry_count' field must trigger guard (rc=3)."""
        snapshot_dir = tmp_path / "snapshots"
        snapshot_dir.mkdir()
        (snapshot_dir / "leaderboard_single.json").write_text("[]\n")
        # Freeze is valid JSON dict but missing entry_count
        freeze = {
            "schema_version": "pre-cleanup-freeze/v1",
            "entry_ids": ["id-1"],
            "frozen_entries": [],
        }
        (snapshot_dir / "pre_cleanup_freeze.json").write_text(json.dumps(freeze) + "\n")
        monkeypatch.setattr(
            "sys.argv",
            ["cleanup", "--snapshot-dir", str(snapshot_dir)],
        )
        rc = cleanup_mod.main()
        assert rc == 3
        captured = capsys.readouterr()
        assert "entry_count" in captured.err

    def test_freeze_entry_count_wrong_type_rejected(
        self, cleanup_mod, tmp_path, monkeypatch, capsys
    ):
        """Freeze with entry_count as string must trigger guard (rc=3)."""
        snapshot_dir = tmp_path / "snapshots"
        snapshot_dir.mkdir()
        (snapshot_dir / "leaderboard_single.json").write_text("[]\n")
        # entry_count is a string "21" instead of int 21
        freeze = {
            "schema_version": "pre-cleanup-freeze/v1",
            "entry_count": "21",
            "entry_ids": [],
            "frozen_entries": [],
        }
        (snapshot_dir / "pre_cleanup_freeze.json").write_text(json.dumps(freeze) + "\n")
        monkeypatch.setattr(
            "sys.argv",
            ["cleanup", "--snapshot-dir", str(snapshot_dir)],
        )
        rc = cleanup_mod.main()
        assert rc == 3
        captured = capsys.readouterr()
        assert "wrong type" in captured.err

    def test_freeze_not_a_dict_rejected(
        self, cleanup_mod, tmp_path, monkeypatch, capsys
    ):
        """Freeze that is a JSON list instead of dict must trigger guard (rc=3)."""
        snapshot_dir = tmp_path / "snapshots"
        snapshot_dir.mkdir()
        (snapshot_dir / "leaderboard_single.json").write_text("[]\n")
        # Freeze is a JSON list, not an object
        (snapshot_dir / "pre_cleanup_freeze.json").write_text("[1, 2, 3]\n")
        monkeypatch.setattr(
            "sys.argv",
            ["cleanup", "--snapshot-dir", str(snapshot_dir)],
        )
        rc = cleanup_mod.main()
        assert rc == 3
        captured = capsys.readouterr()
        assert "not a JSON object" in captured.err

    def test_force_on_empty_input_with_freeze_rejected(
        self, cleanup_mod, tmp_path, monkeypatch, capsys
    ):
        """--force on empty input with existing freeze must be refused (rc=4)."""
        snapshot_dir = tmp_path / "snapshots"
        snapshot_dir.mkdir()
        (snapshot_dir / "leaderboard_single.json").write_text("[]\n")
        freeze = {
            "schema_version": "pre-cleanup-freeze/v1",
            "entry_count": 21,
            "entry_ids": [],
            "frozen_entries": [],
        }
        (snapshot_dir / "pre_cleanup_freeze.json").write_text(json.dumps(freeze) + "\n")
        monkeypatch.setattr(
            "sys.argv",
            [
                "cleanup",
                "--snapshot-dir",
                str(snapshot_dir),
                "--force",
            ],
        )
        rc = cleanup_mod.main()
        assert rc == 4
        captured = capsys.readouterr()
        assert "--force" in captured.err

    def test_force_on_nonempty_input_succeeds(self, cleanup_mod, tmp_path, monkeypatch):
        """--force on non-empty input with existing freeze must succeed (rc=0)."""
        snapshot_dir = tmp_path / "snapshots"
        snapshot_dir.mkdir()
        # Non-empty input (restored original)
        entry = _valid_active_entry()
        (snapshot_dir / "leaderboard_single.json").write_text(
            json.dumps([entry]) + "\n"
        )
        freeze = {
            "schema_version": "pre-cleanup-freeze/v1",
            "entry_count": 21,
            "entry_ids": [],
            "frozen_entries": [],
        }
        (snapshot_dir / "pre_cleanup_freeze.json").write_text(json.dumps(freeze) + "\n")
        monkeypatch.setattr(
            "sys.argv",
            [
                "cleanup",
                "--snapshot-dir",
                str(snapshot_dir),
                "--force",
            ],
        )
        rc = cleanup_mod.main()
        assert rc == 0

    def test_corrupted_freeze_with_force_and_nonempty_succeeds(
        self, cleanup_mod, tmp_path, monkeypatch
    ):
        """--force on non-empty input with corrupted freeze must succeed (rc=0).

        The fail-closed corruption check only applies when the input is empty
        (re-run scenario).  With a non-empty restored input and --force, the
        user is explicitly regenerating from a known-good original.
        """
        snapshot_dir = tmp_path / "snapshots"
        snapshot_dir.mkdir()
        entry = _valid_active_entry()
        (snapshot_dir / "leaderboard_single.json").write_text(
            json.dumps([entry]) + "\n"
        )
        # Corrupted freeze
        (snapshot_dir / "pre_cleanup_freeze.json").write_text("{corrupted json}\n")
        monkeypatch.setattr(
            "sys.argv",
            [
                "cleanup",
                "--snapshot-dir",
                str(snapshot_dir),
                "--force",
            ],
        )
        rc = cleanup_mod.main()
        assert rc == 0


class TestCleanupPreservesIssue146SuspectSection:
    def test_regen_preserves_suspect_section(self, cleanup_mod, tmp_path, monkeypatch):
        snapshot_dir = tmp_path / "snapshots"
        snapshot_dir.mkdir()
        entry = _valid_active_entry()
        (snapshot_dir / "leaderboard_single.json").write_text(
            json.dumps([entry]) + "\n"
        )
        freeze = {
            "schema_version": "pre-cleanup-freeze/v1",
            "entry_count": 21,
            "entry_ids": [],
            "frozen_entries": [],
        }
        (snapshot_dir / "pre_cleanup_freeze.json").write_text(json.dumps(freeze) + "\n")
        prior_suspect = {
            "schema_version": "issue-146-suspect/v2",
            "conclusion": "no_regression_reproduced",
            "action": "mark_suspect_noise",
            "analysis_provenance": "reports/issue_146_retest_analysis.json",
            "raw_evidence_dir": "reports/issue_146_retest_raw_results/",
            "note": "no regression reproduced",
            "entries": [
                {
                    "git_commit": "7a63f81e86bd71e980adb635870ff56c9e23b545",  # pragma: allowlist secret  # pragma: allowlist secret
                    "workload": "sonnet-throughput",
                    "workload_params": {
                        "input_length": 1024,
                        "output_length": 256,
                        "batch_size": None,
                        "dataset": "sonnet",
                    },
                    "model": "Qwen/Qwen2.5-14B-Instruct",
                    "original_value": 1589.93,
                    "original_value_unit": "tok/s",
                    "retest_median": 2898.8,
                    "retest_median_unit": "tok/s",
                    "threshold_pct": 10.0,
                    "status": "invalid-suspect-noise",
                    "retest_base_commit": "2206f1f7b7212801187bc001c5f6cb86b2289214",  # pragma: allowlist secret  # pragma: allowlist secret
                    "retest_delta_vs_base_commit_pct": 0.24,
                }
            ],
        }
        (snapshot_dir / "quarantine_leaderboard_entries.json").write_text(
            json.dumps(
                {
                    "schema_version": "quarantine-leaderboard-entries/v2",
                    "quarantined_count": 0,
                    "quarantined_entries": [],
                    "issue_146_suspect_entries": prior_suspect,
                }
            )
            + "\n"
        )
        monkeypatch.setattr(
            "sys.argv",
            ["cleanup", "--snapshot-dir", str(snapshot_dir), "--force"],
        )
        rc = cleanup_mod.main()
        assert rc == 0
        regenerated = json.loads(
            (snapshot_dir / "quarantine_leaderboard_entries.json").read_text(
                encoding="utf-8"
            )
        )
        # [major] review: regeneration must NOT silently drop the additive section.
        assert regenerated["schema_version"] == "quarantine-leaderboard-entries/v2"
        assert regenerated["suspect_entries_count"] == 1
        assert (
            regenerated["issue_146_suspect_entries"]["schema_version"]
            == "issue-146-suspect/v2"
        )
        assert (
            regenerated["issue_146_suspect_entries"]["entries"][0]["git_commit"]
            == "7a63f81e86bd71e980adb635870ff56c9e23b545"  # pragma: allowlist secret
        )
