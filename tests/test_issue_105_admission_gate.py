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
