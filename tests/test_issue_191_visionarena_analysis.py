"""Tests for issue #191 visionarena-online regression re-test analysis."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "analyze_issue_191_visionarena.py"


@pytest.fixture(scope="module")
def analyze_mod():
    spec = importlib.util.spec_from_file_location(
        "analyze_issue_191_visionarena", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_raw_json(
    rep_dir: Path,
    ttft: float = 100.0,
    tpot: float = 50.0,
    throughput: float = 200.0,
    completed: int = 1000,
    failed: int = 0,
) -> None:
    """Write a raw.json with the given metrics and completion counters."""
    data = {
        "mean_ttft_ms": ttft,
        "mean_tpot_ms": tpot,
        "output_throughput": throughput,
        "completed": completed,
        "failed": failed,
    }
    (rep_dir / "raw.json").write_text(json.dumps(data))


def _write_env_manifest(
    rep_dir: Path, engine_sha: str = "a" * 40, plugin_sha: str = "b" * 40
) -> None:
    """Write a valid env-manifest.json."""
    manifest = {
        "engine_commit_requested": engine_sha[:10],
        "engine_commit_observed": engine_sha,
        "plugin_commit_requested": plugin_sha[:10],
        "plugin_commit_observed": plugin_sha,
        "python_version": "Python 3.11.15",
        "max_model_len": 32768,
        "gpu_memory_utilization": 0.6,
        "dtype": "float16",
        "artifact_class": "diagnostic_historical_retest",
        "official_target": False,
    }
    (rep_dir / "env-manifest.json").write_text(json.dumps(manifest))


def _make_rep(tmp_path: Path, commit: str, workload: str, rep: int, **kwargs) -> Path:
    """Create a complete valid rep directory."""
    rep_dir = tmp_path / commit / workload / f"rep-{rep}"
    rep_dir.mkdir(parents=True)
    _write_raw_json(rep_dir, **kwargs)
    _write_env_manifest(rep_dir)
    (rep_dir / ".completed").touch()
    return rep_dir


_INTERVAL = {
    "name": "test-interval",
    "base_commit": "aaaaaaaaaa",
    "head_commit": "bbbbbbbbbb",
    "workload": "test-workload",
    "reported_jump": "test jump",
}


# ---------------------------------------------------------------------------
# is_valid_sha
# ---------------------------------------------------------------------------


class TestIsValidSha:
    def test_valid_40_char_sha(self, analyze_mod):
        assert analyze_mod.is_valid_sha("a" * 40)

    def test_none_returns_false(self, analyze_mod):
        assert not analyze_mod.is_valid_sha(None)

    def test_short_sha_returns_false(self, analyze_mod):
        assert not analyze_mod.is_valid_sha("a" * 39)

    def test_non_hex_returns_false(self, analyze_mod):
        assert not analyze_mod.is_valid_sha("z" * 40)


# ---------------------------------------------------------------------------
# load_raw_metrics
# ---------------------------------------------------------------------------


class TestLoadRawMetrics:
    def test_valid_raw_json(self, analyze_mod, tmp_path):
        _write_raw_json(tmp_path, ttft=150.0, tpot=55.0, throughput=180.0)
        metrics = analyze_mod.load_raw_metrics(tmp_path)
        assert metrics is not None
        assert metrics["mean_ttft_ms"] == 150.0
        assert metrics["mean_tpot_ms"] == 55.0
        assert metrics["output_throughput"] == 180.0
        assert metrics["completed"] == 1000
        assert metrics["failed"] == 0

    def test_missing_raw_json(self, analyze_mod, tmp_path):
        assert analyze_mod.load_raw_metrics(tmp_path) is None

    def test_missing_completed_fail_closed(self, analyze_mod, tmp_path):
        """A rep without completed/failed counters is rejected fail-closed."""
        data = {
            "mean_ttft_ms": 100.0,
            "mean_tpot_ms": 50.0,
            "output_throughput": 200.0,
        }
        (tmp_path / "raw.json").write_text(json.dumps(data))
        assert analyze_mod.load_raw_metrics(tmp_path) is None

    def test_missing_metric_field(self, analyze_mod, tmp_path):
        data = {
            "mean_ttft_ms": 100.0,
            "mean_tpot_ms": 50.0,
            "completed": 1000,
            "failed": 0,
        }
        (tmp_path / "raw.json").write_text(json.dumps(data))
        assert analyze_mod.load_raw_metrics(tmp_path) is None


# ---------------------------------------------------------------------------
# collect_rep_results (failure-rate gate)
# ---------------------------------------------------------------------------


class TestCollectRepResults:
    def test_valid_rep_accepted(self, analyze_mod, tmp_path):
        _make_rep(tmp_path, "aaaaaaaaaa", "test-workload", 1)
        results = analyze_mod.collect_rep_results(
            tmp_path, "aaaaaaaaaa", "test-workload", None, None
        )
        assert len(results) == 1

    def test_missing_completed_marker_skipped(self, analyze_mod, tmp_path):
        rep_dir = tmp_path / "aaaaaaaaaa" / "test-workload" / "rep-1"
        rep_dir.mkdir(parents=True)
        _write_raw_json(rep_dir)
        _write_env_manifest(rep_dir)
        results = analyze_mod.collect_rep_results(
            tmp_path, "aaaaaaaaaa", "test-workload", None, None
        )
        assert len(results) == 0

    def test_high_failure_rate_rejected(self, analyze_mod, tmp_path):
        """A rep failing >1% of requests is rejected fail-closed."""
        _make_rep(
            tmp_path,
            "aaaaaaaaaa",
            "test-workload",
            1,
            completed=287,
            failed=713,
        )
        results = analyze_mod.collect_rep_results(
            tmp_path, "aaaaaaaaaa", "test-workload", None, None
        )
        assert len(results) == 0

    def test_zero_failure_rate_accepted(self, analyze_mod, tmp_path):
        _make_rep(
            tmp_path,
            "aaaaaaaaaa",
            "test-workload",
            1,
            completed=1000,
            failed=0,
        )
        results = analyze_mod.collect_rep_results(
            tmp_path, "aaaaaaaaaa", "test-workload", None, None
        )
        assert len(results) == 1


# ---------------------------------------------------------------------------
# compare_interval (fail-closed)
# ---------------------------------------------------------------------------


class TestCompareInterval:
    def _setup_interval(
        self,
        tmp_path: Path,
        reps: int = 3,
        base_completed: int = 1000,
        base_failed: int = 0,
        head_completed: int = 1000,
        head_failed: int = 0,
    ) -> Path:
        for i in range(1, reps + 1):
            _make_rep(
                tmp_path,
                "aaaaaaaaaa",
                "test-workload",
                i,
                completed=base_completed,
                failed=base_failed,
            )
            _make_rep(
                tmp_path,
                "bbbbbbbbbb",
                "test-workload",
                i,
                completed=head_completed,
                failed=head_failed,
            )
        return tmp_path

    def test_not_reproducible_with_three_valid_reps(self, analyze_mod, tmp_path):
        """3 valid reps per side with small change -> not_reproducible."""
        self._setup_interval(tmp_path)
        result = analyze_mod.compare_interval(dict(_INTERVAL), tmp_path, None, None)
        assert result["verdict"] == "not_reproducible"
        assert result["base_reps"] == 3
        assert result["head_reps"] == 3

    def test_incomplete_evidence_two_reps(self, analyze_mod, tmp_path):
        """Only 2 valid reps per side -> incomplete_evidence (fail-closed)."""
        self._setup_interval(tmp_path, reps=2)
        result = analyze_mod.compare_interval(dict(_INTERVAL), tmp_path, None, None)
        assert result["verdict"] == "incomplete_evidence"
        assert result["base_reps"] == 2
        assert result["head_reps"] == 2

    def test_incomplete_evidence_one_side_missing(self, analyze_mod, tmp_path):
        """Base side missing -> incomplete_evidence."""
        self._setup_interval(tmp_path)
        import shutil

        shutil.rmtree(tmp_path / "aaaaaaaaaa")
        result = analyze_mod.compare_interval(dict(_INTERVAL), tmp_path, None, None)
        assert result["verdict"] == "incomplete_evidence"
        assert result["base_reps"] == 0
        assert result["head_reps"] == 3

    def test_incomplete_evidence_high_failure_rate(self, analyze_mod, tmp_path):
        """Any side with reps failing the rate gate -> incomplete_evidence."""
        self._setup_interval(
            tmp_path,
            head_completed=287,
            head_failed=713,
        )
        result = analyze_mod.compare_interval(dict(_INTERVAL), tmp_path, None, None)
        assert result["verdict"] == "incomplete_evidence"
        assert result["head_reps"] == 0
        assert result["base_reps"] == 3
