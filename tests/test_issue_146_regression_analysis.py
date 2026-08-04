"""Tests for issue #146 regression re-test analysis."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "analyze_issue_146_regression.py"


@pytest.fixture(scope="module")
def analyze_mod():
    spec = importlib.util.spec_from_file_location("analyze_issue_146", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# _extract_metric
# ---------------------------------------------------------------------------


class TestExtractMetric:
    def test_sonnet_throughput_tokens_per_second(self, analyze_mod):
        """Actual vllm bench throughput output uses tokens_per_second."""
        raw = {"tokens_per_second": 2849.10, "requests_per_second": 4.13}
        assert analyze_mod._extract_metric(raw, "sonnet-throughput") == 2849.10

    def test_sonnet_throughput_throughput_tps(self, analyze_mod):
        """Leaderboard export format uses throughput_tps."""
        raw = {"throughput_tps": 498.38}
        assert analyze_mod._extract_metric(raw, "sonnet-throughput") == 498.38

    def test_sonnet_throughput_legacy_tokens_slash(self, analyze_mod):
        """Older vllm versions used tokens/s."""
        raw = {"tokens/s": 1926.63}
        assert analyze_mod._extract_metric(raw, "sonnet-throughput") == 1926.63

    def test_sonnet_throughput_nested(self, analyze_mod):
        raw = {"throughput": {"tokens/s": 500.0}}
        assert analyze_mod._extract_metric(raw, "sonnet-throughput") == 500.0

    def test_sonnet_throughput_missing(self, analyze_mod):
        raw = {"other": 1}
        assert analyze_mod._extract_metric(raw, "sonnet-throughput") is None

    def test_random_latency_mean_ttft_ms(self, analyze_mod):
        """Serve benchmark output uses mean_ttft_ms directly."""
        raw = {"mean_ttft_ms": 7483.04}
        assert analyze_mod._extract_metric(raw, "random-latency") == 7483.04

    def test_random_latency_avg_latency_seconds_to_ms(self, analyze_mod):
        """bench latency outputs avg_latency in seconds; convert to ms."""
        raw = {"avg_latency": 12.20170}
        assert analyze_mod._extract_metric(raw, "random-latency") == pytest.approx(
            12201.70, abs=0.01
        )

    def test_random_latency_ttft_ms(self, analyze_mod):
        raw = {"ttft_ms": 7483.04}
        assert analyze_mod._extract_metric(raw, "random-latency") == 7483.04

    def test_random_latency_p50(self, analyze_mod):
        raw = {"p50": 5000.0}
        assert analyze_mod._extract_metric(raw, "random-latency") == 5000.0

    def test_random_latency_missing(self, analyze_mod):
        raw = {"other": 1}
        assert analyze_mod._extract_metric(raw, "random-latency") is None

    def test_unknown_workload(self, analyze_mod):
        raw = {"tokens_per_second": 100}
        assert analyze_mod._extract_metric(raw, "unknown-workload") is None


# ---------------------------------------------------------------------------
# _delta_pct
# ---------------------------------------------------------------------------


class TestDeltaPct:
    def test_positive_delta(self, analyze_mod):
        assert analyze_mod._delta_pct(100.0, 150.0) == 50.0

    def test_negative_delta(self, analyze_mod):
        assert analyze_mod._delta_pct(1926.63, 1589.93) == pytest.approx(-17.5, abs=0.1)

    def test_zero_base(self, analyze_mod):
        assert analyze_mod._delta_pct(0.0, 100.0) is None

    def test_none_base(self, analyze_mod):
        assert analyze_mod._delta_pct(None, 100.0) is None

    def test_none_head(self, analyze_mod):
        assert analyze_mod._delta_pct(100.0, None) is None


# ---------------------------------------------------------------------------
# compute_medians
# ---------------------------------------------------------------------------


class TestComputeMedians:
    def test_empty_results(self, analyze_mod):
        results = {"sonnet-throughput": {"2206f1f7b7": []}}
        summary = analyze_mod.compute_medians(results)
        assert summary["sonnet-throughput"]["2206f1f7b7"]["median"] is None
        assert summary["sonnet-throughput"]["2206f1f7b7"]["count"] == 0

    def test_single_value(self, analyze_mod):
        results = {"sonnet-throughput": {"2206f1f7b7": [500.0]}}
        summary = analyze_mod.compute_medians(results)
        assert summary["sonnet-throughput"]["2206f1f7b7"]["median"] == 500.0
        assert summary["sonnet-throughput"]["2206f1f7b7"]["count"] == 1

    def test_three_values(self, analyze_mod):
        results = {"sonnet-throughput": {"2206f1f7b7": [400.0, 500.0, 600.0]}}
        summary = analyze_mod.compute_medians(results)
        assert summary["sonnet-throughput"]["2206f1f7b7"]["median"] == 500.0
        assert summary["sonnet-throughput"]["2206f1f7b7"]["min"] == 400.0
        assert summary["sonnet-throughput"]["2206f1f7b7"]["max"] == 600.0

    def test_even_count(self, analyze_mod):
        results = {"random-latency": {"2206f1f7b7": [100.0, 200.0, 300.0, 400.0]}}
        summary = analyze_mod.compute_medians(results)
        assert summary["random-latency"]["2206f1f7b7"]["median"] == 250.0


# ---------------------------------------------------------------------------
# analyze_regression
# ---------------------------------------------------------------------------


class TestAnalyzeRegression:
    def _make_summary(
        self,
        sonnet_base: float | None = 1926.0,
        sonnet_head: float | None = 1589.0,
        latency_base: float | None = 7483.0,
        latency_head: float | None = 12201.0,
        sonnet_base_count: int = 3,
        sonnet_head_count: int = 3,
        latency_base_count: int = 3,
        latency_head_count: int = 3,
        sonnet_base_values: list[float] | None = None,
        sonnet_head_values: list[float] | None = None,
        latency_base_values: list[float] | None = None,
        latency_head_values: list[float] | None = None,
    ):
        def _entry(median, count, values):
            if values is None:
                values = [median] * count if median is not None else []
            return {"median": median, "count": count, "values": values}

        return {
            "sonnet-throughput": {
                "2206f1f7b7": _entry(
                    sonnet_base, sonnet_base_count, sonnet_base_values
                ),
                "7a63f81e86": _entry(
                    sonnet_head, sonnet_head_count, sonnet_head_values
                ),
                "83cf83ff20": _entry(None, 0, []),
            },
            "random-latency": {
                "2206f1f7b7": _entry(
                    latency_base, latency_base_count, latency_base_values
                ),
                "7a63f81e86": _entry(None, 0, []),
                "83cf83ff20": _entry(
                    latency_head, latency_head_count, latency_head_values
                ),
            },
        }

    def test_both_regressions_confirmed(self, analyze_mod):
        summary = self._make_summary()
        findings = analyze_mod.analyze_regression(summary)
        assert findings["sonnet-throughput"]["regression_reproducible"] is True
        assert findings["random-latency"]["regression_reproducible"] is True
        assert findings["overall"]["any_regression_confirmed"] is True
        assert findings["overall"]["action"] == "bisect_and_fix"

    def test_sonnet_no_regression(self, analyze_mod):
        summary = self._make_summary(sonnet_base=1926.0, sonnet_head=1850.0)
        findings = analyze_mod.analyze_regression(summary)
        assert findings["sonnet-throughput"]["regression_reproducible"] is False
        assert findings["sonnet-throughput"]["conclusion"] == "no_regression_or_noise"

    def test_random_latency_no_regression(self, analyze_mod):
        summary = self._make_summary(latency_base=7483.0, latency_head=8000.0)
        findings = analyze_mod.analyze_regression(summary)
        assert findings["random-latency"]["regression_reproducible"] is False
        assert findings["random-latency"]["conclusion"] == "no_regression_or_noise"

    def test_both_no_regression(self, analyze_mod):
        summary = self._make_summary(
            sonnet_base=1926.0,
            sonnet_head=1850.0,
            latency_base=7483.0,
            latency_head=8000.0,
        )
        findings = analyze_mod.analyze_regression(summary)
        assert findings["overall"]["any_regression_confirmed"] is False
        assert findings["overall"]["action"] == "no_action_diagnostic_only"

    def test_sonnet_at_threshold(self, analyze_mod):
        """Exactly 10% drop is not a regression (not strictly less than -10%)."""
        summary = self._make_summary(sonnet_base=1000.0, sonnet_head=900.0)
        findings = analyze_mod.analyze_regression(summary)
        assert findings["sonnet-throughput"]["regression_reproducible"] is False

    def test_sonnet_just_below_threshold(self, analyze_mod):
        """11% drop IS a regression (strictly less than -10%)."""
        summary = self._make_summary(sonnet_base=1000.0, sonnet_head=890.0)
        findings = analyze_mod.analyze_regression(summary)
        assert findings["sonnet-throughput"]["regression_reproducible"] is True

    def test_random_latency_at_threshold(self, analyze_mod):
        """Exactly 20% increase is not a regression (not strictly greater than 20%)."""
        summary = self._make_summary(latency_base=1000.0, latency_head=1200.0)
        findings = analyze_mod.analyze_regression(summary)
        assert findings["random-latency"]["regression_reproducible"] is False

    def test_none_values(self, analyze_mod):
        summary = self._make_summary(
            sonnet_base=None,
            sonnet_head=None,
            latency_base=None,
            latency_head=None,
            sonnet_base_count=0,
            sonnet_head_count=0,
            latency_base_count=0,
            latency_head_count=0,
        )
        findings = analyze_mod.analyze_regression(summary)
        assert findings["sonnet-throughput"]["regression_reproducible"] is False
        assert findings["random-latency"]["regression_reproducible"] is False
        assert findings["overall"]["action"] == "incomplete_evidence"
        assert findings["overall"]["any_evidence_incomplete"] is True

    def test_insufficient_reps_marks_incomplete(self, analyze_mod):
        """Fewer than 3 reps must yield incomplete, not no_regression."""
        summary = self._make_summary(
            sonnet_base=1926.0,
            sonnet_head=1850.0,
            sonnet_head_count=2,
            sonnet_head_values=[1850.0, 1860.0],
        )
        findings = analyze_mod.analyze_regression(summary)
        assert findings["sonnet-throughput"]["conclusion"] == "incomplete"
        assert findings["sonnet-throughput"]["evidence_sufficient"] is False
        assert findings["overall"]["action"] == "incomplete_evidence"

    def test_zero_or_nan_metric_marks_incomplete(self, analyze_mod):
        """Zero or NaN values must yield incomplete, not no_regression."""
        summary = self._make_summary(
            sonnet_base=1926.0,
            sonnet_head=0.0,
            sonnet_head_values=[0.0, 0.0, 0.0],
        )
        findings = analyze_mod.analyze_regression(summary)
        assert findings["sonnet-throughput"]["conclusion"] == "incomplete"
        assert findings["overall"]["action"] == "incomplete_evidence"

    def test_no_clean_leaderboard_action(self, analyze_mod):
        """The action must never be 'clean_leaderboard_points'."""
        summary = self._make_summary(
            sonnet_base=1926.0,
            sonnet_head=1850.0,
            latency_base=7483.0,
            latency_head=8000.0,
        )
        findings = analyze_mod.analyze_regression(summary)
        assert findings["overall"]["action"] != "clean_leaderboard_points"


# ---------------------------------------------------------------------------
# collect_results (uses temp directory)
# ---------------------------------------------------------------------------


class TestCollectResults:
    def test_collect_from_directory(self, analyze_mod, tmp_path):
        # Create mock results with actual vllm bench output field names
        for commit in ["2206f1f7b7", "7a63f81e86"]:
            for rep in [1, 2, 3]:
                d = tmp_path / commit / "sonnet-throughput" / f"rep-{rep}"
                d.mkdir(parents=True)
                (d / "raw.json").write_text(
                    json.dumps({"tokens_per_second": 1000.0 + rep * 10})
                )

        results = analyze_mod.collect_results(tmp_path)
        assert "sonnet-throughput" in results
        assert "2206f1f7b7" in results["sonnet-throughput"]
        assert len(results["sonnet-throughput"]["2206f1f7b7"]) == 3
        assert results["sonnet-throughput"]["2206f1f7b7"][0] == 1010.0

    def test_collect_latency_with_avg_latency_seconds(self, analyze_mod, tmp_path):
        """Verify avg_latency (seconds) is collected and converted to ms."""
        for rep in [1, 2, 3]:
            d = tmp_path / "2206f1f7b7" / "random-latency" / f"rep-{rep}"
            d.mkdir(parents=True)
            (d / "raw.json").write_text(json.dumps({"avg_latency": 10.0 + rep * 0.1}))

        results = analyze_mod.collect_results(tmp_path)
        # rep-1: avg_latency=10.1s -> 10100.0 ms
        assert results["random-latency"]["2206f1f7b7"][0] == pytest.approx(
            10100.0, abs=0.1
        )

    def test_missing_directory(self, analyze_mod, tmp_path):
        results = analyze_mod.collect_results(tmp_path)
        for workload in ["sonnet-throughput", "random-latency"]:
            for commit in ["2206f1f7b7", "7a63f81e86", "83cf83ff20"]:
                assert results[workload][commit] == []

    def test_corrupt_json_skipped(self, analyze_mod, tmp_path):
        d = tmp_path / "2206f1f7b7" / "sonnet-throughput" / "rep-1"
        d.mkdir(parents=True)
        (d / "raw.json").write_text("not valid json")
        results = analyze_mod.collect_results(tmp_path)
        assert results["sonnet-throughput"]["2206f1f7b7"] == []
