"""Tests for issue #151 regression re-test analysis."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "analyze_issue_151_regression.py"


@pytest.fixture(scope="module")
def analyze_mod():
    spec = importlib.util.spec_from_file_location("analyze_issue_151", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_raw_json(
    rep_dir: Path, ttft: float = 100.0, tpot: float = 50.0, throughput: float = 200.0
) -> None:
    """Write a valid raw.json with the given metrics and default failure counts.

    failed/num_prompts are required by the analyzer's failure-rate
    policy (MAX_FAILURE_RATE); default to a clean run so reps pass.
    """
    data = {
        "mean_ttft_ms": ttft,
        "mean_tpot_ms": tpot,
        "output_throughput": throughput,
        "failed": 0,
        "num_prompts": 100,
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


# ---------------------------------------------------------------------------
# is_valid_sha
# ---------------------------------------------------------------------------


class TestIsValidSha:
    def test_valid_40_char_sha(self, analyze_mod):
        sha = "a" * 40
        assert analyze_mod.is_valid_sha(sha) is True

    def test_valid_hex_sha(self, analyze_mod):
        sha = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret
        assert analyze_mod.is_valid_sha(sha) is True

    def test_none_returns_false(self, analyze_mod):
        assert analyze_mod.is_valid_sha(None) is False

    def test_empty_string_returns_false(self, analyze_mod):
        assert analyze_mod.is_valid_sha("") is False

    def test_short_sha_returns_false(self, analyze_mod):
        assert analyze_mod.is_valid_sha("abc123") is False

    def test_uppercase_hex_returns_false(self, analyze_mod):
        sha = "A" * 40
        assert analyze_mod.is_valid_sha(sha) is False

    def test_non_hex_returns_false(self, analyze_mod):
        sha = "g" * 40
        assert analyze_mod.is_valid_sha(sha) is False


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

    def test_missing_raw_json(self, analyze_mod, tmp_path):
        metrics = analyze_mod.load_raw_metrics(tmp_path)
        assert metrics is None

    def test_missing_field(self, analyze_mod, tmp_path):
        (tmp_path / "raw.json").write_text(json.dumps({"mean_ttft_ms": 100.0}))
        metrics = analyze_mod.load_raw_metrics(tmp_path)
        assert metrics is None

    def test_non_numeric_field(self, analyze_mod, tmp_path):
        (tmp_path / "raw.json").write_text(
            json.dumps(
                {
                    "mean_ttft_ms": "not_a_number",
                    "mean_tpot_ms": 50.0,
                    "output_throughput": 200.0,
                }
            )
        )
        metrics = analyze_mod.load_raw_metrics(tmp_path)
        assert metrics is None

    def test_invalid_json(self, analyze_mod, tmp_path):
        (tmp_path / "raw.json").write_text("{invalid json")
        metrics = analyze_mod.load_raw_metrics(tmp_path)
        assert metrics is None


# ---------------------------------------------------------------------------
# compute_median
# ---------------------------------------------------------------------------


class TestComputeMedian:
    def test_odd_count(self, analyze_mod):
        results = [
            {"metrics": {"mean_ttft_ms": 100.0}},
            {"metrics": {"mean_ttft_ms": 200.0}},
            {"metrics": {"mean_ttft_ms": 300.0}},
        ]
        median = analyze_mod.compute_median(results, "mean_ttft_ms")
        assert median == 200.0

    def test_even_count(self, analyze_mod):
        results = [
            {"metrics": {"mean_ttft_ms": 100.0}},
            {"metrics": {"mean_ttft_ms": 200.0}},
            {"metrics": {"mean_ttft_ms": 300.0}},
            {"metrics": {"mean_ttft_ms": 400.0}},
        ]
        median = analyze_mod.compute_median(results, "mean_ttft_ms")
        assert median == 250.0

    def test_empty_list(self, analyze_mod):
        median = analyze_mod.compute_median([], "mean_ttft_ms")
        assert median is None

    def test_single_value(self, analyze_mod):
        results = [{"metrics": {"mean_ttft_ms": 42.0}}]
        median = analyze_mod.compute_median(results, "mean_ttft_ms")
        assert median == 42.0


# ---------------------------------------------------------------------------
# compare_interval
# ---------------------------------------------------------------------------


class TestCompareInterval:
    def _setup_interval(
        self,
        tmp_path: Path,
        base_ttft: float,
        head_ttft: float,
        base_tpot: float = 50.0,
        head_tpot: float = 55.0,
        base_tput: float = 200.0,
        head_tput: float = 195.0,
        reps: int = 3,
    ) -> Path:
        """Set up a result directory with base and head reps."""
        interval = {
            "name": "test-interval",
            "base_commit": "aaaaaaaaaa",
            "head_commit": "bbbbbbbbbb",
            "workload": "test-workload",
            "reported_jump": "test jump",
        }
        for i in range(1, reps + 1):
            base_dir = (
                tmp_path / interval["base_commit"] / interval["workload"] / f"rep-{i}"
            )
            base_dir.mkdir(parents=True)
            _write_raw_json(
                base_dir, ttft=base_ttft, tpot=base_tpot, throughput=base_tput
            )
            _write_env_manifest(base_dir)
            (base_dir / ".completed").touch()

            head_dir = (
                tmp_path / interval["head_commit"] / interval["workload"] / f"rep-{i}"
            )
            head_dir.mkdir(parents=True)
            _write_raw_json(
                head_dir, ttft=head_ttft, tpot=head_tpot, throughput=head_tput
            )
            _write_env_manifest(head_dir)
            (head_dir / ".completed").touch()
        return tmp_path

    def test_not_reproducible(self, analyze_mod, tmp_path):
        """Small TTFT change (< 20%) should be not_reproducible."""
        self._setup_interval(tmp_path, base_ttft=100.0, head_ttft=110.0)
        interval = {
            "name": "test-interval",
            "base_commit": "aaaaaaaaaa",
            "head_commit": "bbbbbbbbbb",
            "workload": "test-workload",
            "reported_jump": "test jump",
        }
        result = analyze_mod.compare_interval(interval, tmp_path, None, None)
        assert result["verdict"] == "not_reproducible"
        assert result["base_reps"] == 3
        assert result["head_reps"] == 3

    def test_reproducible_regression_ttft(self, analyze_mod, tmp_path):
        """TTFT increase > 20% should be reproducible_regression."""
        self._setup_interval(tmp_path, base_ttft=100.0, head_ttft=150.0)
        interval = {
            "name": "test-interval",
            "base_commit": "aaaaaaaaaa",
            "head_commit": "bbbbbbbbbb",
            "workload": "test-workload",
            "reported_jump": "test jump",
        }
        result = analyze_mod.compare_interval(interval, tmp_path, None, None)
        assert result["verdict"] == "reproducible_regression"
        assert result["relative_changes"]["ttft"] > 0.20

    def test_reproducible_regression_tpot(self, analyze_mod, tmp_path):
        """TPOT increase > 20% should be reproducible_regression."""
        self._setup_interval(
            tmp_path, base_ttft=100.0, head_ttft=110.0, base_tpot=50.0, head_tpot=70.0
        )
        interval = {
            "name": "test-interval",
            "base_commit": "aaaaaaaaaa",
            "head_commit": "bbbbbbbbbb",
            "workload": "test-workload",
            "reported_jump": "test jump",
        }
        result = analyze_mod.compare_interval(interval, tmp_path, None, None)
        assert result["verdict"] == "reproducible_regression"
        assert result["relative_changes"]["tpot"] > 0.20

    def test_reproducible_regression_throughput(self, analyze_mod, tmp_path):
        """Throughput decrease > 10% should be reproducible_regression."""
        self._setup_interval(
            tmp_path, base_ttft=100.0, head_ttft=110.0, base_tput=200.0, head_tput=150.0
        )
        interval = {
            "name": "test-interval",
            "base_commit": "aaaaaaaaaa",
            "head_commit": "bbbbbbbbbb",
            "workload": "test-workload",
            "reported_jump": "test jump",
        }
        result = analyze_mod.compare_interval(interval, tmp_path, None, None)
        assert result["verdict"] == "reproducible_regression"
        assert result["relative_changes"]["throughput"] < -0.10

    def test_incomplete_evidence_no_base(self, analyze_mod, tmp_path):
        """Missing base results should be incomplete_evidence."""
        # Only set up head, not base
        head_dir = tmp_path / "bbbbbbbbbb" / "test-workload" / "rep-1"
        head_dir.mkdir(parents=True)
        _write_raw_json(head_dir, ttft=100.0)
        _write_env_manifest(head_dir)
        (head_dir / ".completed").touch()

        interval = {
            "name": "test-interval",
            "base_commit": "aaaaaaaaaa",
            "head_commit": "bbbbbbbbbb",
            "workload": "test-workload",
            "reported_jump": "test jump",
        }
        result = analyze_mod.compare_interval(interval, tmp_path, None, None)
        assert result["verdict"] == "incomplete_evidence"
        assert result["base_reps"] == 0

    def test_incomplete_evidence_no_head(self, analyze_mod, tmp_path):
        """Missing head results should be incomplete_evidence."""
        base_dir = tmp_path / "aaaaaaaaaa" / "test-workload" / "rep-1"
        base_dir.mkdir(parents=True)
        _write_raw_json(base_dir, ttft=100.0)
        _write_env_manifest(base_dir)
        (base_dir / ".completed").touch()

        interval = {
            "name": "test-interval",
            "base_commit": "aaaaaaaaaa",
            "head_commit": "bbbbbbbbbb",
            "workload": "test-workload",
            "reported_jump": "test jump",
        }
        result = analyze_mod.compare_interval(interval, tmp_path, None, None)
        assert result["verdict"] == "incomplete_evidence"
        assert result["head_reps"] == 0


# ---------------------------------------------------------------------------
# load_env_manifest
# ---------------------------------------------------------------------------


class TestLoadEnvManifest:
    def test_valid_manifest(self, analyze_mod, tmp_path):
        _write_env_manifest(tmp_path)
        manifest = analyze_mod.load_env_manifest(tmp_path)
        assert manifest is not None
        assert manifest["engine_commit_observed"] == "a" * 40
        assert manifest["plugin_commit_observed"] == "b" * 40

    def test_missing_manifest(self, analyze_mod, tmp_path):
        manifest = analyze_mod.load_env_manifest(tmp_path)
        assert manifest is None

    def test_invalid_json(self, analyze_mod, tmp_path):
        (tmp_path / "env-manifest.json").write_text("{invalid")
        manifest = analyze_mod.load_env_manifest(tmp_path)
        assert manifest is None


# ---------------------------------------------------------------------------
# INTERVALS configuration
# ---------------------------------------------------------------------------


class TestIntervalsConfig:
    def test_intervals_defined(self, analyze_mod):
        assert len(analyze_mod.INTERVALS) == 2

    def test_random_online_interval(self, analyze_mod):
        interval = next(
            i for i in analyze_mod.INTERVALS if i["name"] == "random-online"
        )
        assert interval["base_commit"] == "2206f1f7b7"
        assert interval["head_commit"] == "f273f9c5e2"
        assert interval["workload"] == "random-online"

    def test_agent_research_online_interval(self, analyze_mod):
        interval = next(
            i for i in analyze_mod.INTERVALS if i["name"] == "agent-research-online"
        )
        assert interval["base_commit"] == "7a63f81e86"
        assert interval["head_commit"] == "ec4847981f"
        assert interval["workload"] == "agent-research-online"

    def test_thresholds(self, analyze_mod):
        assert analyze_mod.TTFT_INCREASE_THRESHOLD == 0.20
        assert analyze_mod.TPOT_INCREASE_THRESHOLD == 0.20
        assert analyze_mod.THROUGHPUT_DECREASE_THRESHOLD == 0.10

    def test_metric_fields(self, analyze_mod):
        assert "mean_ttft_ms" in analyze_mod.METRIC_FIELDS
        assert "mean_tpot_ms" in analyze_mod.METRIC_FIELDS
        assert "output_throughput" in analyze_mod.METRIC_FIELDS
