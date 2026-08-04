"""Tests for issue #146 regression re-test analysis."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
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


def _write_valid_manifest(rep_dir: Path) -> None:
    """Write a valid env-manifest.json with matching patch file SHA-256.

    Used by collect_results tests that need reps to pass manifest validation.
    """
    patch_content = "=== PATCH ===\n+change\n"
    patch_file = rep_dir / "derived_patch_engine.diff"
    patch_file.write_text(patch_content)
    plugin_patch_file = rep_dir / "derived_patch_plugin.diff"
    plugin_patch_file.write_text(patch_content)

    sha = hashlib.sha256(patch_content.encode()).hexdigest()
    manifest = {
        "engine_commit_requested": "a" * 10,
        "engine_commit_observed": "a" * 40,
        "engine_patch_sha256": sha,
        "engine_patch_file": "derived_patch_engine.diff",
        "plugin_commit_requested": "b" * 10,
        "plugin_commit_observed": "b" * 40,
        "plugin_patch_sha256": sha,
        "plugin_patch_file": "derived_patch_plugin.diff",
    }
    (rep_dir / "env-manifest.json").write_text(json.dumps(manifest))


def _make_temp_git_repo(repo_dir: Path, *, short_sha_len: int = 10) -> tuple[Path, str]:
    """Create a temporary git repo with one commit.

    Returns (repo_path, real_full_sha).  Used by commit-object verification
    tests to verify that real SHAs pass and padded/fabricated SHAs are rejected.
    """
    repo_dir.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet"], cwd=repo_dir, check=True)
    subprocess.run(
        ["git", "-C", str(repo_dir), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_dir), "config", "user.name", "Test"],
        check=True,
    )
    (repo_dir / "README").write_text("test\n")
    subprocess.run(["git", "-C", str(repo_dir), "add", "README"], check=True)
    subprocess.run(
        ["git", "-C", str(repo_dir), "commit", "-m", "test", "--quiet"],
        capture_output=True,
        text=True,
        check=True,
    )
    full_sha = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert len(full_sha) == 40
    return repo_dir, full_sha


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
                # .completed marker + valid manifest required for collection
                (d / ".completed").touch()
                _write_valid_manifest(d)

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
            (d / ".completed").touch()
            _write_valid_manifest(d)

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
        (d / ".completed").touch()
        _write_valid_manifest(d)
        results = analyze_mod.collect_results(tmp_path)
        assert results["sonnet-throughput"]["2206f1f7b7"] == []

    def test_stale_rep_without_completed_marker_skipped(self, analyze_mod, tmp_path):
        """Reverse test: a rep with raw.json but NO .completed marker must be
        skipped by collect_results.

        This prevents stale results from a previous failed rerun from being
        consumed by the analysis.  Per reviewer round 2: '请在开始 rep 前清空
        最终目录，或使用本次 run 的独立目录和完成标记'.
        """
        d = tmp_path / "2206f1f7b7" / "sonnet-throughput" / "rep-1"
        d.mkdir(parents=True)
        # Stale raw.json present but NO .completed marker
        (d / "raw.json").write_text(json.dumps({"tokens_per_second": 9999.0}))
        _write_valid_manifest(d)
        # Deliberately do NOT create .completed

        results = analyze_mod.collect_results(tmp_path)
        assert results["sonnet-throughput"]["2206f1f7b7"] == []

    def test_mixed_completed_and_incomplete_reps(self, analyze_mod, tmp_path):
        """Only completed reps should be collected; incomplete ones skipped."""
        for rep in [1, 2, 3]:
            d = tmp_path / "2206f1f7b7" / "sonnet-throughput" / f"rep-{rep}"
            d.mkdir(parents=True)
            (d / "raw.json").write_text(
                json.dumps({"tokens_per_second": 1000.0 + rep * 10})
            )
            _write_valid_manifest(d)
            # rep-2 is "incomplete" (failed, no .completed)
            if rep != 2:
                (d / ".completed").touch()

        results = analyze_mod.collect_results(tmp_path)
        # Only 2 reps collected (rep-1 and rep-3), rep-2 skipped
        assert len(results["sonnet-throughput"]["2206f1f7b7"]) == 2
        assert 1010.0 in results["sonnet-throughput"]["2206f1f7b7"]
        assert 1030.0 in results["sonnet-throughput"]["2206f1f7b7"]


# ---------------------------------------------------------------------------
# validate_env_manifest (issue #146 reviewer round 2: SHA-256 patch identity)
# ---------------------------------------------------------------------------


class TestValidateEnvManifest:
    """Tests for validate_env_manifest — verifies that the env-manifest.json
    has complete provenance with SHA-256 (not MD5) patch identity and
    references saved patch files on disk.

    Per reviewer round 2: 'patch identity 只对 git diff HEAD 做 MD5，捕获不到
    ensure_build_info 生成的未跟踪文件，也没有保存可复现的 patch 内容；请把
    tracked/untracked 派生修改完整落盘并用 SHA-256 绑定'.
    """

    def test_valid_manifest_with_sha256_and_patch_files(self, analyze_mod, tmp_path):
        """A manifest with SHA-256 patch identity and existing patch files passes."""
        rep_dir = tmp_path / "rep-1"
        rep_dir.mkdir()
        patch_content = "=== TRACKED DIFF ===\n+some change\n"
        (rep_dir / "derived_patch_engine.diff").write_text(patch_content)
        (rep_dir / "derived_patch_plugin.diff").write_text(patch_content)

        import hashlib

        sha = hashlib.sha256(patch_content.encode()).hexdigest()
        manifest = {
            "engine_commit_requested": "a" * 10,
            "engine_commit_observed": "a" * 40,
            "engine_patch_sha256": sha,
            "engine_patch_file": "derived_patch_engine.diff",
            "plugin_commit_requested": "b" * 10,
            "plugin_commit_observed": "b" * 40,
            "plugin_patch_sha256": sha,
            "plugin_patch_file": "derived_patch_plugin.diff",
        }
        (rep_dir / "env-manifest.json").write_text(json.dumps(manifest))

        valid, reason = analyze_mod.validate_env_manifest(rep_dir)
        assert valid, f"Expected valid, got: {reason}"

    def test_valid_manifest_with_clean_repos(self, analyze_mod, tmp_path):
        """A manifest with 'clean' patch identity (no modifications) passes."""
        rep_dir = tmp_path / "rep-1"
        rep_dir.mkdir()
        manifest = {
            "engine_commit_requested": "a" * 10,
            "engine_commit_observed": "a" * 40,
            "engine_patch_sha256": "clean",
            "plugin_commit_requested": "b" * 10,
            "plugin_commit_observed": "b" * 40,
            "plugin_patch_sha256": "clean",
        }
        (rep_dir / "env-manifest.json").write_text(json.dumps(manifest))

        valid, reason = analyze_mod.validate_env_manifest(rep_dir)
        assert valid, f"Expected valid, got: {reason}"

    def test_rejects_old_md5_only_manifest(self, analyze_mod, tmp_path):
        """Reverse test: a manifest with engine_patch_identity (MD5) but no
        engine_patch_sha256 must be rejected."""
        rep_dir = tmp_path / "rep-1"
        rep_dir.mkdir()
        manifest = {
            "engine_commit_observed": "a" * 40,
            "engine_patch_identity": "abc123def456",  # pragma: allowlist secret
            "plugin_commit_observed": "b" * 40,
            "plugin_patch_identity": "xyz789abc012",  # pragma: allowlist secret
        }
        (rep_dir / "env-manifest.json").write_text(json.dumps(manifest))

        valid, reason = analyze_mod.validate_env_manifest(rep_dir)
        assert not valid
        assert "deprecated" in reason or "sha256" in reason.lower()

    def test_rejects_missing_sha256_field(self, analyze_mod, tmp_path):
        """Reverse test: a manifest without engine_patch_sha256 is rejected."""
        rep_dir = tmp_path / "rep-1"
        rep_dir.mkdir()
        manifest = {
            "engine_commit_observed": "a" * 40,
            # engine_patch_sha256 deliberately missing
            "plugin_commit_observed": "b" * 40,
            "plugin_patch_sha256": "clean",
        }
        (rep_dir / "env-manifest.json").write_text(json.dumps(manifest))

        valid, reason = analyze_mod.validate_env_manifest(rep_dir)
        assert not valid
        assert "engine_patch_sha256" in reason

    def test_rejects_invalid_sha256_length(self, analyze_mod, tmp_path):
        """Reverse test: a SHA-256 that is not 64 hex chars is rejected."""
        rep_dir = tmp_path / "rep-1"
        rep_dir.mkdir()
        manifest = {
            "engine_commit_observed": "a" * 40,
            "engine_patch_sha256": "tooshort",  # not 64 chars
            "plugin_commit_observed": "b" * 40,
            "plugin_patch_sha256": "clean",
        }
        (rep_dir / "env-manifest.json").write_text(json.dumps(manifest))

        valid, reason = analyze_mod.validate_env_manifest(rep_dir)
        assert not valid
        assert "not valid 64-hex SHA-256" in reason

    def test_rejects_non_hex_sha256(self, analyze_mod, tmp_path):
        """Reverse test: 64 chars but with non-hex characters is rejected.

        Per reviewer round 3: 'validator 还需要校验 64 位十六进制' — must
        verify hex format with regex, not just length==64.
        """
        rep_dir = tmp_path / "rep-1"
        rep_dir.mkdir()
        manifest = {
            "engine_commit_observed": "a" * 40,
            # 64 chars but contains 'g' and 'z' (not hex)
            "engine_patch_sha256": "g" * 32 + "z" * 32,
            "plugin_commit_observed": "b" * 40,
            "plugin_patch_sha256": "clean",
        }
        (rep_dir / "env-manifest.json").write_text(json.dumps(manifest))

        valid, reason = analyze_mod.validate_env_manifest(rep_dir)
        assert not valid
        assert "not valid 64-hex SHA-256" in reason

    # ------------------------------------------------------------------
    # Round 4: reject fabricated padded observed commit SHAs
    # ------------------------------------------------------------------

    def test_rejects_padded_engine_commit_observed(self, analyze_mod, tmp_path):
        """Reverse test: engine_commit_observed is a short SHA padded with
        zeros (e.g. '2206f1f7b7' + 30 zeros) — must be rejected.

        Per reviewer round 5: 'test_rejects_padded_engine_commit_observed 现在对
        2206f1f7b7 后补 30 个零的值断言 valid...40 位十六进制和 requested prefix
        只能做格式检查，不能证明对象存在。请让 observed SHA 能在对应仓库解析到
        commit...并把这个测试改成确实拒绝该补零值'.

        The test creates a real git repo with a real commit, then sets
        engine_commit_observed to a padded SHA (short prefix + 30 zeros).
        The validator must reject this because the padded SHA does not resolve
        to a commit object in the repo (verified via ``git cat-file -t``).
        """
        # Create a real git repo with one commit.
        repo_dir, real_sha = _make_temp_git_repo(tmp_path / "vllm-hust")
        short_sha = real_sha[:10]

        # Build a padded SHA: short prefix + 30 zeros.  This passes format
        # (40-hex) and prefix (startswith short_sha) checks but does NOT
        # resolve to a real commit object.
        padded_sha = short_sha + "0" * 30
        assert len(padded_sha) == 40
        assert padded_sha.startswith(short_sha)
        assert padded_sha != real_sha

        rep_dir = tmp_path / "rep-1"
        rep_dir.mkdir()
        manifest = {
            "engine_commit_requested": short_sha,
            "engine_commit_observed": padded_sha,
            "engine_patch_sha256": "clean",
            "plugin_commit_requested": "b" * 10,
            "plugin_commit_observed": "b" * 40,
            "plugin_patch_sha256": "clean",
        }
        (rep_dir / "env-manifest.json").write_text(json.dumps(manifest))

        # With repo_paths provided, the validator must verify the observed SHA
        # resolves to a real commit object.  The padded SHA must be rejected.
        valid, reason = analyze_mod.validate_env_manifest(
            rep_dir, repo_paths={"engine": repo_dir}
        )
        assert not valid, (
            f"Padded SHA {padded_sha!r} must be rejected when repo is "
            f"available, but validator returned valid.  reason={reason!r}"
        )
        assert "does not resolve to a commit object" in reason

    def test_real_engine_commit_observed_passes_verification(
        self, analyze_mod, tmp_path
    ):
        """Positive test: a real commit SHA that resolves in the repo passes.

        Per reviewer round 5: '请让 observed SHA 能在对应仓库解析到 commit'.
        """
        repo_dir, real_sha = _make_temp_git_repo(tmp_path / "vllm-hust")
        short_sha = real_sha[:10]

        rep_dir = tmp_path / "rep-1"
        rep_dir.mkdir()
        manifest = {
            "engine_commit_requested": short_sha,
            "engine_commit_observed": real_sha,
            "engine_patch_sha256": "clean",
            "plugin_commit_requested": "b" * 10,
            "plugin_commit_observed": "b" * 40,
            "plugin_patch_sha256": "clean",
        }
        (rep_dir / "env-manifest.json").write_text(json.dumps(manifest))

        valid, reason = analyze_mod.validate_env_manifest(
            rep_dir, repo_paths={"engine": repo_dir}
        )
        assert valid, f"Real commit SHA should pass verification: {reason}"

    def test_padded_sha_passes_without_repo(self, analyze_mod, tmp_path):
        """Without repo_paths, the validator can only do format+prefix checks.

        A padded SHA that passes format+prefix will be accepted when no repo
        is available for commit-object verification (fail-open for CI
        environments without repo access).  This documents the limitation.
        """
        rep_dir = tmp_path / "rep-1"
        rep_dir.mkdir()
        manifest = {
            "engine_commit_requested": "2206f1f7b7",
            "engine_commit_observed": "2206f1f7b7" + "0" * 30,
            "engine_patch_sha256": "clean",
            "plugin_commit_requested": "b" * 10,
            "plugin_commit_observed": "b" * 40,
            "plugin_patch_sha256": "clean",
        }
        (rep_dir / "env-manifest.json").write_text(json.dumps(manifest))

        # No repo_paths — only format + prefix checks are done.
        valid, reason = analyze_mod.validate_env_manifest(rep_dir)
        assert valid, f"Without repo access, format+prefix checks should pass: {reason}"

    def test_rejects_padded_engine_commit_observed_mismatched_prefix(
        self, analyze_mod, tmp_path
    ):
        """Reverse test: engine_commit_observed doesn't start with the
        requested short SHA — rejected as suspected padded SHA.

        Per reviewer round 4: '至少拒绝这种补零值'.
        """
        rep_dir = tmp_path / "rep-1"
        rep_dir.mkdir()
        manifest = {
            "engine_commit_requested": "2206f1f7b7",
            # Padded with zeros but requested was a different prefix.
            # Use a different valid 40-hex SHA that doesn't start with the
            # requested short SHA.
            "engine_commit_observed": "f" * 40,  # pragma: allowlist secret
            "engine_patch_sha256": "clean",
            "plugin_commit_requested": "b" * 10,
            "plugin_commit_observed": "b" * 40,
            "plugin_patch_sha256": "clean",
        }
        (rep_dir / "env-manifest.json").write_text(json.dumps(manifest))

        valid, reason = analyze_mod.validate_env_manifest(rep_dir)
        assert not valid
        assert "does not start with" in reason or "padded" in reason.lower()

    def test_rejects_missing_engine_commit_observed(self, analyze_mod, tmp_path):
        """Reverse test: manifest without engine_commit_observed — rejected."""
        rep_dir = tmp_path / "rep-1"
        rep_dir.mkdir()
        manifest = {
            # engine_commit_observed deliberately missing
            "engine_patch_sha256": "clean",
            "plugin_commit_observed": "b" * 40,
            "plugin_patch_sha256": "clean",
        }
        (rep_dir / "env-manifest.json").write_text(json.dumps(manifest))

        valid, reason = analyze_mod.validate_env_manifest(rep_dir)
        assert not valid
        assert "engine_commit_observed" in reason

    def test_rejects_short_engine_commit_observed(self, analyze_mod, tmp_path):
        """Reverse test: engine_commit_observed is a 10-char short SHA — rejected."""
        rep_dir = tmp_path / "rep-1"
        rep_dir.mkdir()
        manifest = {
            "engine_commit_requested": "2206f1f7b7",
            "engine_commit_observed": "2206f1f7b7",  # 10 chars, not 40
            "engine_patch_sha256": "clean",
            "plugin_commit_observed": "b" * 40,
            "plugin_patch_sha256": "clean",
        }
        (rep_dir / "env-manifest.json").write_text(json.dumps(manifest))

        valid, reason = analyze_mod.validate_env_manifest(rep_dir)
        assert not valid
        assert "engine_commit_observed" in reason
        assert "40-hex SHA" in reason

    def test_rejects_missing_patch_file_on_disk(self, analyze_mod, tmp_path):
        """Reverse test: a manifest referencing a patch file that doesn't
        exist on disk is rejected."""
        rep_dir = tmp_path / "rep-1"
        rep_dir.mkdir()
        manifest = {
            "engine_commit_observed": "a" * 40,
            "engine_patch_sha256": "a" * 64,
            "engine_patch_file": "derived_patch_engine.diff",
            # But the file is NOT created on disk
            "plugin_commit_observed": "b" * 40,
            "plugin_patch_sha256": "clean",
        }
        (rep_dir / "env-manifest.json").write_text(json.dumps(manifest))

        valid, reason = analyze_mod.validate_env_manifest(rep_dir)
        assert not valid
        assert "patch file not found" in reason

    def test_rejects_tampered_patch_file(self, analyze_mod, tmp_path):
        """Reverse test: patch file content modified after manifest creation.

        Per reviewer round 3: '重新计算 patch 文件 SHA-256，而不是只看长度和
        文件存在' — the validator must recompute the SHA-256 of the patch file
        and compare with the manifest value to detect tampering.
        """
        import hashlib

        rep_dir = tmp_path / "rep-1"
        rep_dir.mkdir()
        original_content = "=== ORIGINAL PATCH ===\n+original line\n"
        tampered_content = "=== TAMPERED PATCH ===\n+malicious line\n"
        (rep_dir / "derived_patch_engine.diff").write_text(tampered_content)

        # SHA-256 of the ORIGINAL content, but file now has TAMPERED content
        original_sha = hashlib.sha256(original_content.encode()).hexdigest()
        manifest = {
            "engine_commit_observed": "a" * 40,
            "engine_patch_sha256": original_sha,
            "engine_patch_file": "derived_patch_engine.diff",
            "plugin_commit_observed": "b" * 40,
            "plugin_patch_sha256": "clean",
        }
        (rep_dir / "env-manifest.json").write_text(json.dumps(manifest))

        valid, reason = analyze_mod.validate_env_manifest(rep_dir)
        assert not valid
        assert "SHA-256 mismatch" in reason
        assert "tampered" in reason

    def test_rejects_missing_manifest(self, analyze_mod, tmp_path):
        """Reverse test: a rep directory without env-manifest.json is rejected."""
        rep_dir = tmp_path / "rep-1"
        rep_dir.mkdir()
        # No env-manifest.json created

        valid, reason = analyze_mod.validate_env_manifest(rep_dir)
        assert not valid
        assert "missing" in reason

    def test_rejects_corrupt_manifest(self, analyze_mod, tmp_path):
        """Reverse test: a corrupt env-manifest.json is rejected."""
        rep_dir = tmp_path / "rep-1"
        rep_dir.mkdir()
        (rep_dir / "env-manifest.json").write_text("not valid json")

        valid, reason = analyze_mod.validate_env_manifest(rep_dir)
        assert not valid
        assert "corrupt" in reason


# ---------------------------------------------------------------------------
# collect_results manifest enforcement (issue #146 reviewer round 3)
# ---------------------------------------------------------------------------


class TestCollectResultsManifestEnforcement:
    """Tests that collect_results enforces validate_env_manifest.

    Per reviewer round 3: 'collect_results 只检查 marker 和 raw.json，从未调用
    新增的 validate_env_manifest，因此当前仓库里仍是 MD5 manifest、没有 derived
    patch 文件的旧结果也会继续参与结论。请...收集时强制 validate_env_manifest'.
    """

    def test_collect_skips_rep_with_missing_manifest(self, analyze_mod, tmp_path):
        """collect_results must skip reps where env-manifest.json is missing,
        even if .completed and raw.json exist.

        Per reviewer round 3: '后者失败时 marker 会保留' — but now the bash
        script writes .completed AFTER manifest.  Still, collect_results must
        enforce manifest validation independently.
        """
        d = tmp_path / "2206f1f7b7" / "sonnet-throughput" / "rep-1"
        d.mkdir(parents=True)
        (d / "raw.json").write_text(json.dumps({"tokens_per_second": 1000.0}))
        (d / ".completed").touch()
        # Deliberately NO env-manifest.json

        results = analyze_mod.collect_results(tmp_path)
        assert results["sonnet-throughput"]["2206f1f7b7"] == []

    def test_collect_skips_rep_with_old_md5_manifest(self, analyze_mod, tmp_path):
        """collect_results must skip reps with old MD5-only manifests.

        Per reviewer round 3: '当前仓库里仍是 MD5 manifest、没有 derived patch
        文件的旧结果也会继续参与结论' — must be rejected.
        """
        d = tmp_path / "2206f1f7b7" / "sonnet-throughput" / "rep-1"
        d.mkdir(parents=True)
        (d / "raw.json").write_text(json.dumps({"tokens_per_second": 1000.0}))
        (d / ".completed").touch()
        # Old MD5 manifest (no sha256 fields)
        old_manifest = {
            "engine_commit_observed": "a" * 40,
            "engine_patch_identity": "abc123def456",  # pragma: allowlist secret
            "plugin_commit_observed": "b" * 40,
            "plugin_patch_identity": "xyz789abc012",  # pragma: allowlist secret
        }
        (d / "env-manifest.json").write_text(json.dumps(old_manifest))

        results = analyze_mod.collect_results(tmp_path)
        assert results["sonnet-throughput"]["2206f1f7b7"] == []

    def test_collect_skips_rep_with_tampered_patch(self, analyze_mod, tmp_path):
        """collect_results must skip reps where the patch file was tampered
        (SHA-256 mismatch between manifest and file content).

        Per reviewer round 3: 'patch 被篡改的反向测试'.
        """
        import hashlib

        d = tmp_path / "2206f1f7b7" / "sonnet-throughput" / "rep-1"
        d.mkdir(parents=True)
        (d / "raw.json").write_text(json.dumps({"tokens_per_second": 1000.0}))
        (d / ".completed").touch()

        # Write a patch file with tampered content
        tampered_content = "=== TAMPERED ===\n"
        (d / "derived_patch_engine.diff").write_text(tampered_content)
        (d / "derived_patch_plugin.diff").write_text(tampered_content)

        # But manifest records SHA-256 of different (original) content
        original_sha = hashlib.sha256(b"=== ORIGINAL ===\n").hexdigest()
        manifest = {
            "engine_commit_observed": "a" * 40,
            "engine_patch_sha256": original_sha,
            "engine_patch_file": "derived_patch_engine.diff",
            "plugin_commit_observed": "b" * 40,
            "plugin_patch_sha256": original_sha,
            "plugin_patch_file": "derived_patch_plugin.diff",
        }
        (d / "env-manifest.json").write_text(json.dumps(manifest))

        results = analyze_mod.collect_results(tmp_path)
        assert results["sonnet-throughput"]["2206f1f7b7"] == []

    def test_collect_includes_rep_with_valid_manifest(self, analyze_mod, tmp_path):
        """collect_results must include reps with valid manifest + .completed."""
        d = tmp_path / "2206f1f7b7" / "sonnet-throughput" / "rep-1"
        d.mkdir(parents=True)
        (d / "raw.json").write_text(json.dumps({"tokens_per_second": 1000.0}))
        (d / ".completed").touch()
        _write_valid_manifest(d)

        results = analyze_mod.collect_results(tmp_path)
        assert len(results["sonnet-throughput"]["2206f1f7b7"]) == 1
        assert results["sonnet-throughput"]["2206f1f7b7"][0] == 1000.0

    def test_collect_mixed_valid_and_invalid_manifests(self, analyze_mod, tmp_path):
        """Only reps with valid manifests are collected; invalid ones skipped."""
        # rep-1: valid manifest
        d1 = tmp_path / "2206f1f7b7" / "sonnet-throughput" / "rep-1"
        d1.mkdir(parents=True)
        (d1 / "raw.json").write_text(json.dumps({"tokens_per_second": 1000.0}))
        (d1 / ".completed").touch()
        _write_valid_manifest(d1)

        # rep-2: missing manifest (simulates manifest write failure)
        d2 = tmp_path / "2206f1f7b7" / "sonnet-throughput" / "rep-2"
        d2.mkdir(parents=True)
        (d2 / "raw.json").write_text(json.dumps({"tokens_per_second": 2000.0}))
        (d2 / ".completed").touch()
        # No env-manifest.json

        # rep-3: valid manifest
        d3 = tmp_path / "2206f1f7b7" / "sonnet-throughput" / "rep-3"
        d3.mkdir(parents=True)
        (d3 / "raw.json").write_text(json.dumps({"tokens_per_second": 3000.0}))
        (d3 / ".completed").touch()
        _write_valid_manifest(d3)

        results = analyze_mod.collect_results(tmp_path)
        # Only rep-1 and rep-3 collected (rep-2 skipped due to missing manifest)
        assert len(results["sonnet-throughput"]["2206f1f7b7"]) == 2
        assert 1000.0 in results["sonnet-throughput"]["2206f1f7b7"]
        assert 3000.0 in results["sonnet-throughput"]["2206f1f7b7"]
