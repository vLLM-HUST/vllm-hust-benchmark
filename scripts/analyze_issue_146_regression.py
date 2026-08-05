"""Analyze issue #146 regression re-test results.

Reads raw benchmark results from the re-test directory, computes medians,
and determines whether the two suspected regressions are reproducible.

Issue #146 acceptance criteria:
- sonnet-throughput: 2206f1f7b7 -> 7a63f81e86 should converge within 10%
- random-latency: 2206f1f7b7 -> 83cf83ff20 should not stably exceed 20% higher
- If noise/environment drift: clean corresponding leaderboard points

Usage:
    python analyze_issue_146_regression.py --result-dir /data/issue146-retest-results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# SHA-256 is exactly 64 lowercase hex characters.
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
# Full git commit SHA is exactly 40 lowercase hex characters.
_FULL_HEX_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def verify_commit_in_repo(repo_path: Path | None, sha: str) -> bool:
    """Verify that ``sha`` resolves to a real commit object in ``repo_path``.

    Uses ``git cat-file -t <sha>`` to check that the SHA is a valid commit
    object in the repository.  This rejects fabricated/padded SHAs that pass
    format checks but don't correspond to any real commit.

    Per reviewer round 5: '40 位十六进制和 requested prefix 只能做格式检查，
    不能证明对象存在。请让 observed SHA 能在对应仓库解析到 commit'.

    Per reviewer round 6: 'commit-object 校验仍然是可选且 fail-open。
    verify_commit_in_repo() 在 repo 不存在或 git cat-file 异常时返回 True'
    — must be fail-closed: any missing repo or git error means the manifest
    is invalid (we cannot prove the commit object exists).

    Returns True only if the repo exists and the SHA resolves to a commit.
    Returns False if repo_path is None, doesn't exist, or git errors out.
    """
    if repo_path is None or not repo_path.is_dir():
        # Cannot verify without repo access — fail-closed (reject).
        return False
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "cat-file", "-t", sha],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and result.stdout.strip() == "commit"
    except (subprocess.SubprocessError, OSError):
        # If git fails, fail-closed (reject) — we cannot prove the object
        # exists, so the manifest must be treated as invalid.
        return False


# Commits under investigation (issue #146)
ENGINE_COMMITS = ["2206f1f7b7", "7a63f81e86", "83cf83ff20"]

# Workloads with suspected regressions
WORKLOADS = ["sonnet-throughput", "random-latency"]

# Acceptance thresholds from issue #146
SONNET_THROUGHPUT_THRESHOLD_PCT = 10.0
RANDOM_LATENCY_THRESHOLD_PCT = 20.0


def _load_raw_json(path: Path) -> dict[str, Any] | None:
    """Load a raw.json benchmark output file."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _extract_metric(raw: dict[str, Any], workload: str) -> float | None:
    """Extract the primary metric from a raw benchmark output.

    - sonnet-throughput: ``tokens_per_second`` (tokens/s, higher is better)
    - random-latency: ``avg_latency`` converted to ms (lower is better)

    The field names mirror the actual vllm bench output format (verified
    against vllm/benchmarks/latency.py and the throughput subcommand).
    ``leaderboard_export._derive_metrics_from_benchmark_result`` uses the
    same fallback chain when converting raw output to leaderboard metrics.
    """
    if workload == "sonnet-throughput":
        # vllm bench throughput output:
        #   {"tokens_per_second": ..., "requests_per_second": ...}
        # The leaderboard stores this as throughput_tps.
        val = (
            raw.get("tokens_per_second")
            or raw.get("throughput_tps")
            or raw.get("tokens/s")
        )
        if val is None and isinstance(raw.get("throughput"), dict):
            val = raw["throughput"].get("tokens/s")
        return float(val) if val is not None else None
    elif workload == "random-latency":
        # vllm bench latency output:
        #   {"avg_latency": <seconds>, "latencies": [...], "percentiles": {...}}
        # The leaderboard stores this as ttft_ms (avg_latency * 1000).
        # Return in ms for consistency with the leaderboard convention.
        val = raw.get("mean_ttft_ms")
        if val is not None:
            return float(val)
        val = raw.get("ttft_ms")
        if val is not None:
            return float(val)
        avg_latency_s = raw.get("avg_latency")
        if avg_latency_s is not None:
            return float(avg_latency_s) * 1000.0
        # Fallback for other output formats
        val = raw.get("p50")
        if val is not None:
            return float(val)
        return None
    return None


def collect_results(
    result_dir: Path,
    *,
    repo_paths: dict[str, Path] | None = None,
) -> dict[str, dict[str, list[float]]]:
    """Collect all benchmark results from the re-test directory.

    Only collects results from reps that:
    1. Have a ``.completed`` marker file (benchmark + manifest succeeded).
    2. Pass ``validate_env_manifest`` (provenance is complete and untampered).

    Per reviewer round 6: 'validate_env_manifest() 在 repo_paths=None 时跳过
    校验' — repo_paths is now REQUIRED (fail-closed).  The default consumption
    path must require both engine and plugin repo paths so that observed SHAs
    are verified to resolve to real commit objects.

    Returns: {workload: {commit: [metric1, metric2, ...]}}
    """
    results: dict[str, dict[str, list[float]]] = {}
    for workload in WORKLOADS:
        results[workload] = {}
        for commit in ENGINE_COMMITS:
            commit_dir = result_dir / commit / workload
            if not commit_dir.is_dir():
                results[workload][commit] = []
                continue
            metrics: list[float] = []
            for rep_dir in sorted(commit_dir.iterdir()):
                if not rep_dir.is_dir() or not rep_dir.name.startswith("rep-"):
                    continue
                # Skip incomplete reps — no .completed marker means the
                # benchmark failed or was interrupted, and any raw.json
                # present is stale from a previous run.
                if not (rep_dir / ".completed").is_file():
                    continue
                # Enforce manifest validation before consuming results.
                # Per reviewer round 3: this must be fail-closed — reps with
                # missing, corrupt, or tampered manifests are skipped.
                manifest_valid, manifest_reason = validate_env_manifest(
                    rep_dir, repo_paths=repo_paths
                )
                if not manifest_valid:
                    print(
                        f"  WARNING: skipping {rep_dir.name}: {manifest_reason}",
                        file=sys.stderr,
                    )
                    continue
                raw = _load_raw_json(rep_dir / "raw.json")
                if raw is None:
                    continue
                metric = _extract_metric(raw, workload)
                if metric is not None:
                    metrics.append(metric)
            results[workload][commit] = metrics
    return results


def validate_env_manifest(
    rep_dir: Path,
    *,
    repo_paths: dict[str, Path] | None = None,
) -> tuple[bool, str]:
    """Validate that an env-manifest.json has complete and untampered provenance.

    Per reviewer feedback (round 6):
    - ``repo_paths`` is now REQUIRED (fail-closed).  When None, the manifest
      is rejected because we cannot prove observed SHAs resolve to real
      commits.  Both 'engine' and 'plugin' entries must be present and point
      to existing git repos.

    Per reviewer feedback (round 5):
    - Must verify observed SHA resolves to a real commit object in the
      corresponding repo via ``git cat-file -t <sha>``.  Format + prefix
      checks alone cannot prove the object exists.

    Per reviewer feedback (round 4):
    - Must reject fabricated manifests where observed commit SHAs are padded
      short SHAs (e.g. "2206f1f7b7" + 30 zeros).  Both engine_commit_observed
      and plugin_commit_observed must be real 40-char hex SHAs.

    Per reviewer feedback (round 3):
    - Must verify SHA-256 is exactly 64 lowercase hex (regex, not just length).
    - Must recompute SHA-256 of the patch file on disk and compare with the
      manifest value — reject if they differ (patch tampered).
    - collect_results must call this validator and skip reps that fail.

    Returns (valid, reason).  ``reason`` is empty when valid.
    """
    manifest_path = rep_dir / "env-manifest.json"
    if not manifest_path.is_file():
        return False, "env-manifest.json missing"

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"env-manifest.json corrupt: {exc}"

    # Reject old MD5-only manifests (engine_patch_identity without sha256)
    if "engine_patch_identity" in manifest and "engine_patch_sha256" not in manifest:
        return False, (
            "manifest uses deprecated engine_patch_identity (MD5); "
            "must use engine_patch_sha256 with saved patch file"
        )
    if "plugin_patch_identity" in manifest and "plugin_patch_sha256" not in manifest:
        return False, (
            "manifest uses deprecated plugin_patch_identity (MD5); "
            "must use plugin_patch_sha256 with saved patch file"
        )

    # Per reviewer round 4: validate observed commit SHAs are real 40-hex,
    # not padded short SHAs (e.g. "2206f1f7b7" + 30 zeros).
    # We check:
    # 1. Must be exactly 40 lowercase hex chars.
    # 2. Must start with the requested short SHA (proving the checkout matched).
    engine_requested = manifest.get("engine_commit_requested", "")
    plugin_requested = manifest.get("plugin_commit_requested", "")
    engine_observed = manifest.get("engine_commit_observed")
    plugin_observed = manifest.get("plugin_commit_observed")
    if not engine_observed or not isinstance(engine_observed, str):
        return False, "engine_commit_observed missing or not a string"
    if not plugin_observed or not isinstance(plugin_observed, str):
        return False, "plugin_commit_observed missing or not a string"
    if not _FULL_HEX_SHA_RE.match(engine_observed):
        return False, (
            f"engine_commit_observed not a valid 40-hex SHA: "
            f"{engine_observed!r} (padded short SHAs are rejected)"
        )
    if not _FULL_HEX_SHA_RE.match(plugin_observed):
        return False, (
            f"plugin_commit_observed not a valid 40-hex SHA: "
            f"{plugin_observed!r} (padded short SHAs are rejected)"
        )
    # Verify observed SHA starts with the requested short SHA — this catches
    # padded values where the short SHA was extended with zeros instead of
    # the real full SHA.  Per reviewer round 4: '至少拒绝这种补零值'.
    if (
        isinstance(engine_requested, str)
        and engine_requested
        and not engine_observed.startswith(engine_requested)
    ):
        return False, (
            f"engine_commit_observed {engine_observed!r} does not start with "
            f"requested short SHA {engine_requested!r} (padded SHA suspected)"
        )
    if (
        isinstance(plugin_requested, str)
        and plugin_requested
        and not plugin_observed.startswith(plugin_requested)
    ):
        return False, (
            f"plugin_commit_observed {plugin_observed!r} does not start with "
            f"requested short SHA {plugin_requested!r} (padded SHA suspected)"
        )

    # Per reviewer round 6: 'validate_env_manifest() 在 repo_paths=None 时跳过
    # 校验，因此默认消费路径仍会接受格式正确但不存在的 SHA。用于生成结论
    # 时必须要求两个 repo 路径都存在且对应 SHA 可解析，任何缺失或 git 错误
    # 都应判 manifest invalid' — this is now fail-closed: repo_paths must
    # be provided with both engine and plugin repos, and both observed SHAs
    # must resolve to real commit objects.
    if repo_paths is None:
        return False, (
            "repo_paths is required for commit-object verification "
            "(fail-closed: cannot prove observed SHAs resolve to commits)"
        )
    engine_repo = repo_paths.get("engine")
    if engine_repo is None:
        return False, "repo_paths missing 'engine' entry (fail-closed)"
    plugin_repo = repo_paths.get("plugin")
    if plugin_repo is None:
        return False, "repo_paths missing 'plugin' entry (fail-closed)"
    if not verify_commit_in_repo(engine_repo, engine_observed):
        return False, (
            f"engine_commit_observed {engine_observed!r} does not resolve "
            f"to a commit object in {engine_repo} (fail-closed: missing repo, "
            f"git error, or fabricated SHA)"
        )
    if not verify_commit_in_repo(plugin_repo, plugin_observed):
        return False, (
            f"plugin_commit_observed {plugin_observed!r} does not resolve "
            f"to a commit object in {plugin_repo} (fail-closed: missing repo, "
            f"git error, or fabricated SHA)"
        )

    # Verify SHA-256 fields exist and are strings
    engine_sha = manifest.get("engine_patch_sha256")
    plugin_sha = manifest.get("plugin_patch_sha256")
    if not engine_sha or not isinstance(engine_sha, str):
        return False, "engine_patch_sha256 missing or not a string"
    if not plugin_sha or not isinstance(plugin_sha, str):
        return False, "plugin_patch_sha256 missing or not a string"

    # Validate SHA-256 format: exactly 64 lowercase hex chars.
    # Per reviewer round 3: "validator 还需要校验 64 位十六进制" — must use
    # regex, not just check length==64.
    if engine_sha != "clean":
        if not _SHA256_RE.match(engine_sha):
            return False, (
                f"engine_patch_sha256 not valid 64-hex SHA-256: {engine_sha!r}"
            )
    if plugin_sha != "clean":
        if not _SHA256_RE.match(plugin_sha):
            return False, (
                f"plugin_patch_sha256 not valid 64-hex SHA-256: {plugin_sha!r}"
            )

    # Recompute SHA-256 of the patch file and compare with manifest value.
    # Per reviewer round 3: "重新计算 patch 文件 SHA-256，而不是只看长度和文件存在"
    # — this detects patch tampering after manifest creation.
    if engine_sha != "clean":
        engine_patch_file = manifest.get("engine_patch_file")
        if not engine_patch_file:
            return False, "engine_patch_file reference missing"
        engine_patch_path = rep_dir / engine_patch_file
        if not engine_patch_path.is_file():
            return False, f"engine patch file not found: {engine_patch_file}"
        recomputed = hashlib.sha256(engine_patch_path.read_bytes()).hexdigest()
        if recomputed != engine_sha:
            return False, (
                f"engine patch SHA-256 mismatch: manifest={engine_sha} "
                f"recomputed={recomputed} (patch may be tampered)"
            )

    if plugin_sha != "clean":
        plugin_patch_file = manifest.get("plugin_patch_file")
        if not plugin_patch_file:
            return False, "plugin_patch_file reference missing"
        plugin_patch_path = rep_dir / plugin_patch_file
        if not plugin_patch_path.is_file():
            return False, f"plugin patch file not found: {plugin_patch_file}"
        recomputed = hashlib.sha256(plugin_patch_path.read_bytes()).hexdigest()
        if recomputed != plugin_sha:
            return False, (
                f"plugin patch SHA-256 mismatch: manifest={plugin_sha} "
                f"recomputed={recomputed} (patch may be tampered)"
            )

    return True, ""


def compute_medians(
    results: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compute median, min, max, and count for each workload/commit.

    Returns: {workload: {commit: {median, min, max, count, values}}}
    """
    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for workload, commit_results in results.items():
        summary[workload] = {}
        for commit, values in commit_results.items():
            if not values:
                summary[workload][commit] = {
                    "median": None,
                    "min": None,
                    "max": None,
                    "count": 0,
                    "values": [],
                }
            else:
                summary[workload][commit] = {
                    "median": round(statistics.median(values), 2),
                    "min": round(min(values), 2),
                    "max": round(max(values), 2),
                    "count": len(values),
                    "values": [round(v, 2) for v in values],
                }
    return summary


def _delta_pct(base: float | None, head: float | None) -> float | None:
    """Compute percentage change from base to head."""
    if base is None or head is None or base == 0:
        return None
    return round(((head - base) / base) * 100, 2)


# Minimum number of valid results required per commit/workload to draw a
# conclusion.  Below this threshold the report must be marked incomplete
# rather than claiming no regression.
MIN_REPS_FOR_CONCLUSION = 3


def _is_valid_metric(value: float | None) -> bool:
    """Return True only for finite, positive metrics."""
    if value is None:
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    # Must be finite (not NaN/inf) and strictly positive
    import math

    return math.isfinite(v) and v > 0.0


def _check_evidence_sufficient(
    summary: dict[str, dict[str, dict[str, Any]]],
    workload: str,
    commits: list[str],
) -> tuple[bool, str]:
    """Verify each commit has at least MIN_REPS_FOR_CONCLUSION valid results.

    Returns (sufficient, reason).  ``reason`` is empty when sufficient.
    """
    for commit in commits:
        entry = summary.get(workload, {}).get(commit, {})
        count = entry.get("count", 0)
        median = entry.get("median")
        if count < MIN_REPS_FOR_CONCLUSION:
            return False, (
                f"{workload}/{commit}: only {count} result(s), "
                f"need >= {MIN_REPS_FOR_CONCLUSION}"
            )
        if not _is_valid_metric(median):
            return False, (
                f"{workload}/{commit}: median {median!r} is not finite/positive"
            )
        # Also verify every individual value is valid
        values = entry.get("values", [])
        for i, v in enumerate(values):
            if not _is_valid_metric(v):
                return False, (
                    f"{workload}/{commit}: rep-{i + 1} value {v!r} is not "
                    f"finite/positive"
                )
    return True, ""


def analyze_regression(summary: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """Analyze whether the suspected regressions are reproducible.

    Issue #146:
    - sonnet-throughput: 2206f1f7b7 -> 7a63f81e86 (throughput drop)
    - random-latency: 2206f1f7b7 -> 83cf83ff20 (latency increase)

    Acceptance:
    - sonnet: delta should converge within 10% (not a real regression)
    - random-latency: delta should not stably exceed 20% (not a real regression)

    Evidence gating:
    - Each compared commit/workload must have >= 3 valid, finite, positive
      results.  Otherwise the conclusion is ``incomplete`` and no
      ``clean_leaderboard_points`` action is emitted.
    """
    findings: dict[str, Any] = {}

    # --- sonnet-throughput regression check ---
    sonnet_commits = ["2206f1f7b7", "7a63f81e86"]
    sonnet_ok, sonnet_reason = _check_evidence_sufficient(
        summary, "sonnet-throughput", sonnet_commits
    )

    sonnet_base = (
        summary.get("sonnet-throughput", {}).get("2206f1f7b7", {}).get("median")
    )
    sonnet_head = (
        summary.get("sonnet-throughput", {}).get("7a63f81e86", {}).get("median")
    )
    sonnet_delta = _delta_pct(sonnet_base, sonnet_head)

    sonnet_reproducible = False
    sonnet_conclusion: str
    if not sonnet_ok:
        sonnet_conclusion = "incomplete"
    else:
        if sonnet_delta is not None and sonnet_delta < -SONNET_THROUGHPUT_THRESHOLD_PCT:
            sonnet_reproducible = True
            sonnet_conclusion = "regression_confirmed"
        else:
            sonnet_conclusion = "no_regression_or_noise"

    findings["sonnet-throughput"] = {
        "base_commit": "2206f1f7b7",
        "head_commit": "7a63f81e86",
        "base_median": sonnet_base,
        "head_median": sonnet_head,
        "delta_pct": sonnet_delta,
        "threshold_pct": SONNET_THROUGHPUT_THRESHOLD_PCT,
        "regression_reproducible": sonnet_reproducible,
        "conclusion": sonnet_conclusion,
        "evidence_sufficient": sonnet_ok,
        "evidence_reason": sonnet_reason,
    }

    # --- random-latency regression check ---
    latency_commits = ["2206f1f7b7", "83cf83ff20"]
    latency_ok, latency_reason = _check_evidence_sufficient(
        summary, "random-latency", latency_commits
    )

    latency_base = summary.get("random-latency", {}).get("2206f1f7b7", {}).get("median")
    latency_head = summary.get("random-latency", {}).get("83cf83ff20", {}).get("median")
    latency_delta = _delta_pct(latency_base, latency_head)

    latency_reproducible = False
    latency_conclusion: str
    if not latency_ok:
        latency_conclusion = "incomplete"
    else:
        if latency_delta is not None and latency_delta > RANDOM_LATENCY_THRESHOLD_PCT:
            latency_reproducible = True
            latency_conclusion = "regression_confirmed"
        else:
            latency_conclusion = "no_regression_or_noise"

    findings["random-latency"] = {
        "base_commit": "2206f1f7b7",
        "head_commit": "83cf83ff20",
        "base_median": latency_base,
        "head_median": latency_head,
        "delta_pct": latency_delta,
        "threshold_pct": RANDOM_LATENCY_THRESHOLD_PCT,
        "regression_reproducible": latency_reproducible,
        "conclusion": latency_conclusion,
        "evidence_sufficient": latency_ok,
        "evidence_reason": latency_reason,
    }

    # --- Overall conclusion ---
    any_incomplete = (not sonnet_ok) or (not latency_ok)
    any_confirmed = sonnet_reproducible or latency_reproducible

    if any_incomplete:
        overall_action = "incomplete_evidence"
        overall_resolution = (
            "Evidence insufficient for at least one comparison. "
            "Do not clean leaderboard points or draw regression conclusions "
            "until each compared commit/workload has >= "
            f"{MIN_REPS_FOR_CONCLUSION} valid results."
        )
    elif any_confirmed:
        overall_action = "bisect_and_fix"
        overall_resolution = "Regression(s) reproduced. Bisect to find root cause."
    else:
        overall_action = "no_action_diagnostic_only"
        overall_resolution = (
            "No reproducible regression. Results remain diagnostic/historical "
            "re-test artifacts and are NOT published as official leaderboard "
            "targets (max_model_len=30720 differs from the official 32768)."
        )

    findings["overall"] = {
        "any_regression_confirmed": any_confirmed,
        "any_evidence_incomplete": any_incomplete,
        "action": overall_action,
        "issue_146_resolution": overall_resolution,
    }

    return findings


def generate_report(
    summary: dict[str, dict[str, dict[str, Any]]],
    findings: dict[str, Any],
    result_dir: Path,
) -> dict[str, Any]:
    """Generate the full analysis report."""
    return {
        "schema_version": "issue-146-retest-report/v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_dir": str(result_dir),
        "engine_commits": ENGINE_COMMITS,
        "workloads": WORKLOADS,
        # This re-test uses max_model_len=30720 which differs from the
        # official fixed-target value of 32768.  Results are therefore
        # diagnostic/historical re-test artifacts and must NOT be published
        # as current official leaderboard targets.
        "artifact_class": "diagnostic_historical_retest",
        "official_target": False,
        "max_model_len_note": (
            "Re-test used max_model_len=30720 (original backfill value); "
            "official fixed-target spec requires 32768."
        ),
        "results_summary": summary,
        "regression_analysis": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("/data/issue146-retest-results"),
        help="Directory containing re-test results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file (default: stdout)",
    )
    parser.add_argument(
        "--engine-repo",
        type=Path,
        required=True,
        help="Path to vllm-hust repo for commit-object verification (required, fail-closed)",
    )
    parser.add_argument(
        "--plugin-repo",
        type=Path,
        required=True,
        help="Path to vllm-ascend-hust repo for commit-object verification (required, fail-closed)",
    )
    args = parser.parse_args()

    result_dir: Path = args.result_dir
    if not result_dir.is_dir():
        print(f"ERROR: {result_dir} not found", file=sys.stderr)
        return 2

    # Per reviewer round 6: '把脚本调用示例改成强制传入两仓库' — both repos
    # are now required CLI args.  Fail-closed: if either repo is missing or
    # not a directory, validate_env_manifest will reject all manifests.
    repo_paths: dict[str, Path] = {
        "engine": args.engine_repo,
        "plugin": args.plugin_repo,
    }

    results = collect_results(result_dir, repo_paths=repo_paths)
    summary = compute_medians(results)
    findings = analyze_regression(summary)
    report = generate_report(summary, findings, result_dir)

    output_json = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(output_json + "\n", encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(output_json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
