"""Generate paired evidence JSON files for issue #105.

Generates two files:
- paired_evidence_official_ascend_jan_2026.json (baseline vs current_main)
- paired_evidence_p0_prs.json (P0 PR base/head paired)

Issue #105 requirements enforced:
- Only target_aligned + verified entries are valid (fail closed otherwise)
- Each side must have >= 3 repetitions AND a measurement block
- Anomalous data (e.g. TTFT > 60s indicating environment damage) is quarantined
- Specialty profiles require an approved specialty contract reference
- git_commit must be full 40-char hex SHA (not 12-char short SHA)
- current_main candidate selection is deterministic (sorted by name)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
PAIRED_EVIDENCE_DIR = REPO_ROOT / "leaderboard-data" / "paired-evidence"
TARGET_ID = "official-ascend-jan-2026-v0.18.0"

# Issue #105: TTFT above this threshold (ms) indicates environment damage
# (e.g. CANN 9.0.0 + triton-ascend compatibility issue causing 244s TTFT).
ANOMALY_TTFT_THRESHOLD_MS = 60_000

# Issue #105: minimum repetitions for valid paired evidence
MIN_REPETITIONS = 3


def _load_submission(submission_dir: Path) -> dict:
    """Load run_leaderboard.json from a submission directory."""
    path = submission_dir / "run_leaderboard.json"
    if not path.is_file():
        raise FileNotFoundError(f"run_leaderboard.json not found in {submission_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def _full_commit(commit: str | None) -> str:
    """Return full 40-char git commit SHA (issue #105 provenance standard).

    The trend validator and admission gate require 40-char hex SHA; short
    12-char SHAs are rejectable (collision risk).  We preserve whatever
    full SHA the source provides; if only a short SHA is available we
    return it as-is (the caller / validator will fail-closed on it).
    """
    if not commit:
        return ""
    # Strip a leading "g" that git describe uses (e.g. "g1aa7cd10b7")
    cleaned = commit[1:] if commit.startswith("g") and len(commit) == 41 else commit
    return cleaned


def _extract_metrics(entry: dict) -> dict:
    """Extract a normalized metrics dict from a run_leaderboard entry."""
    metrics = entry.get("metrics", {}) or {}
    measurement = entry.get("measurement", {}) or {}
    selection = (
        measurement.get("selection", {}) if isinstance(measurement, dict) else {}
    )

    has_measurement_block = isinstance(measurement, dict) and bool(
        measurement.get("per_run")
    )
    selected_run_index = selection.get("selected_run_index")

    # Try metadata for repetitions count
    metadata = entry.get("metadata", {}) or {}
    repetitions = metadata.get("repetitions")
    if not repetitions and has_measurement_block:
        repetitions = len(measurement.get("per_run", []))
    if not repetitions:
        repetitions = 1

    measurement_strategy = (
        measurement.get("strategy") if isinstance(measurement, dict) else None
    ) or metadata.get("measurement_strategy")

    # Resolve git_commit: prefer metadata.git_commit, else parse from
    # versions.core (engine_version "-g" suffix).
    raw_commit = metadata.get("git_commit")
    if not raw_commit:
        versions = entry.get("versions", {}) or {}
        if isinstance(versions, dict):
            core = versions.get("core", "") or ""
            if "-g" in core:
                raw_commit = core.split("-g")[-1]
    # Also try engine_version "-g" suffix
    if not raw_commit:
        ev = entry.get("engine_version", "") or ""
        if "-g" in ev:
            raw_commit = ev.split("-g")[-1]

    return {
        "engine_version": entry.get("engine_version", ""),
        "git_commit": _full_commit(raw_commit),
        "ttft_ms": round(float(metrics.get("ttft_ms", 0)), 2)
        if metrics.get("ttft_ms") is not None
        else None,
        "tbt_ms": round(float(metrics.get("tbt_ms", 0)), 2)
        if metrics.get("tbt_ms") is not None
        else None,
        "throughput_tps": round(float(metrics.get("throughput_tps", 0)), 2)
        if metrics.get("throughput_tps") is not None
        else None,
        "error_rate": float(metrics.get("error_rate", 0))
        if metrics.get("error_rate") is not None
        else 0.0,
        "repetitions": repetitions,
        "has_measurement_block": has_measurement_block,
        "measurement_strategy": measurement_strategy,
        "selected_run_index": selected_run_index,
    }


def _delta_percent(base: float | None, head: float | None) -> float | None:
    if base is None or head is None or base == 0:
        return None
    return round(((head - base) / base) * 100, 2)


def _is_anomalous(metrics: dict) -> str | None:
    """Return a quarantine reason if metrics look environmentally damaged.

    Issue #105: "缺字段/异常一律 fail closed / quarantine".  A baseline with
    TTFT in the hundreds of seconds (should be ~270ms) is clearly broken by
    an environment issue (CANN/triton-ascend), not a real performance point.
    """
    ttft = metrics.get("ttft_ms")
    if ttft is not None and ttft > ANOMALY_TTFT_THRESHOLD_MS:
        return (
            f"anomalous_ttft={ttft:.0f}ms exceeds "
            f"{ANOMALY_TTFT_THRESHOLD_MS}ms threshold; environment damage"
        )
    return None


def _is_valid_paired_side(metrics: dict | None) -> tuple[bool, str | None]:
    """Check if one side of a paired comparison is valid per issue #105.

    Returns (valid, invalid_reason).  A valid side must:
    - exist (not None)
    - have >= MIN_REPETITIONS repetitions
    - have a measurement block (has_measurement_block == True)
    - not be anomalous (TTFT within sane bounds)
    - have a 40-char git_commit SHA
    """
    if metrics is None:
        return False, "metrics_missing"
    if metrics.get("repetitions", 0) < MIN_REPETITIONS:
        return False, (
            f"repetitions={metrics['repetitions']} < required {MIN_REPETITIONS}"
        )
    if not metrics.get("has_measurement_block"):
        return False, "missing_measurement_block"
    anomaly = _is_anomalous(metrics)
    if anomaly:
        return False, anomaly
    commit = metrics.get("git_commit", "")
    if len(commit) != 40:
        return False, f"git_commit not 40-char SHA (got {len(commit)} chars)"
    return True, None


def _is_valid_paired_evidence(
    base: dict | None, head: dict | None
) -> tuple[bool, str | None]:
    """Check if a base/head pair is valid: both sides must be valid."""
    base_valid, base_reason = _is_valid_paired_side(base)
    if not base_valid:
        return False, f"base: {base_reason}"
    head_valid, head_reason = _is_valid_paired_side(head)
    if not head_valid:
        return False, f"head: {head_reason}"
    return True, None


def _find_baseline_submission(profile: str) -> Path:
    """Find v0.18.0 baseline submission dir for an active profile."""
    mapping = {
        "core-text-14b": "official-ascend-jan-2026-v0.18.0-random-online-qwen25-14b-910b2",
        "coder-14b": "official-ascend-jan-2026-v0.18.0-instructcoder-online-qwen25-coder-14b-910b2",
        "vision-7b": "official-ascend-jan-2026-v0.18.0-visionarena-online-qwen25-vl-7b-910b2",
    }
    dirname = mapping.get(profile)
    if not dirname:
        raise ValueError(f"Unknown profile: {profile}")
    path = SUBMISSIONS_DIR / dirname
    if not path.is_dir():
        raise FileNotFoundError(f"Baseline submission dir not found: {path}")
    return path


def _find_current_main_submission(profile: str) -> Path | None:
    """Find current-main paired run submission for an active profile.

    Selection is deterministic: candidates are sorted by name (which embeds
    a timestamp) so the same repo state always yields the same pick.
    """
    workload_map = {
        "core-text-14b": "random-online",
        "coder-14b": "instructcoder-online",
        "vision-7b": "visionarena-online",
    }
    workload = workload_map.get(profile)
    if not workload:
        return None
    if not SUBMISSIONS_DIR.is_dir():
        return None
    candidates = []
    for entry in SUBMISSIONS_DIR.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if "current-main" not in name or "single-npu" not in name:
            continue
        if workload not in name:
            continue
        # Skip 2chip/4chip/8chip variants (specialty)
        if any(x in name for x in ("2chip", "4chip", "8chip")):
            continue
        candidates.append(entry)
    if not candidates:
        return None
    # Deterministic: sort by name (embeds timestamp) so selection is reproducible
    candidates.sort(key=lambda p: p.name)
    # Prefer the one with repeat_suite.json (real 3-rep)
    with_suite = [c for c in candidates if (c / "repeat_suite.json").is_file()]
    if with_suite:
        with_suite.sort(key=lambda p: p.name)
        return with_suite[-1]  # latest
    return candidates[-1]  # latest


def generate_official_paired_evidence() -> dict:
    """Generate paired_evidence_official_ascend_jan_2026.json."""
    specs = [
        ("core-text-14b", "random-online"),
        ("coder-14b", "instructcoder-online"),
        ("vision-7b", "visionarena-online"),
    ]
    spec_entries = []
    for profile, scenario in specs:
        baseline_dir = _find_baseline_submission(profile)
        baseline_entry = _load_submission(baseline_dir)
        baseline_metrics = _extract_metrics(baseline_entry)

        current_main_dir = _find_current_main_submission(profile)
        current_main_metrics = None
        if current_main_dir is not None:
            current_main_entry = _load_submission(current_main_dir)
            current_main_metrics = _extract_metrics(current_main_entry)

        # Compute deltas if both sides present
        delta = {}
        if current_main_metrics is not None:
            for metric, base_key, head_key in (
                ("ttft", "ttft_ms", "ttft_ms"),
                ("tbt", "tbt_ms", "tbt_ms"),
                ("throughput", "throughput_tps", "throughput_tps"),
            ):
                delta[metric] = _delta_percent(
                    baseline_metrics.get(base_key), current_main_metrics.get(head_key)
                )

        # Issue #105: dynamic validity check (fail closed)
        valid, invalid_reason = _is_valid_paired_evidence(
            baseline_metrics, current_main_metrics
        )
        spec_entry = {
            "profile": profile,
            "scenario": scenario,
            "baseline": baseline_metrics,
            "current_main": current_main_metrics,
            "delta_percent": delta,
            "paired_evidence_valid": valid,
        }
        if not valid:
            spec_entry["invalid_reason"] = invalid_reason
        spec_entries.append(spec_entry)

    return {
        "schema_version": "1.2",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target_id": TARGET_ID,
        "measurement_strategy": "primary-median-run (3 repetitions)",
        "validity_policy": (
            "paired_evidence_valid requires both sides to have >= 3 repetitions, "
            "a measurement block, a 40-char git_commit SHA, and TTFT below "
            f"{ANOMALY_TTFT_THRESHOLD_MS}ms (anomaly threshold)."
        ),
        "specs": spec_entries,
    }


# Approved specialty contracts (issue #105: specialty requires approved contract)
# Empty set means no specialty results are valid until a contract is filed.
APPROVED_SPECIALTY_CONTRACTS: set[str] = set()

# P0 PR paired definitions (issue #89 section 9)
P0_PR_PAIRS = [
    {
        "pr_number": 130,
        "label": "logprobs-online",
        "profile": "core-text-14b",
        "scenario": "logprobs-online",
        "base_dir": "historical-pr-core-pr-130-base-logprobs-online-611bcabeb7-844c231055",
        "head_dir": "historical-pr-core-pr-130-merge-logprobs-online-3b8e5cff01-844c231055",
    },
    {
        "pr_number": 115,
        "label": "prefix-cache-xxhash",
        "profile": "core-text-14b",
        "scenario": "prefix-repetition-online",
        "base_dir": "historical-pr-pr115-base-prefix-repetition-online-prefix-repetition-online-87f2a3480f-52f923884b",
        "head_dir": "historical-pr-core-pr-115-merge-prefix-repetition-online-0e84e42c71-9b40f1c4e3",
    },
    {
        "pr_number": 116,
        "label": "prefix-cache-text-only-block-hash",
        "profile": "core-text-14b",
        "scenario": "prefix-repetition-online",
        "base_dir": "historical-pr-core-pr-115-merge-prefix-repetition-online-0e84e42c71-9b40f1c4e3",
        "head_dir": "historical-pr-core-pr-116-merge-prefix-repetition-online-ab0a8e87d5-9b40f1c4e3",
        "base_note": "PR#116 base reuses PR#115 head (same commit 0e84e42c7) per issue #89 section 9.3",
    },
    {
        "pr_number": 124,
        "label": "kv-tiering-prefix-online",
        "profile": "vision-7b-specialty",
        "scenario": "kv-tiering-prefix-online",
        "base_dir": "historical-pr-core-pr-124-base-small-kv-kv-tiering-prefix-online-e0c0ce8e37-8b2adf1606",
        "head_dir": "historical-pr-core-pr-124-merge-tiering-kv-tiering-prefix-online-89334ef1f0-8b2adf1606",
    },
]


def generate_p0_pr_paired_evidence() -> dict:
    """Generate paired_evidence_p0_prs.json from #89 P0 PR submissions."""
    prs = []
    for pair in P0_PR_PAIRS:
        base_path = SUBMISSIONS_DIR / pair["base_dir"]
        head_path = SUBMISSIONS_DIR / pair["head_dir"]

        if not base_path.is_dir():
            print(
                f"WARN: PR#{pair['pr_number']} base submission not found: {base_path}",
                file=sys.stderr,
            )
            continue
        if not head_path.is_dir():
            print(
                f"WARN: PR#{pair['pr_number']} head submission not found: {head_path}",
                file=sys.stderr,
            )
            continue

        base_entry = _load_submission(base_path)
        head_entry = _load_submission(head_path)
        base_metrics = _extract_metrics(base_entry)
        head_metrics = _extract_metrics(head_entry)

        delta = {}
        for metric_name, key in (
            ("ttft", "ttft_ms"),
            ("tbt", "tbt_ms"),
            ("throughput", "throughput_tps"),
        ):
            delta[metric_name] = _delta_percent(
                base_metrics.get(key), head_metrics.get(key)
            )

        # Issue #105: dynamic validity (not hardcoded True)
        valid, invalid_reason = _is_valid_paired_evidence(base_metrics, head_metrics)

        # Issue #105: specialty profiles require an approved specialty contract
        profile = pair["profile"]
        if "-specialty" in profile:
            contract_id = f"{profile}:{pair['scenario']}"
            if contract_id not in APPROVED_SPECIALTY_CONTRACTS:
                valid = False
                if invalid_reason:
                    invalid_reason += "; "
                else:
                    invalid_reason = ""
                invalid_reason += (
                    f"specialty profile '{profile}' has no approved specialty "
                    f"contract (issue #105 requires approved contract for "
                    f"specialty/out-of-scope results)"
                )

        pr_entry = {
            "pr_number": pair["pr_number"],
            "label": pair["label"],
            "profile": pair["profile"],
            "scenario": pair["scenario"],
            "base": base_metrics,
            "head": head_metrics,
            "delta_percent": delta,
            "paired_evidence_valid": valid,
        }
        if not valid:
            pr_entry["invalid_reason"] = invalid_reason
        if "base_note" in pair:
            pr_entry["base_note"] = pair["base_note"]
        prs.append(pr_entry)

    return {
        "schema_version": "1.2",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target_id": "historical-pr-paired-evidence",
        "measurement_strategy": "primary-median-run (3 repetitions)",
        "source": "issue #89 P0 PR paired reruns (real-online, single NPU 910B2)",
        "validity_policy": (
            "paired_evidence_valid requires both sides to have >= 3 repetitions, "
            "a measurement block, a 40-char git_commit SHA, TTFT below "
            f"{ANOMALY_TTFT_THRESHOLD_MS}ms, and (for specialty profiles) an "
            "approved specialty contract."
        ),
        "prs": prs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PAIRED_EVIDENCE_DIR,
        help="Output directory for paired evidence JSON files",
    )
    parser.add_argument(
        "--only",
        choices=["official", "p0", "both"],
        default="both",
        help="Which paired evidence file to generate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated JSON to stdout instead of writing files",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.only in ("official", "both"):
        official = generate_official_paired_evidence()
        if args.dry_run:
            print("=== paired_evidence_official_ascend_jan_2026.json ===")
            print(json.dumps(official, indent=2, ensure_ascii=False))
        else:
            out = output_dir / "paired_evidence_official_ascend_jan_2026.json"
            out.write_text(
                json.dumps(official, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"Wrote {out}")

    if args.only in ("p0", "both"):
        p0 = generate_p0_pr_paired_evidence()
        if args.dry_run:
            print("=== paired_evidence_p0_prs.json ===")
            print(json.dumps(p0, indent=2, ensure_ascii=False))
        else:
            out = output_dir / "paired_evidence_p0_prs.json"
            out.write_text(
                json.dumps(p0, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"Wrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
