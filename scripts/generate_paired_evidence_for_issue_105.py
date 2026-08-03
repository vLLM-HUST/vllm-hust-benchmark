#!/usr/bin/env python3
"""Issue #105 paired evidence generator.

Generates two paired-evidence JSON files under leaderboard-data/paired-evidence/:

1. paired_evidence_official_ascend_jan_2026.json
   - 3 active specs (core-text-14b, coder-14b, vision-7b)
   - baseline = v0.18.0 3-rep canonical (from official-ascend-jan-2026-v0.18.0-* submissions)
   - current_main = latest main commit paired run (from historical-pr-current-main-* submissions)
   - delta_percent for ttft/tbt/throughput

2. paired_evidence_p0_prs.json
   - 4 P0 PRs from issue #89 (PR#130/#115/#116/#124)
   - base vs merge (or base vs head) per PR
   - PR#116 base reuses PR#115 head (same commit), per #89 mapping section 9.3
   - delta_percent for ttft/tbt/throughput

Reads metrics + measurement block from each submission's run_leaderboard.json.
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


def _load_submission(submission_dir: Path) -> dict:
    """Load run_leaderboard.json from a submission directory."""
    path = submission_dir / "run_leaderboard.json"
    if not path.is_file():
        raise FileNotFoundError(f"run_leaderboard.json not found in {submission_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def _short_commit(commit: str | None) -> str:
    if not commit:
        return ""
    return commit[:12]


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

    return {
        "engine_version": entry.get("engine_version", ""),
        "git_commit": _short_commit(
            metadata.get("git_commit")
            or (entry.get("versions", {}) or {}).get("core", "").split("-g")[-1]
            if isinstance(entry.get("versions"), dict)
            else None
        ),
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

    For core-text-14b: historical-pr-current-main-single-npu-prefix-clean-random-online-*
    For coder-14b: historical-pr-current-main-single-npu-smoke-instructcoder-online-*
    For vision-7b: historical-pr-current-main-single-npu-visionarena-online-* (or similar)
    """
    workload_map = {
        "core-text-14b": "random-online",
        "coder-14b": "instructcoder-online",
        "vision-7b": "visionarena-online",
    }
    workload = workload_map.get(profile)
    if not workload:
        return None
    # Look for current-main submissions matching the workload
    candidates = []
    if not SUBMISSIONS_DIR.is_dir():
        return None
    for entry in SUBMISSIONS_DIR.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if "current-main" not in name or "single-npu" not in name:
            continue
        # Match workload in directory name
        if workload not in name:
            continue
        # Skip 2chip/4chip/8chip variants (specialty)
        if any(x in name for x in ("2chip", "4chip", "8chip")):
            continue
        candidates.append(entry)
    if not candidates:
        return None
    # Prefer the one with repeat_suite.json (real 3-rep)
    with_suite = [c for c in candidates if (c / "repeat_suite.json").is_file()]
    if with_suite:
        return with_suite[0]
    return candidates[0]


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
            # Override git_commit for current_main (use engine_version's -g suffix)
            ev = current_main_entry.get("engine_version", "")
            if "-g" in ev:
                current_main_metrics["git_commit"] = _short_commit(ev.split("-g")[-1])

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

        spec_entries.append(
            {
                "profile": profile,
                "scenario": scenario,
                "baseline": baseline_metrics,
                "current_main": current_main_metrics,
                "delta_percent": delta,
                "paired_evidence_valid": current_main_metrics is not None,
            }
        )

    return {
        "schema_version": "1.1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target_id": TARGET_ID,
        "measurement_strategy": "primary-median-run (3 repetitions)",
        "specs": spec_entries,
    }


# P0 PR paired definitions (issue #89 section 9)
P0_PR_PAIRS = [
    {
        "pr_number": 130,
        "label": "logprobs-online",
        "profile": "core-text-14b",  # 14B random-online line; logprobs is a workload variant
        "scenario": "logprobs-online",
        "base_dir": "historical-pr-core-pr-130-base-logprobs-online-611bcabeb7-844c231055",
        "head_dir": "historical-pr-core-pr-130-merge-logprobs-online-3b8e5cff01-844c231055",
    },
    {
        "pr_number": 115,
        "label": "prefix-cache-xxhash",
        "profile": "core-text-14b",
        "scenario": "prefix-repetition-online",
        # base is on origin/feat/issue-89-evidence (not on main); head is on main
        "base_dir": "historical-pr-pr115-base-prefix-repetition-online-prefix-repetition-online-87f2a3480f-52f923884b",
        "head_dir": "historical-pr-core-pr-115-merge-prefix-repetition-online-0e84e42c71-9b40f1c4e3",
    },
    {
        "pr_number": 116,
        "label": "prefix-cache-text-only-block-hash",
        "profile": "core-text-14b",
        "scenario": "prefix-repetition-online",
        # PR#116 base reuses PR#115 head (same commit 0e84e42c7) per #89 mapping section 9.3
        "base_dir": "historical-pr-core-pr-115-merge-prefix-repetition-online-0e84e42c71-9b40f1c4e3",
        "head_dir": "historical-pr-core-pr-116-merge-prefix-repetition-online-ab0a8e87d5-9b40f1c4e3",
        "base_note": "PR#116 base reuses PR#115 head (same commit 0e84e42c7) per issue #89 section 9.3",
    },
    {
        "pr_number": 124,
        "label": "kv-tiering-prefix-online",
        "profile": "vision-7b-specialty",  # 7B specialty (not main 14B line)
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

        # Override git_commit from engine_version -g suffix
        for entry, metrics in ((base_entry, base_metrics), (head_entry, head_metrics)):
            ev = entry.get("engine_version", "")
            if "-g" in ev:
                metrics["git_commit"] = _short_commit(ev.split("-g")[-1])

        delta = {}
        for metric_name, key in (
            ("ttft", "ttft_ms"),
            ("tbt", "tbt_ms"),
            ("throughput", "throughput_tps"),
        ):
            delta[metric_name] = _delta_percent(
                base_metrics.get(key), head_metrics.get(key)
            )

        pr_entry = {
            "pr_number": pair["pr_number"],
            "label": pair["label"],
            "profile": pair["profile"],
            "scenario": pair["scenario"],
            "base": base_metrics,
            "head": head_metrics,
            "delta_percent": delta,
            "paired_evidence_valid": True,
        }
        if "base_note" in pair:
            pr_entry["base_note"] = pair["base_note"]
        prs.append(pr_entry)

    return {
        "schema_version": "1.1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target_id": "historical-pr-paired-evidence",
        "measurement_strategy": "primary-median-run (3 repetitions)",
        "source": "issue #89 P0 PR paired reruns (real-online, single NPU 910B2)",
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
