from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

COMPARABLE_FIELDS = (
    "schema_version",
    "benchmark_config_fingerprint",
    "runner_class",
    "soc_version",
)
GATE_THRESHOLDS = {
    "serve.output_throughput_tps": {"direction": "higher", "threshold_pct": 5.0},
    "throughput.tokens_per_second": {"direction": "higher", "threshold_pct": 5.0},
    "serve.mean_ttft_ms": {"direction": "lower", "threshold_pct": 8.0},
    "latency.mean_ms": {"direction": "lower", "threshold_pct": 8.0},
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _nested_get(payload: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _percent_delta(current: float, baseline: float) -> float | None:
    if baseline == 0:
        return None
    return ((current - baseline) / baseline) * 100.0


def _check_comparable(current: Mapping[str, Any], baseline: Mapping[str, Any]) -> str | None:
    for field in COMPARABLE_FIELDS:
        if current.get(field) != baseline.get(field):
            return field
    return None


def compare_against_baseline(
    *,
    current: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    baseline_label: str,
) -> dict[str, Any]:
    if baseline is None:
        return {"label": baseline_label, "status": "missing", "metrics": {}, "failures": []}

    comparable_mismatch = _check_comparable(current, baseline)
    if comparable_mismatch is not None:
        status = (
            "fingerprint_mismatch"
            if comparable_mismatch == "benchmark_config_fingerprint"
            else "incomparable"
        )
        return {
            "label": baseline_label,
            "status": status,
            "mismatch_field": comparable_mismatch,
            "metrics": {},
            "failures": [],
        }

    failures: list[str] = []
    metrics: dict[str, Any] = {}
    for metric_name, rule in GATE_THRESHOLDS.items():
        current_value = _to_float(_nested_get(current, metric_name))
        baseline_value = _to_float(_nested_get(baseline, metric_name))
        delta_pct = None
        if current_value is not None and baseline_value is not None:
            delta_pct = _percent_delta(current_value, baseline_value)
            if delta_pct is not None:
                if rule["direction"] == "higher" and delta_pct < -rule["threshold_pct"]:
                    failures.append(metric_name)
                if rule["direction"] == "lower" and delta_pct > rule["threshold_pct"]:
                    failures.append(metric_name)
        metrics[metric_name] = {
            "current": current_value,
            "baseline": baseline_value,
            "delta_pct": delta_pct,
        }

    error_rate = _to_float(_nested_get(current, "serve.error_rate")) or 0.0
    if error_rate > 0:
        failures.append("serve.error_rate")

    return {
        "label": baseline_label,
        "status": "available",
        "metrics": metrics,
        "failures": sorted(set(failures)),
    }


def build_compare_summary(
    *,
    current: Mapping[str, Any],
    same_peer_ancestor: Mapping[str, Any] | None,
    latest_protected_combo: Mapping[str, Any] | None,
    branch_freshness: Mapping[str, Any] | None,
) -> dict[str, Any]:
    same_peer_summary = compare_against_baseline(
        current=current,
        baseline=same_peer_ancestor,
        baseline_label="same_peer_ancestor",
    )
    latest_protected_summary = compare_against_baseline(
        current=current,
        baseline=latest_protected_combo,
        baseline_label="latest_protected_combo",
    )

    would_fail = False
    for baseline_summary in (same_peer_summary, latest_protected_summary):
        if baseline_summary["status"] == "available" and baseline_summary["failures"]:
            would_fail = True

    error_rate = _to_float(_nested_get(current, "serve.error_rate")) or 0.0
    if error_rate > 0:
        would_fail = True

    if branch_freshness and branch_freshness.get("status") == "rebase_required":
        would_fail = True

    return {
        "current": current,
        "branch_freshness": branch_freshness,
        "same_peer_ancestor": same_peer_summary,
        "latest_protected_combo": latest_protected_summary,
        "would_fail": would_fail,
    }


def render_compare_summary_markdown(summary: Mapping[str, Any]) -> str:
    current = summary["current"]
    lines = [
        "# Ascend L1 Perf Smoke Summary",
        f"- Trigger repo: `{current['source_combo']['trigger_repo']}`",
        f"- Current source combo fingerprint: `{current['source_combo_fingerprint']}`",
        f"- Benchmark config fingerprint: `{current['benchmark_config_fingerprint']}`",
        f"- Would fail gate: `{summary['would_fail']}`",
    ]

    branch_freshness = summary.get("branch_freshness")
    if branch_freshness:
        lines.extend(
            [
                "",
                "## Branch Freshness",
                f"- Status: `{branch_freshness['status']}`",
                f"- Merge base age: `{branch_freshness['merge_base_age_days']}` days",
                f"- Base branch ahead commits: `{branch_freshness['base_branch_ahead_commits']}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Current Metrics",
            f"- Output throughput: `{_nested_get(current, 'serve.output_throughput_tps')}` tok/s",
            f"- Tokens/s: `{_nested_get(current, 'throughput.tokens_per_second')}`",
            f"- Mean TTFT: `{_nested_get(current, 'serve.mean_ttft_ms')}` ms",
            f"- Mean latency: `{_nested_get(current, 'latency.mean_ms')}` ms",
            f"- Error rate: `{_nested_get(current, 'serve.error_rate')}`",
        ]
    )

    for key, title in (
        ("same_peer_ancestor", "Same-Peer Ancestor"),
        ("latest_protected_combo", "Latest Protected Combo"),
    ):
        comparison = summary[key]
        lines.extend(["", f"## {title}", f"- Status: `{comparison['status']}`"])
        if comparison["status"] != "available":
            mismatch_field = comparison.get("mismatch_field")
            if mismatch_field:
                lines.append(f"- Mismatch field: `{mismatch_field}`")
            continue
        lines.append(
            f"- Failures: `{', '.join(comparison['failures']) if comparison['failures'] else 'none'}`"
        )
        for metric_name, metric_payload in comparison["metrics"].items():
            lines.append(
                "- "
                f"{metric_name}: current=`{metric_payload['current']}` baseline=`{metric_payload['baseline']}` "
                f"delta=`{metric_payload['delta_pct']}`%"
            )
    return "\n".join(lines) + "\n"


def compare_combo_baseline(
    *,
    head_result: Path,
    same_peer_ancestor_result: Path | None,
    latest_protected_result: Path | None,
    branch_freshness_json: Path | None,
    summary_file: Path | None,
    summary_json: Path | None,
    strict: bool,
) -> int:
    current = _read_json(head_result)
    same_peer_ancestor = (
        _read_json(same_peer_ancestor_result) if same_peer_ancestor_result else None
    )
    latest_protected = (
        _read_json(latest_protected_result) if latest_protected_result else None
    )
    branch_freshness = (
        _read_json(branch_freshness_json) if branch_freshness_json else None
    )
    summary = build_compare_summary(
        current=current,
        same_peer_ancestor=same_peer_ancestor,
        latest_protected_combo=latest_protected,
        branch_freshness=branch_freshness,
    )
    markdown = render_compare_summary_markdown(summary)

    if summary_file:
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(markdown, encoding="utf-8")
    else:
        print(markdown, end="")

    if summary_json:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    step_summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary_path:
        Path(step_summary_path).write_text(markdown, encoding="utf-8")

    if strict and summary["would_fail"]:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="compare-combo-baseline",
        description="Compare an L1 perf smoke head result against same-peer ancestor and latest protected combo baselines.",
    )
    parser.add_argument("--head-result", required=True)
    parser.add_argument("--same-peer-ancestor-result")
    parser.add_argument("--latest-protected-result")
    parser.add_argument("--branch-freshness-json")
    parser.add_argument("--summary-file")
    parser.add_argument("--summary-json")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    return compare_combo_baseline(
        head_result=Path(args.head_result).resolve(),
        same_peer_ancestor_result=(
            Path(args.same_peer_ancestor_result).resolve()
            if args.same_peer_ancestor_result
            else None
        ),
        latest_protected_result=(
            Path(args.latest_protected_result).resolve()
            if args.latest_protected_result
            else None
        ),
        branch_freshness_json=(
            Path(args.branch_freshness_json).resolve()
            if args.branch_freshness_json
            else None
        ),
        summary_file=Path(args.summary_file).resolve() if args.summary_file else None,
        summary_json=Path(args.summary_json).resolve() if args.summary_json else None,
        strict=args.strict,
    )


if __name__ == "__main__":
    raise SystemExit(main())