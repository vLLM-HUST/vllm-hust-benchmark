from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricComparison:
    name: str
    baseline: float
    current: float
    direction: str
    passed: bool
    delta_percent: float


@dataclass(frozen=True)
class StageComparison:
    passed: bool
    metrics: dict[str, MetricComparison]


@dataclass(frozen=True)
class TwoStageReport:
    overall_passed: bool
    markdown: str
    stage1: StageComparison
    stage2: StageComparison | None = None


METRICS: tuple[tuple[str, str], ...] = (
    ("throughput_tps", "higher_is_better"),
    ("ttft_ms", "lower_is_better"),
    ("tbt_ms", "lower_is_better"),
)

DISPLAY_NAMES = {
    "throughput_tps": "Throughput (tok/s)",
    "ttft_ms": "TTFT (ms)",
    "tbt_ms": "TBT (ms)",
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be a JSON object")
    return payload


def _extract_metrics(path: Path) -> dict[str, float]:
    payload = _load_json(path)
    raw_metrics = payload.get("metrics")
    if not isinstance(raw_metrics, dict):
        raise ValueError(f"{path} must include object key: metrics")

    metrics: dict[str, float] = {}
    missing: list[str] = []
    for name, _direction in METRICS:
        if name not in raw_metrics or raw_metrics[name] is None:
            missing.append(name)
            continue
        try:
            value = float(raw_metrics[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path} metrics.{name} must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"{path} metrics.{name} must be finite")
        if value < 0:
            raise ValueError(f"{path} metrics.{name} must be non-negative")
        metrics[name] = value
    if missing:
        raise ValueError(f"{path} metrics missing required non-null keys: {', '.join(missing)}")
    return metrics


def _same_spec_identity(path: Path, *, expected_spec_id: str | None = None) -> tuple[str, str]:
    payload = _load_json(path)
    same_spec = payload.get("same_spec")
    if not isinstance(same_spec, dict):
        raise ValueError(f"{path} must include object key: same_spec")
    spec_id = str(same_spec.get("spec_id") or "").strip()
    spec_hash = str(same_spec.get("resolved_spec_hash") or "").strip()
    if not spec_id:
        raise ValueError(f"{path} same_spec.spec_id must be non-empty")
    if not spec_hash:
        raise ValueError(f"{path} same_spec.resolved_spec_hash must be non-empty")
    if expected_spec_id and spec_id != expected_spec_id:
        raise ValueError(f"{path} same_spec.spec_id must be {expected_spec_id!r}, got {spec_id!r}")
    return spec_id, spec_hash


def _validate_same_spec(current_path: Path, baseline_path: Path, *, expected_spec_id: str | None = None) -> None:
    current_spec_id, current_hash = _same_spec_identity(current_path, expected_spec_id=expected_spec_id)
    baseline_spec_id, baseline_hash = _same_spec_identity(baseline_path, expected_spec_id=expected_spec_id)
    if current_spec_id != baseline_spec_id:
        raise ValueError(
            "same_spec.spec_id mismatch: "
            f"current={current_spec_id!r} baseline={baseline_spec_id!r}"
        )
    if current_hash != baseline_hash:
        raise ValueError(
            "same_spec.resolved_spec_hash mismatch: "
            f"current={current_hash!r} baseline={baseline_hash!r}"
        )


def _compare_metric(name: str, current: float, baseline: float, direction: str) -> MetricComparison:
    if direction == "higher_is_better":
        passed = current >= baseline
    elif direction == "lower_is_better":
        passed = current <= baseline
    else:  # pragma: no cover - internal guard
        raise ValueError(f"unknown metric direction: {direction}")

    if baseline == 0:
        delta_percent = 0.0 if current == 0 else float("inf")
    else:
        delta_percent = ((current - baseline) / baseline) * 100.0

    return MetricComparison(
        name=name,
        baseline=baseline,
        current=current,
        direction=direction,
        passed=passed,
        delta_percent=round(delta_percent, 2),
    )


def compare_benchmark_results(
    current: Path | str,
    baseline: Path | str,
    *,
    expected_spec_id: str | None = None,
) -> StageComparison:
    current_path = Path(current)
    baseline_path = Path(baseline)
    _validate_same_spec(current_path, baseline_path, expected_spec_id=expected_spec_id)
    current_metrics = _extract_metrics(current_path)
    baseline_metrics = _extract_metrics(baseline_path)

    comparisons = {
        name: _compare_metric(
            name=name,
            current=current_metrics[name],
            baseline=baseline_metrics[name],
            direction=direction,
        )
        for name, direction in METRICS
    }
    return StageComparison(
        passed=all(metric.passed for metric in comparisons.values()),
        metrics=comparisons,
    )


def _format_metric_table(comparison: StageComparison) -> str:
    lines = [
        "| Metric | Baseline | Current | Delta | Status |",
        "|--------|----------|---------|-------|--------|",
    ]
    for name, _direction in METRICS:
        metric = comparison.metrics[name]
        if math.isinf(metric.delta_percent):
            delta = "+∞%"
        else:
            delta = f"{metric.delta_percent:+.2f}%"
        status = "PASS" if metric.passed else "FAIL"
        lines.append(
            f"| {DISPLAY_NAMES[name]} | {metric.baseline:.2f} | {metric.current:.2f} | {delta} | {status} |"
        )
    return "\n".join(lines)


def _stage_status(comparison: StageComparison) -> str:
    return "PASS" if comparison.passed else "FAIL"


def _validate_two_stage_inputs(
    *,
    fork_point: str | None,
    m2_commit: str | None,
    stage2_current_file: Path | str | None,
    stage2_baseline_file: Path | str | None,
    stage2_rebase_conflict: bool,
    stage2_skipped: bool,
    stage2_not_run: bool = False,
) -> None:
    active_states = sum(1 for value in (stage2_rebase_conflict, stage2_skipped, stage2_not_run) if value)
    if active_states > 1:
        raise ValueError("stage2 states are mutually exclusive")
    if stage2_skipped:
        if not fork_point or not m2_commit:
            raise ValueError("stage2 skipped requires both fork-point and m2-commit")
        if fork_point != m2_commit:
            raise ValueError("stage2 can be skipped only when fork-point equals m2-commit")
        if stage2_current_file or stage2_baseline_file:
            raise ValueError("stage2 current/baseline files must not be provided when stage2 is skipped")
    if stage2_rebase_conflict and (stage2_current_file or stage2_baseline_file):
        raise ValueError("stage2 current/baseline files must not be provided when stage2 has rebase conflict")
    if stage2_not_run and (stage2_current_file or stage2_baseline_file):
        raise ValueError("stage2 current/baseline files must not be provided when stage2 was not run")


def generate_two_stage_report(
    *,
    stage1_current_file: Path | str,
    stage1_baseline_file: Path | str,
    fork_point: str | None = None,
    stage2_current_file: Path | str | None = None,
    stage2_baseline_file: Path | str | None = None,
    m2_commit: str | None = None,
    stage2_rebase_conflict: bool = False,
    stage2_rebase_conflict_file: Path | str | None = None,
    stage2_skipped: bool = False,
    stage2_skip_reason: str | None = None,
    stage2_not_run: bool = False,
    stage2_not_run_reason: str | None = None,
    expected_spec_id: str | None = None,
    mode: str = "report",
) -> TwoStageReport:
    _validate_two_stage_inputs(
        fork_point=fork_point,
        m2_commit=m2_commit,
        stage2_current_file=stage2_current_file,
        stage2_baseline_file=stage2_baseline_file,
        stage2_rebase_conflict=stage2_rebase_conflict,
        stage2_skipped=stage2_skipped,
        stage2_not_run=stage2_not_run,
    )
    stage1 = compare_benchmark_results(
        Path(stage1_current_file),
        Path(stage1_baseline_file),
        expected_spec_id=expected_spec_id,
    )
    stage2: StageComparison | None = None
    lines = ["## Performance Gate Result", "", "### Stage 1: B1 vs M1 (fork-point)", "", _format_metric_table(stage1), ""]
    lines.append(f"**Stage 1: {_stage_status(stage1)}**")
    if fork_point:
        lines.append(f"- Fork point (M1): `{fork_point}`")
    lines.extend(["", "### Stage 2: B1' (rebased) vs M2 (latest main)", ""])

    if stage2_rebase_conflict:
        overall_passed = False
        lines.extend([
            "**Stage 2: FAIL — rebase conflict**",
            "",
            "PR cannot be cleanly rebased onto latest main.",
            "Please rebase manually and resolve conflicts.",
        ])
        if stage2_rebase_conflict_file:
            conflict_path = Path(stage2_rebase_conflict_file)
            details = conflict_path.read_text(encoding="utf-8") if conflict_path.exists() else ""
            if len(details) > 12000:
                details = details[:12000] + "\n... truncated ...\n"
            if details:
                lines.extend([
                    "",
                    "<details>",
                    "<summary>Conflict details</summary>",
                    "",
                    "```text",
                    details.rstrip(),
                    "```",
                    "",
                    "</details>",
                ])
    elif stage2_skipped:
        overall_passed = stage1.passed
        reason = stage2_skip_reason or "fork-point is already latest main"
        lines.append(f"**Stage 2: SKIPPED** — {reason}")
    elif stage2_not_run:
        overall_passed = False
        reason = stage2_not_run_reason or "Stage 2 was not run"
        lines.append(f"**Stage 2: NOT RUN** — {reason}")
    else:
        if not stage2_current_file or not stage2_baseline_file:
            raise ValueError("stage2 current/baseline files are required unless stage2 is skipped or conflicted")
        stage2 = compare_benchmark_results(
            Path(stage2_current_file),
            Path(stage2_baseline_file),
            expected_spec_id=expected_spec_id,
        )
        overall_passed = stage1.passed and stage2.passed
        lines.extend([_format_metric_table(stage2), "", f"**Stage 2: {_stage_status(stage2)}**"])
        if m2_commit:
            lines.append(f"- Latest main (M2): `{m2_commit}`")

    lines.extend(["", "---", "", f"**Overall: {'PASS' if overall_passed else 'FAIL'}**", f"- Mode: `{mode}`", ""])
    return TwoStageReport(overall_passed=overall_passed, markdown="\n".join(lines), stage1=stage1, stage2=stage2)


def _write_report(path: Path | None, markdown: str) -> None:
    if path is None:
        print(markdown)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="perfgate", description="Compare vLLM-HUST leaderboard artifacts for performance gate checks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare = subparsers.add_parser("compare", help="Run one-stage B1 vs M1 comparison.")
    compare.add_argument("--current", required=True)
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--fork-point")
    compare.add_argument("--report-file")
    compare.add_argument("--expected-spec-id")
    compare.add_argument("--mode", choices=["report", "enforce"], default="report")

    compare2 = subparsers.add_parser("compare2", help="Run two-stage performance gate comparison.")
    compare2.add_argument("--stage1-current", required=True)
    compare2.add_argument("--stage1-baseline", required=True)
    compare2.add_argument("--stage2-current")
    compare2.add_argument("--stage2-baseline")
    compare2.add_argument("--fork-point")
    compare2.add_argument("--m2-commit")
    compare2.add_argument("--stage2-rebase-conflict", action="store_true")
    compare2.add_argument("--stage2-rebase-conflict-file")
    compare2.add_argument("--stage2-skipped", action="store_true")
    compare2.add_argument("--stage2-skip-reason")
    compare2.add_argument("--stage2-not-run", action="store_true")
    compare2.add_argument("--stage2-not-run-reason")
    compare2.add_argument("--report-file")
    compare2.add_argument("--expected-spec-id")
    compare2.add_argument("--mode", choices=["report", "enforce"], default="report")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "compare":
            report = generate_two_stage_report(
                stage1_current_file=args.current,
                stage1_baseline_file=args.baseline,
                fork_point=args.fork_point or "single-stage",
                m2_commit=args.fork_point or "single-stage",
                stage2_skipped=True,
                stage2_skip_reason="single-stage comparison",
                expected_spec_id=args.expected_spec_id,
                mode=args.mode,
            )
        else:
            report = generate_two_stage_report(
                stage1_current_file=args.stage1_current,
                stage1_baseline_file=args.stage1_baseline,
                fork_point=args.fork_point,
                stage2_current_file=args.stage2_current,
                stage2_baseline_file=args.stage2_baseline,
                m2_commit=args.m2_commit,
                stage2_rebase_conflict=args.stage2_rebase_conflict,
                stage2_rebase_conflict_file=args.stage2_rebase_conflict_file,
                stage2_skipped=args.stage2_skipped,
                stage2_skip_reason=args.stage2_skip_reason,
                stage2_not_run=args.stage2_not_run,
                stage2_not_run_reason=args.stage2_not_run_reason,
                expected_spec_id=args.expected_spec_id,
                mode=args.mode,
            )
        _write_report(Path(args.report_file) if args.report_file else None, report.markdown)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 2

    if args.mode == "enforce" and not report.overall_passed:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
