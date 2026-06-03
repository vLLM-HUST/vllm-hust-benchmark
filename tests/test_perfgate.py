from __future__ import annotations

import json
from pathlib import Path

from vllm_hust_benchmark import perfgate


def _write_run_leaderboard(path: Path, *, throughput: float, ttft: float, tbt: float) -> None:
    path.write_text(
        json.dumps(
            {
                "engine": "vllm-hust",
                "metrics": {
                    "throughput_tps": throughput,
                    "ttft_ms": ttft,
                    "tbt_ms": tbt,
                    "error_rate": 0.0,
                },
                "same_spec": {
                    "spec_id": "perfgate-ascend-qwen25-05b-910b3",
                    "resolved_spec_hash": "abc123",
                },
            }
        ),
        encoding="utf-8",
    )


def test_compare_benchmark_results_passes_when_throughput_improves_and_latency_drops(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    _write_run_leaderboard(baseline, throughput=100.0, ttft=50.0, tbt=10.0)
    _write_run_leaderboard(current, throughput=103.0, ttft=45.0, tbt=9.5)

    result = perfgate.compare_benchmark_results(current, baseline)

    assert result.passed is True
    assert result.metrics["throughput_tps"].passed is True
    assert result.metrics["ttft_ms"].delta_percent == -10.0
    assert result.metrics["tbt_ms"].direction == "lower_is_better"


def test_compare_benchmark_results_fails_on_regression(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    _write_run_leaderboard(baseline, throughput=100.0, ttft=50.0, tbt=10.0)
    _write_run_leaderboard(current, throughput=99.0, ttft=52.0, tbt=10.5)

    result = perfgate.compare_benchmark_results(current, baseline)

    assert result.passed is False
    assert result.metrics["throughput_tps"].passed is False
    assert result.metrics["ttft_ms"].passed is False
    assert result.metrics["tbt_ms"].passed is False


def test_generate_two_stage_report_marks_rebase_conflict_as_overall_fail(tmp_path: Path) -> None:
    stage1_baseline = tmp_path / "m1.json"
    stage1_current = tmp_path / "b1.json"
    conflict_file = tmp_path / "conflict.txt"
    _write_run_leaderboard(stage1_baseline, throughput=100.0, ttft=50.0, tbt=10.0)
    _write_run_leaderboard(stage1_current, throughput=101.0, ttft=49.0, tbt=9.0)
    conflict_file.write_text("CONFLICT (content): Merge conflict in vllm/engine.py\n", encoding="utf-8")

    report = perfgate.generate_two_stage_report(
        stage1_current_file=stage1_current,
        stage1_baseline_file=stage1_baseline,
        fork_point="aaa11111",
        stage2_rebase_conflict=True,
        stage2_rebase_conflict_file=conflict_file,
        mode="enforce",
    )

    assert report.overall_passed is False
    assert "Stage 2: FAIL — rebase conflict" in report.markdown
    assert "CONFLICT (content)" in report.markdown
    assert "**Overall: FAIL**" in report.markdown


def test_generate_two_stage_report_uses_stage1_when_stage2_skipped(tmp_path: Path) -> None:
    stage1_baseline = tmp_path / "m1.json"
    stage1_current = tmp_path / "b1.json"
    _write_run_leaderboard(stage1_baseline, throughput=100.0, ttft=50.0, tbt=10.0)
    _write_run_leaderboard(stage1_current, throughput=101.0, ttft=49.0, tbt=9.0)

    report = perfgate.generate_two_stage_report(
        stage1_current_file=stage1_current,
        stage1_baseline_file=stage1_baseline,
        fork_point="aaa11111",
        stage2_skipped=True,
        stage2_skip_reason="fork-point is already latest main",
        mode="enforce",
    )

    assert report.overall_passed is True
    assert "Stage 2: SKIPPED" in report.markdown
    assert "fork-point is already latest main" in report.markdown
    assert "**Overall: PASS**" in report.markdown


def test_compare2_cli_writes_report_and_returns_failure_for_stage2_regression(tmp_path: Path) -> None:
    stage1_baseline = tmp_path / "m1.json"
    stage1_current = tmp_path / "b1.json"
    stage2_baseline = tmp_path / "m2.json"
    stage2_current = tmp_path / "b1prime.json"
    report_file = tmp_path / "report.md"
    _write_run_leaderboard(stage1_baseline, throughput=100.0, ttft=50.0, tbt=10.0)
    _write_run_leaderboard(stage1_current, throughput=101.0, ttft=49.0, tbt=9.0)
    _write_run_leaderboard(stage2_baseline, throughput=100.0, ttft=50.0, tbt=10.0)
    _write_run_leaderboard(stage2_current, throughput=90.0, ttft=55.0, tbt=11.0)

    exit_code = perfgate.main(
        [
            "compare2",
            "--stage1-current", str(stage1_current),
            "--stage1-baseline", str(stage1_baseline),
            "--stage2-current", str(stage2_current),
            "--stage2-baseline", str(stage2_baseline),
            "--fork-point", "aaa11111",
            "--m2-commit", "bbb22222",
            "--report-file", str(report_file),
            "--mode", "enforce",
        ]
    )

    assert exit_code == 1
    report = report_file.read_text(encoding="utf-8")
    assert "Stage 1: PASS" in report
    assert "Stage 2: FAIL" in report
    assert "**Overall: FAIL**" in report


def test_compare_benchmark_results_rejects_same_spec_mismatch(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    _write_run_leaderboard(baseline, throughput=100.0, ttft=50.0, tbt=10.0)
    _write_run_leaderboard(current, throughput=103.0, ttft=45.0, tbt=9.5)
    payload = json.loads(current.read_text(encoding="utf-8"))
    payload["same_spec"]["resolved_spec_hash"] = "different"
    current.write_text(json.dumps(payload), encoding="utf-8")

    try:
        perfgate.compare_benchmark_results(current, baseline)
    except ValueError as error:
        assert "same_spec.resolved_spec_hash mismatch" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected same-spec mismatch to fail")


def test_compare_benchmark_results_requires_non_null_tbt(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    _write_run_leaderboard(baseline, throughput=100.0, ttft=50.0, tbt=10.0)
    _write_run_leaderboard(current, throughput=103.0, ttft=45.0, tbt=9.5)
    payload = json.loads(current.read_text(encoding="utf-8"))
    payload["metrics"]["tbt_ms"] = None
    current.write_text(json.dumps(payload), encoding="utf-8")

    try:
        perfgate.compare_benchmark_results(current, baseline)
    except ValueError as error:
        assert "tbt_ms" in str(error)
        assert "non-null" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected null tbt_ms to fail")


def test_compare2_cli_report_mode_does_not_block_on_failure(tmp_path: Path) -> None:
    stage1_baseline = tmp_path / "m1.json"
    stage1_current = tmp_path / "b1.json"
    report_file = tmp_path / "report.md"
    _write_run_leaderboard(stage1_baseline, throughput=100.0, ttft=50.0, tbt=10.0)
    _write_run_leaderboard(stage1_current, throughput=90.0, ttft=55.0, tbt=11.0)

    exit_code = perfgate.main(
        [
            "compare2",
            "--stage1-current", str(stage1_current),
            "--stage1-baseline", str(stage1_baseline),
            "--stage2-skipped",
            "--report-file", str(report_file),
            "--mode", "report",
        ]
    )

    assert exit_code == 0
    assert "**Overall: FAIL**" in report_file.read_text(encoding="utf-8")
