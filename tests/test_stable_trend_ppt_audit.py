import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "reports/ppt-stable-trend-audit-20260817"


def test_ppt_audit_matrix_is_exact_and_does_not_overstate_repeats() -> None:
    audit = json.loads(
        (AUDIT_DIR / "stable_trend_audit.json").read_text(encoding="utf-8")
    )
    cells = audit["cells"]

    assert audit["summary"]["cells"] == 27
    assert audit["summary"]["online_cells"] == 18
    assert audit["summary"]["offline_cells"] == 9
    assert audit["summary"]["independent_3_repeat_median_cells"] == 27
    assert audit["summary"]["single_invocation_cells"] == 0
    assert audit["summary"]["evidence_grade_counts"] == {"A": 27}
    assert len({(cell["checkpoint"], cell["workload"]) for cell in cells}) == 27
    assert all(not cell["repeat_group_present"] for cell in cells)
    assert all(not cell["canonical_aggregate_present"] for cell in cells)


def test_ppt_audit_declares_primary_metrics_and_baseline_gap() -> None:
    audit = json.loads(
        (AUDIT_DIR / "stable_trend_audit.json").read_text(encoding="utf-8")
    )

    assert audit["summary"]["strict_baseline_workloads"] == 1
    assert audit["summary"]["baseline_invocations_required"] == 24
    assert audit["primary_metric_policy"]["agent-research-online"] == {
        "metric": "ttft_ms",
        "direction": "lower",
    }
    assert audit["primary_metric_policy"]["sonnet-throughput"] == {
        "metric": "throughput_tps",
        "direction": "higher",
    }


def test_ppt_audit_csv_and_report_use_historical_health_boundary() -> None:
    with (AUDIT_DIR / "stable_trend_evidence_matrix.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    report = (AUDIT_DIR / "README.md").read_text(encoding="utf-8")

    assert len(rows) == 27
    assert all(row["independent_3_repeat_median"] == "True" for row in rows)
    assert all(row["repeat_suite_path"] for row in rows)
    assert "历史版本健康检查点" in report
    assert "不支持“9 类性能持续提升”" in report
    assert "不代表 current latest" in report
