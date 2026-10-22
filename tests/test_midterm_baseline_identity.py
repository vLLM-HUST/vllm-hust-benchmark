import json
from pathlib import Path

import pytest


REPORT = Path(__file__).resolve().parents[1] / "reports/vllm-hust-midterm-20261019"


@pytest.mark.parametrize("concurrency", [1, 2, 4, 8, 16])
def test_reused_018_windows_retain_identity_and_arithmetic(concurrency):
    directory = REPORT / "baseline018-sweprefix" / f"C{concurrency}-900s"
    meta = json.loads((directory / "server-metadata.json").read_text())
    result = json.loads((directory / "workload-result/summary.json").read_text())
    assert meta["runtime"]["vllm_release"] == "0.18.0"
    assert meta["runtime"]["vllm_ascend_release"] == "0.18.0"
    assert (
        meta["runtime"]["vllm_applied_commit"] != meta["runtime"]["vllm_release_commit"]
    )
    assert meta["workload"]["protocol"] == "swe-prefix-reuse/v1"
    assert meta["workload"]["concurrency"] == concurrency
    assert result["valid"] is True
    assert result["failed_requests"] == 0
    assert result["measurement_seconds"] == 900
    rate = result["observed_output_tokens_in_window"] / result["measurement_seconds"]
    assert result["output_tokens_per_second"] == pytest.approx(rate)
    text = (REPORT / "中期测试报告.md").read_text()
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in text.splitlines()
        if line.startswith("| C")
    ]
    assert [
        f"C{concurrency}",
        str(result["observed_output_tokens_in_window"]),
        str(rate),
        "0",
    ] in rows


def test_report_distinguishes_development_comparison_and_designated_baseline():
    text = " ".join((REPORT / "中期测试报告.md").read_text().split())
    assert "指定基线为vLLM 0.18 / vLLM-Ascend 0.18" in text
    assert "不代表0.18指定基线" in text
    assert "原生基线证据，10月12日至15日" not in text
    assert "不替代MMLU-Pro质量测量或M01运算计数" in text
    assert text.count("0.18对应工况判定待完成") == 2
