import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_projection_preserves_prior_task_results(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import build_midterm_v085 as builder

    def read(path):
        if path == builder.REPORT:
            path = "reports/vllm-hust-midterm-20261019/中期测试报告-v085.md"
        return (ROOT / path).read_bytes()

    data = builder.build("fixture", read)
    old = json.loads((ROOT / "website-publication/data/midterm-v084.json").read_text())
    for key in (
        "wholechain_reference",
        "m01_method",
        "frontierscience_designated_pair",
        "hle_designated_baseline",
        "swe_single_agent",
        "paired_own_compute",
    ):
        assert data[key] == old[key]
    value = data["c16_equal_request_latency"]
    assert value["new_inference_requests"] == 0
    assert value["arms"][0]["records"] == 408
    assert value["arms"][1]["records"] == 1296
    assert value["descriptive_reduction_percent"]["ttft"] > 5
    assert value["descriptive_reduction_percent"]["tpot"] > 5
    assert len({s["id"] for s in data["sections"]}) == len(data["sections"])
    section = next(s for s in data["sections"] if s["id"] == "c16-request-latency")
    assert section["tables"][0]["rows"] == section["tables_en"][0]["rows"]
    assert "质量/SLO" in section["scope"]


def test_formal_report_values_and_no_false_new_scores(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import build_midterm_v085 as builder

    value = json.loads((ROOT / builder.SOURCE).read_text())
    generated = builder.render_report((ROOT / builder.OLD_REPORT).read_text(), value)
    report = (
        ROOT / "reports/vllm-hust-midterm-20261019/中期测试报告-v085.md"
    ).read_text()
    assert "报告版本：V0.85" in report
    for row in builder.table_rows(value):
        for cell in row[1:3]:
            assert cell in generated and cell in report
    assert "官方评分0/1" in report
    assert "同轮计数" in report


def test_v084_snapshot_is_byte_preserved():
    old = json.loads((ROOT / "website-publication/data/midterm-v084.json").read_text())
    snapshot = ROOT / "reports/vllm-hust-midterm-20261019/中期测试报告-v084.md"
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == old["report_sha256"]
