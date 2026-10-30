import importlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_archived_v102_report_binds_actual_outcomes(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    b = importlib.import_module("build_midterm_v102")
    archived = b.BASE + "/中期测试报告-v102.md"
    data = b.build(
        "WORKTREE",
        lambda path: (ROOT / (archived if path == b.REPORT else path)).read_bytes(),
    )
    assert data["version"] == "V1.02"
    assert data["m01_method"] == data["milestone_contract"]["method"]
    ref = data["wholechain_reference"]
    assert ref["reference_qualified"] is False and ref["B1"]["eta_percent"] is None
    assert ref["model_diagnostic"]["graph_recomputation"][0]["efficiency_percent"] > 37
    assert "witness" not in " ".join(ref["pending_evidence"])
    result = data["acceptance_completion"]
    assert result["continuous"]["paired_continuous_stability_pass"] is False
    assert [a["correct"] for a in result["out256_quality"]["arms"]] == [21, 19]
    assert data["sections"][0]["id"] == "acceptance-completion"
    for language in ("scope", "scope_en"):
        assert data["sections"][0][language]
    old = json.loads((ROOT / "website-publication/data/midterm-v101.json").read_text())
    for key in (
        "corrected_quality",
        "swe_incremental_tasks",
        "terminal_circuit_original_model",
    ):
        if key == "corrected_quality":
            assert data[key]["quality"] == old[key]["quality"]
        else:
            assert data[key] == old[key]
    report = (ROOT / archived).read_text()
    assert "报告版本：V1.02" in report
    assert "锁页" in report and "21/200" in report and "37.2302%" in report
    assert "工况参考能力强度与整链见证；替代C参未确定" not in report
    assert "当前合成压力配置实测失败" in report
    # Updating the current M06 row must not overwrite the archived comparison.
    prior_report = (ROOT / b.BASE / "中期测试报告-v101.md").read_text()
    start, end = "<!-- BEGIN ALIGNED_MMLU -->", "<!-- END ALIGNED_MMLU -->"
    assert (
        report.split(start)[1].split(end)[0]
        == prior_report.split(start)[1].split(end)[0]
    )
