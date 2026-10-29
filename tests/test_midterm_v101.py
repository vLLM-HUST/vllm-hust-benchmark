"""Current M01 must not inherit historical attainment or invent a replacement."""

import hashlib
import importlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_same_run_reanalysis_and_null_reference(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    b = importlib.import_module("build_midterm_v101")

    def read(path):
        return (ROOT / path).read_bytes()

    data = b.build("WORKTREE", read)
    raw = json.loads(read(b.SOURCE))
    current = data["wholechain_reference"]
    for arm in ("B0", "B1"):
        assert current[arm]["own_C"] == raw[arm]["ownF"] / (
            2 * raw[arm]["own_T_seconds"] * 1e12
        )
        assert current[arm]["eta_percent"] is None
        assert current[arm]["eta_target_met"] is None
    assert current["reference_C_TFLOPS_per_chip"] is None
    assert current["reference_qualified"] is False
    assert round(current["paired_C_gain_percent"], 4) == 84.9705
    assert current["new_measurements_started"] is False
    assert data["m01_method"] == data["milestone_contract"]["method"]
    assert data["m01_method"]["current"]["passed"] is None
    assert data["current_summary"]["reference"] == current
    assert data["milestone_contract"]["decision_relevant_gaps"]["m01"]
    for section in data["sections"]:
        if section["id"] in {
            "m01-verdict",
            "milestone-v15",
            "v15-remaining",
            "corrected-quality",
        }:
            assert "修订中" in section["scope"]
            assert "under revision" in section["scope_en"]
            assert not any(
                v in section["scope"] for v in ("99.9397", "54.0301", "7.1085")
            )
    old = json.loads(read("website-publication/data/midterm-v099.json"))
    for key in (
        "swe_incremental_tasks",
        "terminal_circuit_original_model",
    ):
        assert data[key] == old[key]
    for arm in ("B0", "B1"):
        row = data["paired_own_compute"]["rows"][arm]
        for key in ("F", "T_seconds", "chips", "C_TFLOPS_per_chip"):
            assert row[key] == old["paired_own_compute"]["rows"][arm][key]
        assert row["attainment"] is None
    assert data["corrected_quality"]["quality"] == old["corrected_quality"]["quality"]
    assert data["corrected_quality"]["reference"]["eta_percent"] is None
    assert raw["B1"]["eta_target_met"] is True  # preserved historical record
    assert hashlib.sha256(read(b.OLD_REPORT)).hexdigest() == old["report_sha256"]


def test_report_current_conclusions_and_historical_boundary(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    b = importlib.import_module("build_midterm_v101")
    report = (ROOT / b.REPORT).read_text()
    assert "报告版本：V1.01" in report
    assert "同轮复算：B0 3.8407、B1 7.1042 TFLOPS/芯片" in " ".join(report.split())
    assert "M01通过" not in report
    first = report.split("## 二、")[0]
    assert "参考修订中" in first
    assert "99.9397" not in first and "54.0301" not in first
    assert "历史整链经验参考（已退出当前判定）" in report
    assert "99.93970959931444%" in report
