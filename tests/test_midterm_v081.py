import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_wholechain_reference_and_projection(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import build_midterm_v081 as builder

    data = json.loads(builder.OUTPUT.read_text())

    def read(path):
        if path == builder.REPORT:
            path = builder.REPORT_DIR + "/中期测试报告-v081.md"
        return (ROOT / path).read_bytes()

    assert data == builder.build(data["source_commit"], read)
    result = data["wholechain_reference"]
    cref = statistics.median(
        w["ownF"] / (2 * w["synchronized_seconds"] * 1e12)
        for w in result["windows"][:3]
    )
    assert cref == result["reference_C_TFLOPS_per_chip"]
    assert result["capacity_checks"]["repeatability_pass"]
    assert result["capacity_checks"]["H_capacity_pass"]
    assert result["all_frozen_applicability_checks_pass"]
    for arm in ("B0", "B1"):
        r = result[arm]
        assert r["T_reference_seconds"] == r["ownF"] / (2 * cref * 1e12)
        assert r["eta_percent"] == 100 * r["own_C"] / cref
        assert r["eta_target_met"] == (r["eta_percent"] >= 90)
    assert result["B1"]["eta_target_met"] and not result["B0"]["eta_target_met"]
    pair = data["paired_own_compute"]
    assert pair["status"] == "own_F_T_C_and_independent_reference_closed"
    assert pair["attainment"] is True
    assert pair["reference_seconds"] == result["B1"]["T_reference_seconds"]
    assert pair["reference_tflops_per_chip"] == cref
    assert pair["rows"]["B0"]["attainment"] is False
    assert data["corrected_quality"]["reference"]["attainment"] is True
    assert result["method"]["shared_code_with_B1"]
    assert not result["method"]["old_B1_time_used_as_denominator"]
    assert not result["method"]["overhead_subtracted"]
    serialized = json.dumps(result)
    for forbidden in ("/root/", "case_ids", "prompt_sha256", "request_id"):
        assert forbidden not in serialized
    assert (
        result["evidence"]["final_result_sha256"]
        == result["evidence"]["independent_result_sha256"]
    )
    assert (
        data["corrected_quality"]["quality"]["corrected_KV_reference"]["correct"] == 143
    )
    sections = {s["id"]: s for s in data["sections"]}
    for name in ("m01-verdict", "v15-remaining", "corrected-quality"):
        assert "99.9397%" in sections[name]["scope"]
        assert "remain incomplete" not in sections[name]["scope_en"]
