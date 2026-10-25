import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_corrected_quality_preserves_failure_and_missing_reference(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import build_midterm_v080 as builder

    data = json.loads(builder.OUTPUT.read_text())
    assert data == builder.build(
        data["source_commit"], lambda path: (ROOT / path).read_bytes()
    )
    quality = data["corrected_quality"]
    assert quality["quality"]["current_corrected_B1"]["correct"] == 156
    assert (
        quality["quality"]["current_corrected_B1"]["original_controller_receipt"]
        == "failed; preserved"
    )
    assert quality["quality"]["corrected_KV_reference"]["correct"] == 143
    assert quality["quality"]["corrected_KV_reference"]["quality_gate_passed"] is False
    for name in ("eta_percent", "C_ref_TFLOPS_per_chip", "T_ref_seconds", "attainment"):
        assert quality["reference"][name] is None
    assert quality["reference"]["existing_B1_timing_used_as_denominator"] is False
    assert quality["reference"]["shared_implementation_allowed"] is True
    assert "/root/" not in json.dumps(quality)
    assert data["current_summary"]["quality"]["b1"]["candidate_correct"] == 156
    assert data["paired_own_compute"]["eta_percent"] is None
