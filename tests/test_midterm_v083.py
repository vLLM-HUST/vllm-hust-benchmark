import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_current_method_is_consistent_and_history_is_preserved(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import build_midterm_v083 as builder

    data = json.loads(builder.OUTPUT.read_text())
    assert data == builder.build(
        data["source_commit"], lambda p: (ROOT / p).read_bytes()
    )
    method = data["m01_method"]
    assert method == data["milestone_contract"]["method"]
    assert method["threshold_percent"] == 90
    assert method["current"]["passed"] is True
    assert method["current_measurement"]["strict_uninstrumented"] is False
    assert method["identity"]["formal_indicator_amendment"] is False
    assert method["identity"]["original_v15"]["threshold_percent"] == 50
    assert data["milestone_contract"]["decision_relevant_gaps"]["m01"] == []
    assert data["milestone_contract"]["cost_method"]["passed"] is None
    assert data["frontierscience_designated_pair"]["quality_passed"] is None
    prior = json.loads(
        (ROOT / "website-publication/data/midterm-v082.json").read_text()
    )
    for key in (
        "wholechain_reference",
        "paired_own_compute",
        "frontierscience_designated_pair",
    ):
        assert data[key] == prior[key]
    history = data["method_history"]["snapshots"][1]
    raw = (ROOT / builder.REPORT_DIR / history["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == history["sha256"]
    assert json.loads(raw)["current"]["efficiency_percent"] is None
    sections = {s["id"]: s for s in data["sections"]}
    for key in ("tables", "tables_en"):
        assert "90%" in sections["milestone-v15"][key][0]["rows"][0][1]
    for section in data["sections"]:
        for evidence in section.get("evidence", []):
            assert not evidence.get("path", "").endswith(
                "milestone-v15/method-status.json"
            )
    assert "not M06" in sections["resources"]["scope_en"]


def test_outline_bytes_and_original_v15_are_bound():
    d = ROOT / "reports/vllm-hust-midterm-20261019"
    assert (
        hashlib.sha256((d / "test-outline-v530.docx").read_bytes()).hexdigest()
        == "756525d3b872cec9641ee045f6b66e730809763f201cb6860c43699c14ee8f05"  # pragma: allowlist secret
    )
    assert (
        hashlib.sha256((d / "milestone-v15/project4-v1.5.txt").read_bytes()).hexdigest()
        == "7f9352a85ec8025731b37b57362d80bfc48f8204ae9d1877eba521377267fa24"  # pragma: allowlist secret
    )


def test_published_source_has_full_commit_identity():
    data = json.loads((ROOT / "website-publication/data/midterm-v083.json").read_text())
    assert len(data["source_commit"]) == 40
    assert f"/blob/{data['source_commit']}/" in data["report_url"]
