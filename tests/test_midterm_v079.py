import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_own_pair_is_not_reference_attainment(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import build_midterm_v079 as builder

    data = json.loads(builder.OUTPUT.read_text())
    pair = data["paired_own_compute"]
    assert pair["eta_percent"] is pair["attainment"] is None
    assert pair["baseline_is_reference_ceiling"] is False
    assert pair["rows"]["B0"]["F"] == 1829161955258989
    assert data["version"] == "V0.79"
    revision = data["source_commit"]
    if (
        subprocess.run(
            ["git", "cat-file", "-e", f"{revision}:{builder.REPORT}"],
            cwd=ROOT,
            capture_output=True,
        ).returncode
        == 0
    ):
        assert data == builder.build(revision)
    else:
        assert (
            hashlib.sha256((ROOT / builder.REPORT).read_bytes()).hexdigest()
            == data["report_sha256"]
        ), "Shallow checkout does not contain the pinned report bytes"
        # Shallow CI still checks complete regeneration from the exact report bytes.
        assert data == builder.build(revision, lambda path: (ROOT / path).read_bytes())
    previous = json.loads(
        (ROOT / "website-publication/data/midterm-v078.json").read_text()
    )
    for key in ("current_summary", "comparison_contract", "kv_audit"):
        assert data[key] == previous[key]
    verdict = next(s for s in data["sections"] if s["id"] == "m01-verdict")
    assert "84.9705%" in verdict["scope"]
    assert "not reference attainment" in verdict["scope_en"]
    for section in data["sections"]:
        for evidence in section["evidence"]:
            assert (ROOT / evidence["path"]).exists()
