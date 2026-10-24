import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_reference_supplement_projection_preserves_performance_and_null_verdict(
    monkeypatch,
):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import build_midterm_v078 as builder

    data = json.loads(builder.OUTPUT.read_text())
    previous = json.loads(
        (ROOT / "website-publication/data/midterm-v077.json").read_text()
    )
    assert data["version"] == "V0.78" and data["source_commit"] == builder.REVISION
    for key in ("current_summary", "comparison_contract", "kv_audit"):
        assert data[key] == previous[key]
    supplement = data["reference_supplement"]
    assert supplement["totals"]["numerical_checks"] == 1402
    assert supplement["model_validation"]["passed"] == 6
    assert supplement["model_validation"]["failed"] == 26
    assert (
        supplement["reference_seconds"]
        is supplement["efficiency_percent"]
        is supplement["attainment"]
        is None
    )
    old_sections = {s["id"]: s for s in previous["sections"]}
    for row in data["sections"]:
        if row["id"] not in {"m01-verdict", "milestone-v15", "v15-remaining"}:
            assert row == old_sections[row["id"]]
        for evidence in row["evidence"]:
            assert (ROOT / evidence["path"]).exists()
    verdict = next(s for s in data["sections"] if s["id"] == "m01-verdict")
    assert "26组未通过" in verdict["scope"] and "26 fail" in verdict["scope_en"]
    assert (
        len(verdict["tables"][-1]["rows"]) == len(verdict["tables_en"][-1]["rows"]) == 4
    )
    if (
        subprocess.run(
            ["git", "cat-file", "-e", f"{builder.REVISION}:{builder.REPORT}"],
            cwd=ROOT,
            capture_output=True,
        ).returncode
        == 0
    ):
        assert data == builder.build()
        assert (
            data["report_sha256"]
            == hashlib.sha256(builder.pinned(builder.REPORT)).hexdigest()
        )
