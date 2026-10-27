import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_publication_identity_and_unchanged_experimental_results(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import build_midterm_v076 as builder

    current = json.loads(builder.OUTPUT.read_text())
    previous = json.loads(
        (ROOT / "website-publication/data/midterm-v075.json").read_text()
    )
    assert current["version"] == "V0.76"
    assert current["source_commit"] == builder.REVISION
    if (
        subprocess.run(
            ["git", "cat-file", "-e", f"{builder.REVISION}:{builder.REPORT}"],
            cwd=ROOT,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    ):
        assert current == builder.build()
        assert (
            current["report_sha256"]
            == hashlib.sha256(builder.pinned(builder.REPORT)).hexdigest()
        )
    method = current["m01_method"]
    assert "C / C_reference" in method["efficiency_formula"]
    assert all(value is None for value in method["current"].values())
    for key in ("current_summary", "comparison_contract", "kv_audit"):
        assert current[key] == previous[key]
    old_sections = {section["id"]: section for section in previous["sections"]}
    for section in current["sections"]:
        old = old_sections[section["id"]]
        if section["id"] == "milestone-v15":
            for key in ("tables", "tables_en"):
                assert section[key][0]["rows"][1:] == old[key][0]["rows"][1:]
                assert section[key][0]["rows"][0][2:] == old[key][0]["rows"][0][2:]
        else:
            assert section["tables"] == old["tables"]
            assert section.get("tables_en") == old.get("tables_en")
        if section["id"] == "m01-verdict":
            assert "η=C/C参" in section["scope"]
            assert "eta=C/C_reference" in section["scope_en"]
