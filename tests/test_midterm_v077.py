import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_same_run_compute_projection_and_retained_results(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import build_midterm_v077 as builder

    current = json.loads(builder.OUTPUT.read_text())
    previous = json.loads(
        (ROOT / "website-publication/data/midterm-v076.json").read_text()
    )
    assert current["version"] == "V0.77"
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
    for key, value in previous["current_summary"].items():
        assert current["current_summary"][key] == value
    for key in ("comparison_contract", "kv_audit"):
        assert current[key] == previous[key]
    method = current["m01_method"]
    assert method["current"]["effective_tflops_per_chip"] == 7.1042207644080255
    assert method["current"]["matched_synchronized_seconds"] == 127.59352582134306
    assert method["current"]["matched_uninstrumented_seconds"] is None
    for key in (
        "reference_seconds",
        "reference_effective_tflops_per_chip",
        "efficiency_percent",
        "passed",
    ):
        assert method["current"][key] is None
    assert method["current_measurement"]["strict_uninstrumented"] is False
    old_sections = {section["id"]: section for section in previous["sections"]}
    for section in current["sections"]:
        if section["id"] not in {"m01-verdict", "milestone-v15", "v15-remaining"}:
            assert section == old_sections[section["id"]]
        for evidence in section["evidence"]:
            assert (ROOT / evidence["path"]).exists()
    verdict = next(s for s in current["sections"] if s["id"] == "m01-verdict")
    assert "7.1042" in verdict["scope"] and "7.1042" in verdict["scope_en"]
    assert "不是严格无插桩" in verdict["scope"]
    assert "not strictly uninstrumented" in verdict["scope_en"]
    assert "待关联" not in verdict["tables"][0]["rows"][0][2]
