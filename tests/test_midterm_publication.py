import importlib.util
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_publication_rebuilds_from_pinned_report_and_verified_archives(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "midterm", ROOT / "scripts/build_midterm_publication.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = json.loads(module.OUTPUT.read_text())
    raw = (ROOT / module.REPORT).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == actual["report_sha256"]
    monkeypatch.setattr(module, "pinned", lambda path: raw)
    assert actual == module.build()
    assert actual["version"] == "V0.52"
    assert module.REVISION in actual["report_url"]
    sections = {s["id"]: s for s in actual["sections"]}
    historical = json.loads(
        (ROOT / "website-publication/data/midterm-v050.json").read_text()
    )
    for prior in historical["sections"]:
        if prior["id"] != "m01":
            assert {
                k: v for k, v in sections[prior["id"]].items() if k != "evidence"
            } == prior
    for section in sections.values():
        assert section["evidence"]
        for locator in section["evidence"]:
            path = ROOT / locator["path"]
            assert path.exists()
            assert locator["kind"] == ("tree" if path.is_dir() else "blob")
    assert actual["kv_audit"] == historical["kv_audit"]
    assert len(sections["runtime-recovery"]["tables"][0]["rows"]) == 2
    state = sections["state-commit"]
    assert [r[0] for r in state["tables"][0]["rows"]] == ["r1", "r2", "r3"]
    assert "导入失败" in state["tables"][0]["rows"][0][2]
    assert "通过" in state["tables"][0]["rows"][2][2]
    assert "不能排除完整模型跨轮状态污染" in state["scope"]
    assert "不采纳其吞吐或M01计量" in sections["m01"]["scope"]
    assert actual["kv_audit"]["runs"] == module.kv_audit()
    assert module.tables("| A | B |\n| --- | ---: |\n| 1 | 2 |\n") == [
        {"headers": ["A", "B"], "rows": [["1", "2"]]}
    ]
    quality = next(s for s in actual["sections"] if s["id"] == "quality")
    assert [r[1] for r in quality["tables"][0]["rows"]] == [
        "147/200",
        "149/200",
        "152/200",
    ]
    assert len(actual["kv_audit"]["runs"]) == 2
    adjacent = (
        "| A | B |\n| --- | ---: |\n| 1 | 2 |\n\n| C | D |\n| --- | --- |\n| 3 | 4 |\n"
    )
    assert len(module.tables(adjacent)) == 2
    for section in actual["sections"]:
        for table in section["tables"]:
            assert all(len(row) == len(table["headers"]) for row in table["rows"])
