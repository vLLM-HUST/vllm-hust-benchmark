import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_publication_rebuilds_from_pinned_report_and_verified_archives():
    spec = importlib.util.spec_from_file_location(
        "midterm", ROOT / "scripts/build_midterm_publication.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = json.loads(module.OUTPUT.read_text())
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
