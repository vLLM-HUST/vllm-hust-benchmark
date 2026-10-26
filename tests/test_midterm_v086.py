import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_incremental_projection_preserves_completed_metrics(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import build_midterm_v086 as builder

    data = builder.build("fixture", lambda path: (ROOT / path).read_bytes())
    old = json.loads((ROOT / "website-publication/data/midterm-v085.json").read_text())
    for key in (
        "wholechain_reference",
        "m01_method",
        "paired_own_compute",
        "frontierscience_designated_pair",
        "hle_designated_baseline",
        "c16_equal_request_latency",
        "swe_single_agent",
    ):
        assert data[key] == old[key]
    value = data["swe_incremental_tasks"]
    assert value["dataset_tasks"] == 731
    assert value["attempted_unique_tasks"] == len(value["entries"])
    assert value["application_evidence_limited_tasks"] == 1
    assert value["full_dataset_acceptance"] is False
    section = next(
        s for s in data["sections"] if s["id"] == "swe-single-agent-observation"
    )
    assert len(section["tables"][0]["rows"]) == value["attempted_unique_tasks"]
    assert "未获证明" in section["scope"]
    assert "unproven" in section["scope_en"]
    remaining = next(s for s in data["sections"] if s["id"] == "v15-remaining")
    for key in ("tables", "tables_en"):
        row = next(r for r in remaining[key][0]["rows"] if r[0] == "SWE-bench-Pro")
        assert f"{value['attempted_unique_tasks']}/731" in row[1]
        assert "128/128/122" in row[2]


def test_catalog_recomputes_and_has_distinct_tasks(monkeypatch):
    folder = ROOT / "reports/swe-native018-incremental-20261026"
    monkeypatch.syspath_prepend(str(folder))
    import derive_catalog

    data = json.loads((folder / "catalog.json").read_text())
    assert data == derive_catalog.build()
    assert len({x["task_identity_sha256"] for x in data["entries"]}) == len(
        data["entries"]
    )
    assert data["unattempted_tasks"] + data["attempted_unique_tasks"] == 731
    assert data["official_unresolved_grading"] == 0


def test_v085_snapshot_and_formal_report(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import build_midterm_v086 as builder

    old = json.loads((ROOT / "website-publication/data/midterm-v085.json").read_text())
    raw = (ROOT / builder.OLD_REPORT).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == old["report_sha256"]
    report = (ROOT / builder.REPORT).read_text()
    assert "报告版本：V0.86" in report
    assert "git apply退出码" in " ".join(report.split())
    assert "99.9397%" in report and "同轮计数" in report
    value = json.loads((ROOT / builder.SOURCE).read_text())
    for row in builder.rows(value):
        assert row[3] in report
