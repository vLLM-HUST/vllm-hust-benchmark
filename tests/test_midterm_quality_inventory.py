import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_quality_inventory_matches_original_tracks_and_preserves_limits():
    spec = importlib.util.spec_from_file_location(
        "midterm", ROOT / "scripts/build_midterm_publication.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text = (ROOT / module.REPORT).read_text()
    selected = module.section(
        text, "### 5.11 HLE-Verified与FrontierScience既有任务质量"
    )
    rows = module.tables(selected)[0]["rows"]
    hle = json.loads(
        (
            ROOT / "website-publication/data/hle-verified-20261015/report.json"
        ).read_text()
    )
    frontier = json.loads(
        (
            ROOT / "website-publication/data/frontierscience-20261013/report.json"
        ).read_text()
    )
    tracks = [
        hle["summaries"][name]["tracks"][name] for name in ("gold", "revision")
    ] + [frontier["summary"]["tracks"][name] for name in ("olympiad", "research")]
    assert len(rows) == len(tracks) == 4
    for row, track in zip(rows, tracks):
        assert row[1] == f"{track['successes']} / {track['total']}"
        assert row[2] == str(track["graded"])
        assert row[3] == str(track.get("unresolved", track.get("judge_unresolved")))
        assert row[4] == str(track["solver_truncated"])
        assert row[5] == "未测"
    for source in (hle, frontier):
        assert source["formal_acceptance"] is False
        assert source["paired_b1"] is None
    for limitation in ("替代裁判", "完整分母", "未测", "不能仅凭公开归档独立重评"):
        assert limitation in selected
    assert "整模型有效算量覆盖仍待补证" not in text
    publication = json.loads(module.OUTPUT.read_text())
    section = next(
        s for s in publication["sections"] if s["id"] == "dataset-quality-inventory"
    )
    assert section["tables"][0]["rows"] == rows
    assert len(section["evidence"]) == 2
    for locator in section["evidence"]:
        assert ".." not in Path(locator["path"]).parts
        assert (ROOT / locator["path"]).is_file()
