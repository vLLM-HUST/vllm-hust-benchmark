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
    selected = module.section(text, "### 5.11 既有任务质量与配置配对")
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
        assert row[5] == "未完成"
    for source in (hle, frontier):
        assert source["formal_acceptance"] is False
        assert source["paired_b1"] is None
    for limitation in ("替代裁判", "完整分母", "未完成", "不能仅凭公开归档独立重评"):
        assert limitation in selected
    assert "整模型有效算量覆盖仍待补证" not in text
    publication = json.loads(module.OUTPUT.read_text())
    section = next(
        s for s in publication["sections"] if s["id"] == "dataset-quality-inventory"
    )
    assert section["tables"][0]["rows"] == rows
    assert len(section["evidence"]) == 12
    for locator in section["evidence"]:
        assert ".." not in Path(locator["path"]).parts
        assert (ROOT / locator["path"]).is_file()

    all_tables = module.tables(selected)
    assert len(all_tables) == 6
    matrix = json.loads(
        (ROOT / "website-publication/data/dataset-matrix-v1.json").read_text()
    )
    indexed = {r["id"]: r for r in matrix["records"]}
    full_ids = (
        "mmlu-pro-native-test-200",
        "mmlu-pro-betterscale-test-200",
        "mmlu-pro-core-attention-boundary-test-200",
        "mmlu-pro-ascend-attention-boundary-test-200",
        "mmlu-pro-controller-20261018-native",
        "mmlu-pro-controller-20261018-request-throttling-controller",
    )
    for row, ident in zip(all_tables[1]["rows"], full_ids):
        result = indexed[ident]["quality"][0]
        assert row[1] == f"{result['correct']} / {result['total']}"
    best = json.loads(
        (
            ROOT
            / "website-publication/data/betterscale-school-runtime-20261016/report.json"
        ).read_text()
    )
    source_tracks = [best["accuracy"][k]["tracks"][k] for k in ("gold", "revision")]
    source_tracks += [
        best["accuracy"]["frontier"]["tracks"][k] for k in ("olympiad", "research")
    ]
    for row, track in zip(all_tables[2]["rows"], source_tracks):
        assert row[1] == f"{track['successes']} / {track['total']}"
        assert row[2] == str(track.get("unresolved", track.get("judge_unresolved")))
    controller = (
        ROOT / "website-publication/data/request-throttling-dataset-matrix-20261018"
    )
    pairs = []
    for name in ("hle-gold", "hle-revision", "frontierscience"):
        source = json.loads((controller / (name + ".json")).read_text())
        assert source["configuration"]["core_version"] == "0.23.0"
        arms = source["quality_pair"]["arms"]
        tracks = (
            ("olympiad", "research")
            if name == "frontierscience"
            else (name.removeprefix("hle-"),)
        )
        for track in tracks:
            pairs.append(
                [
                    f"{arms[arm]['summary']['tracks'][track]['successes']} / {arms[arm]['summary']['tracks'][track]['total']}"
                    for arm in ("native", "controller")
                ]
            )
    assert [row[1:] for row in all_tables[3]["rows"]] == pairs
    combination = json.loads(
        (
            ROOT / "reports/frontierscience-combination-quality-20261022/summary.json"
        ).read_text()
    )
    for row, name in zip(all_tables[4]["rows"], ("olympiad", "research")):
        result = combination["tracks"][name]
        assert row[1:] == [
            f"{result['successes']} / {result['total']}",
            str(result["graded"]),
            str(result["judge_unresolved"]),
            str(result["solver_truncated"]),
            str(result["judge_truncated"]),
        ]
    assert section["tables"][4] == all_tables[4]
    native018 = json.loads(
        (
            ROOT / "reports/mmlu-quality018-20261022/quality018-completed-audit-r1.json"
        ).read_text()
    )
    rows018 = all_tables[5]["rows"]
    assert rows018[0][1:4] == [
        f"{native018['counts']['correct']} / {native018['counts']['total']}",
        str(native018["counts"]["failed"]),
        str(native018["counts"]["truncated"]),
    ]
    comparisons = native018["comparisons"]
    for row, result in zip(
        rows018[1:], (comparisons[1], comparisons[2], comparisons[0]), strict=True
    ):
        assert row[1:] == [
            f"{result['candidate_correct']} / 200",
            str(result["candidate_failed"]),
            str(result["candidate_truncated"]),
            f"{result['accuracy_delta_percentage_points']:+.1f}个百分点",
        ]
    assert section["tables"][5] == all_tables[5]
