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
    monkeypatch.setattr(module, "pinned", lambda path: (ROOT / path).read_bytes())
    assert actual == module.build()
    assert actual["version"] == "V0.72"
    method = actual["m01_method"]
    assert method["method_version"] == "V5.27"
    assert all(value is None for value in method["current"].values())
    assert method["effective_compute_formula"] == "C = F / (k * T * 10^12)"
    assert method["efficiency_formula"] == "eta = T_reference / T * 100%"
    assert method["threshold_percent"] == 50
    assert (
        hashlib.sha256((ROOT / method["outline"]["path"]).read_bytes()).hexdigest()
        == method["outline"]["sha256"]
    )
    assert method["fair_comparison"]["b1_base_version_constraint"] is None
    assert (
        "KV quantization"
        in method["fair_comparison"]["allowed_implementation_differences"]
    )
    assert actual["comparison_contract"]["b0"] == {
        "vllm": "0.18",
        "vllm_ascend": "0.18",
        "fixed": True,
    }
    assert actual["comparison_contract"]["b1"]["base_version_constraint"] is None
    assert actual["comparison_contract"]["b1"]["require_actual_versions"] is True
    candidates = next(s for s in actual["sections"] if s["id"] == "candidates")
    assert "B0" not in json.dumps(candidates["tables"], ensure_ascii=False)
    assert "开发栈对照" in json.dumps(candidates["tables"], ensure_ascii=False)
    summary = actual["current_summary"]
    assert summary["quality"]["b0"]["correct"] == 151
    assert summary["quality"]["b1"]["candidate_correct"] == 152
    assert summary["quality"]["b1"]["candidate_run"] == "mmlu-quality-kvmat-r1"
    assert summary["performance"]["b0"] is None
    assert all(
        row["delta_vs_b0"] is None for row in summary["performance"]["b1_metrics"]
    )
    assert module.REVISION in actual["report_url"]
    sections = {s["id"]: s for s in actual["sections"]}
    inventory = sections["dataset-quality-inventory"]
    assert len(inventory["tables"]) == 6
    assert sum(len(table["rows"]) for table in inventory["tables"]) == 24
    assert [row[1] for row in inventory["tables"][-1]["rows"]] == [
        "151 / 200",
        "152 / 200",
        "149 / 200",
        "146 / 200",
    ]
    assert "mmlu-quality018-20261022" in inventory["evidence"][-1]["path"]
    assert "同题同评分器" in inventory["scope"]
    baseline = sections["baseline018"]
    assert len(baseline["tables"][0]["rows"]) == 5
    assert "开发栈对照" in baseline["scope"]
    assert "baseline018-sweprefix" in baseline["evidence"][0]["path"]
    closure = sections["reference-closure"]
    assert closure["tables"]
    assert "请求窗口核算见M01本轮判定" in closure["scope"]
    assert "3.6%" in closure["scope"]
    assert "reference-closure-20261022" in closure["evidence"][0]["path"]
    worker = sections["worker-model"]
    assert len(worker["tables"][0]["rows"]) == 6
    assert "264" in worker["scope"]
    assert "不替换正式B1" in worker["scope"]
    fenced = sections["fenced-model"]
    assert len(fenced["tables"][0]["rows"]) == 6
    assert "rank 0缺设备任务轨迹" in fenced["scope"]
    assert "M01未通过本轮验收" in fenced["scope"]
    assert "175.853432" in fenced["tables"][0]["rows"][2][1]
    historical = json.loads(
        (ROOT / "website-publication/data/midterm-v050.json").read_text()
    )
    for prior in historical["sections"]:
        # V0.71 corrects presentation identity only; all recorded numbers and
        # historical payload bytes must survive the explicit label migration.
        prior = json.loads(
            json.dumps(prior, ensure_ascii=False).replace("B0", "开发栈对照")
        )
        if prior["id"] != "m01":
            excluded = {"evidence"}
            if prior["id"] == "long-context":
                excluded.add("source_heading")
            assert {
                k: v for k, v in sections[prior["id"]].items() if k not in excluded
            } == {k: v for k, v in prior.items() if k not in excluded}
    for section in sections.values():
        assert section["evidence"]
        for locator in section["evidence"]:
            path = ROOT / locator["path"]
            assert path.exists()
            assert locator["kind"] == ("tree" if path.is_dir() else "blob")
    assert actual["kv_audit"] == historical["kv_audit"]
    assert len(sections["runtime-recovery"]["tables"][0]["rows"]) == 2
    current_long = sections["current-b1-long-context"]
    assert len(current_long["tables"][0]["rows"]) == 6
    assert "0/6" in current_long["scope"] and "6/6" in current_long["scope"]
    assert "不替换原始判分" in current_long["scope"]
    state = sections["state-commit"]
    assert [r[0] for r in state["tables"][0]["rows"]] == ["r1", "r2", "r3"]
    assert "导入失败" in state["tables"][0]["rows"][0][2]
    assert "通过" in state["tables"][0]["rows"][2][2]
    assert "不能排除完整模型跨轮状态污染" in state["scope"]
    assert "不采纳原始证据不可用的r23结果" in sections["m01"]["scope"]
    assert "未通过验收" in sections["m01"]["scope"]
    recovered = sections["model-runtime"]
    assert [row[0] for row in recovered["tables"][0]["rows"]] == ["240", "360"]
    assert all(row[1] == "200 / 0" for row in recovered["tables"][0]["rows"])
    assert recovered["tables"][0]["rows"][1][-1] == "100.000000%"
    assert "运行时恢复未完成" not in sections["runtime-recovery"]["scope"]
    publication = sections["preprocess-publication"]
    assert [row[1] for row in publication["tables"][0]["rows"][:2]] == ["64", "560"]
    assert "完整有效F和M01仍未判定" in publication["scope"]
    projections = sections["projection-references"]
    assert len(projections["tables"][0]["rows"]) == 10
    assert "图填充运算不冒充有效F" in projections["scope"]
    communication = sections["logits-collective-reference"]
    assert [row[1] for row in communication["tables"][0]["rows"]] == ["55", "21"]
    assert "不是整模型HCCL覆盖" in communication["scope"]
    verdict = sections["m01-verdict"]
    assert len(verdict["tables"][0]["rows"]) == 4
    assert "30.2443" not in verdict["scope"]
    assert "复算中 / 未判定" in verdict["scope"]
    assert "not determined" in verdict["scope_en"]
    assert len(verdict["tables_en"][0]["rows"]) == 4
    historical_m01 = sections["m01-historical"]
    assert len(historical_m01["tables"][0]["rows"]) == 3
    assert "30.2443%–30.2505%" in historical_m01["scope"]
    assert (
        "E24/R36 + KV materialization请求重计算 + Prefill Microflow FA"
        in historical_m01["scope"]
    )
    assert "不是同一轮联合成绩" in historical_m01["scope"]
    assert "不外推为未插桩服务结果" in historical_m01["scope"]
    assert "21.92%仅为诊断" in historical_m01["scope"]
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
    recovered_quality = sections["recovered-quality"]
    quality_index = json.loads(
        (
            ROOT
            / "reports/vllm-hust-midterm-20261019/quality-regression/recovered-20261021/archive.json"
        ).read_text()
    )
    result = quality_index["result"]
    assert (
        recovered_quality["tables"][0]["rows"][2][2]
        == f"{result['correct']}/{result['total']}"
    )
    assert "不冒充同环境新增性能配对" in recovered_quality["scope"]
    native = sections["native-paired-quality"]
    assert [row[2] for row in native["tables"][0]["rows"]] == ["147/200", "146/200"]
    assert [row[4] for row in native["tables"][0]["rows"]] == ["19", "22"]
    assert "不宣称全集质量或统计非劣性" in native["scope"]
    assert len(actual["kv_audit"]["runs"]) == 2
    adjacent = (
        "| A | B |\n| --- | ---: |\n| 1 | 2 |\n\n| C | D |\n| --- | --- |\n| 3 | 4 |\n"
    )
    assert len(module.tables(adjacent)) == 2
    for section in actual["sections"]:
        for table in section["tables"]:
            assert all(len(row) == len(table["headers"]) for row in table["rows"])
