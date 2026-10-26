import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "reports/swe-native018-batch-20261026"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_catalog_retains_all_outcomes_and_unique_denominator():
    builder = module("batch_catalog", FOLDER / "derive_catalog.py")
    data = builder.build()
    assert data == json.loads((FOLDER / "catalog.json").read_text())
    assert data["attempted_unique_tasks"] + data["unattempted_tasks"] == 731
    assert len({item["task_identity_sha256"] for item in data["entries"]}) == len(
        data["entries"]
    )
    assert data["application_evidence_limited_tasks"] == 1
    entry = next(
        item for item in data["entries"] if item["run_id"] == "swe018-next-049-r1"
    )
    assert entry["confirmed_patch_application_failure"] is True
    assert entry["apply_exit_code"] == 1 and entry["patch_applied"] is False
    assert entry["patch_application_evidence_limited"] is False
    assert entry["official_resolved"] == 0
    assert data["recorded_official_resolved"] == sum(
        item["official_resolved"] for item in data["entries"]
    )


def test_projection_preserves_non_swe_metrics_and_updates_both_languages(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import build_midterm_v087 as builder

    def read(path):
        if path == builder.REPORT:
            path = "reports/vllm-hust-midterm-20261019/中期测试报告-v087.md"
        return (ROOT / path).read_bytes()

    data = builder.build("fixture", read)
    old = json.loads((ROOT / "website-publication/data/midterm-v086.json").read_text())
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
    assert len({section["id"] for section in data["sections"]}) == len(data["sections"])
    value = data["swe_incremental_tasks"]
    section = next(
        s for s in data["sections"] if s["id"] == "swe-single-agent-observation"
    )
    assert section["tables"][0]["rows"] == builder.rows(value)
    assert section["tables_en"][0]["rows"] == builder.rows(value, True)
    assert "补丁应用失败" in section["tables"][0]["rows"][3][5]
    assert "Patch application failed" in section["tables_en"][0]["rows"][3][5]
    remaining = next(s for s in data["sections"] if s["id"] == "v15-remaining")
    for key in ("tables", "tables_en"):
        row = next(
            row for row in remaining[key][0]["rows"] if row[0] == "SWE-bench-Pro"
        )
        assert f"{value['attempted_unique_tasks']}/731" in row[1]
        assert "128/128/122" in row[2]


def test_original_v086_snapshot_and_report_generation(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import build_midterm_v087 as builder

    old = json.loads((ROOT / "website-publication/data/midterm-v086.json").read_text())
    snapshot = (ROOT / builder.OLD_REPORT).read_bytes()
    assert hashlib.sha256(snapshot).hexdigest() == old["report_sha256"]
    value = json.loads((ROOT / builder.SOURCE).read_text())
    report = (
        ROOT / "reports/vllm-hust-midterm-20261019/中期测试报告-v087.md"
    ).read_text()
    generated = builder.render_report(snapshot.decode(), value)
    assert "报告版本：V0.87" in generated
    assert "报告版本：V0.87" in report
    assert "99.9397%" in report and "同轮计数" in report
    assert "补丁应用失败" in report and "未采集git apply退出码" in report
    assert "测试前（实际待评分）" in report
    assert "应用后及测试后" not in report
    for row in builder.rows(value):
        assert row[3] in report and row[3] in generated


@pytest.mark.parametrize(
    "field", ["apply_exit_code", "patch_applied", "official_result", "released"]
)
def test_semantic_corruption_rejected_even_with_updated_file_hash(tmp_path, field):
    import shutil

    builder = module("batch_catalog_mutation", FOLDER / "derive_catalog.py")
    inputs = json.loads((FOLDER / "inputs.json").read_text())
    needed = [builder.PRIOR]
    prior = json.loads((ROOT / builder.PRIOR).read_text())
    needed += [entry["source"] for entry in prior["entries"]]
    needed += [
        "reports/swe-native018-batch-20261026/" + entry["summary"]
        for entry in inputs["observations"]
    ]
    for name in needed:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    folder = tmp_path / "reports/swe-native018-batch-20261026"
    item = inputs["observations"][0]
    path = folder / item["summary"]
    value = json.loads(path.read_text())
    if field == "released":
        value["release"]["all_devices_released"] = False
    elif field == "official_result":
        value["grading"]["official_result"] = not value["grading"]["official_result"]
    elif field == "patch_applied":
        value["grading"]["patch_applied"] = not value["grading"]["patch_applied"]
    else:
        value["grading"]["apply_exit_code"] = 0
    path.write_text(json.dumps(value))
    item["summary_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (folder / "inputs.json").write_text(json.dumps(inputs))
    with pytest.raises(AssertionError):
        builder.build(tmp_path, folder)
