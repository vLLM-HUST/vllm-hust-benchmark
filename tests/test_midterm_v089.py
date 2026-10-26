import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "reports/swe-native018-next-repository-20261026"


@pytest.mark.parametrize(
    "mutation", ["duplicate", "release", "score", "application", "budget"]
)
def test_catalog_rejects_semantic_corruption_after_rehash(tmp_path, mutation):
    spec = importlib.util.spec_from_file_location(
        "next_catalog_negative", FOLDER / "derive_catalog.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / FOLDER.relative_to(ROOT)
    shutil.copytree(FOLDER, source)
    prior = json.loads((ROOT / module.PRIOR).read_text())
    for path in [module.PRIOR, *[item["source"] for item in prior["entries"]]]:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, target)
    inputs_path = source / "inputs.json"
    inputs = json.loads(inputs_path.read_text())
    item = inputs["observations"][0]
    summary_path = source / item["summary"]
    value = json.loads(summary_path.read_text())
    if mutation == "duplicate":
        inputs["observations"].append(item.copy())
    elif mutation == "release":
        value["release"]["all_devices_released"] = False
    elif mutation == "score":
        value["recorded_official_resolved"] = 1 - value["recorded_official_resolved"]
    elif mutation == "application":
        value["grading"]["patch_applied"] = not value["grading"]["patch_applied"]
    else:
        value["budget"]["step_limit"] = 101
    summary_path.write_text(json.dumps(value))
    for observation in inputs["observations"]:
        if observation["summary"] == item["summary"]:
            observation["summary_sha256"] = hashlib.sha256(
                summary_path.read_bytes()
            ).hexdigest()
    inputs_path.write_text(json.dumps(inputs))
    with pytest.raises(AssertionError):
        module.build(tmp_path, source)


def test_public_next_repository_verifier():
    spec = importlib.util.spec_from_file_location("next_verify", FOLDER / "verify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = module.verify()
    assert value["status"] == "passed"
    assert value["attempted"] + value["unattempted"] == 731


def test_v089_preserves_completed_metrics_and_historical_report(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import build_midterm_v089 as b

    snapshot = "reports/vllm-hust-midterm-20261019/中期测试报告-v089.md"
    data = b.build(
        "fixture",
        lambda path: (ROOT / (snapshot if path == b.REPORT else path)).read_bytes(),
    )
    old = json.loads((ROOT / "website-publication/data/midterm-v088.json").read_text())
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
    assert (
        hashlib.sha256((ROOT / b.OLD_REPORT).read_bytes()).hexdigest()
        == old["report_sha256"]
    )
    value = json.loads((ROOT / b.SOURCE).read_text())
    assert data["swe_incremental_tasks"] == value
    section = next(
        s for s in data["sections"] if s["id"] == "swe-single-agent-observation"
    )
    count = value["attempted_unique_tasks"]
    for key in ("tables", "tables_en"):
        assert len(section[key][0]["rows"]) == count
    assert f"{count}/731" in section["scope"] and f"{count}/731" in section["scope_en"]
    generated = b.render_report(
        (ROOT / b.OLD_REPORT).read_text(),
        value,
        json.loads((ROOT / b.previous.SOURCE).read_text()),
    )
    current = (ROOT / snapshot).read_text()
    assert b.previous.summary(value) in current
    assert "319.777575" in current
    assert "99.9397%" in generated and "319.777575" in generated
    assert "438" in generated and "解析缺失1" in generated
    assert "报告版本：V0.89" in generated
    assert "262.062638" in generated and "14通过、2失败" in generated
    assert "2个包初始化失败" in generated
    assert "14 passed, 2 failed" in section["scope_en"]
    assert len({s["id"] for s in data["sections"]}) == len(data["sections"])
