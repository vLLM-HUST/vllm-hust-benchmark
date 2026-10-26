import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "reports/swe-native018-repository-20261026"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_repository_catalog_preserves_prior_outcomes():
    builder = module("repository_catalog", FOLDER / "derive_catalog.py")
    data = builder.build()
    assert data == json.loads((FOLDER / "catalog.json").read_text())
    prior = json.loads((ROOT / builder.PRIOR).read_text())
    assert data["entries"][:9] == prior["entries"]
    assert data["attempted_unique_tasks"] == 10 and data["unattempted_tasks"] == 721
    assert data["recorded_official_resolved"] == 1
    assert data["entries"][-1]["patch_applied"] is True
    assert data["entries"][-1]["official_resolved"] == 0


def test_public_result_verifier():
    value = module("repository_verify", FOLDER / "verify.py").verify()
    assert value["status"] == "passed"


def test_v088_report_and_projection_keep_completed_metrics(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import build_midterm_v088 as b

    data = b.build("fixture", lambda p: (ROOT / p).read_bytes())
    old = json.loads((ROOT / "website-publication/data/midterm-v087.json").read_text())
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
    section = next(
        s for s in data["sections"] if s["id"] == "swe-single-agent-observation"
    )
    assert section["tables"][0]["rows"] == b.rows(value)
    assert section["tables_en"][0]["rows"] == b.rows(value, True)
    assert "10/731" in section["scope"] and "10/731" in section["scope_en"]
    snapshot = (ROOT / b.OLD_REPORT).read_bytes()
    assert hashlib.sha256(snapshot).hexdigest() == old["report_sha256"]
    generated = b.render_report(snapshot.decode(), value)
    current = (ROOT / b.REPORT).read_text()
    assert "报告版本：V0.88" in generated and "报告版本：V0.88" in current
    for text in (
        "99.9397%",
        "同轮计数",
        "225.544925",
        "437",
        "438",
        "解析缺失1",
        "ARM64",
    ):
        assert text in current
    assert len({s["id"] for s in data["sections"]}) == len(data["sections"])


@pytest.mark.parametrize("bad", ["apply", "release", "repeat"])
def test_bad_summary_cannot_be_accepted_with_updated_hash(tmp_path, bad):
    import shutil

    b = module("repository_catalog_negative", FOLDER / "derive_catalog.py")
    prior = json.loads((ROOT / b.PRIOR).read_text())
    inputs = json.loads((FOLDER / "inputs.json").read_text())
    paths = [b.PRIOR] + [x["source"] for x in prior["entries"]]
    for name in paths:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, p)
    here = tmp_path / FOLDER.relative_to(ROOT)
    here.mkdir(parents=True, exist_ok=True)
    item = inputs["observations"][0]
    v = json.loads((FOLDER / item["summary"]).read_text())
    if bad == "apply":
        v["grading"]["apply_exit_code"] = 1
    if bad == "release":
        v["release"]["all_devices_released"] = False
    if bad == "repeat":
        v["task_identity_sha256"] = prior["entries"][0]["task_identity_sha256"]
    p = here / item["summary"]
    p.write_text(json.dumps(v))
    item["summary_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    (here / "inputs.json").write_text(json.dumps(inputs))
    with pytest.raises(AssertionError):
        b.build(tmp_path, here)
