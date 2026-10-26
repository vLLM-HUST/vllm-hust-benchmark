import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "reports/swe-native018-navidrome-20261027"
CONTROL = ROOT / "reports/terminal-cancel-async-controls-20261027"


def control_module():
    spec = importlib.util.spec_from_file_location(
        "cancel_verify", CONTROL / "verify.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_terminal_control_is_not_model_acceptance():
    assert control_module().verify()["model_task_attempts"] == 0


@pytest.mark.parametrize("case", ["model", "failure", "resource", "score", "scope"])
def test_control_rejects_inflation_with_updated_hash(tmp_path, case):
    shutil.copytree(CONTROL, tmp_path / "control")
    root = tmp_path / "control"
    value = json.loads((root / "summary.json").read_text())
    if case == "model":
        value["model_task_attempts"] = 1
    elif case == "failure":
        value["original_precontainer_failure"]["reward"] = 0
    elif case == "resource":
        value["cpu_limit"] = 2
    elif case == "score":
        value["controls"][0]["reward"] = 1
    else:
        value["full_dataset_acceptance"] = True
    raw = json.dumps(value).encode()
    (root / "summary.json").write_bytes(raw)
    review = json.loads((root / "independent-review.json").read_text())
    review["public_summary_sha256"] = hashlib.sha256(raw).hexdigest()
    (root / "independent-review.json").write_text(json.dumps(review))
    module = control_module()
    module.HERE = root
    with pytest.raises(AssertionError):
        module.verify()


def test_navidrome_public_verifier():
    spec = importlib.util.spec_from_file_location("nav_verify", FOLDER / "verify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.verify() == {
        "status": "passed",
        "attempted": 15,
        "official_resolved": 2,
        "unattempted": 716,
        "model_calls": 0,
    }


def test_v092_projection_keeps_previous_scope(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import build_midterm_v092 as b

    data = b.build("fixture", lambda p: (ROOT / p).read_bytes())
    old = json.loads((ROOT / "website-publication/data/midterm-v091.json").read_text())
    for key in (
        "wholechain_reference",
        "m01_method",
        "paired_own_compute",
        "frontierscience_designated_pair",
        "hle_designated_baseline",
        "c16_equal_request_latency",
        "terminal_model_recovery",
        "terminal_arm64_controls",
    ):
        assert data[key] == old[key]
    assert (
        hashlib.sha256((ROOT / b.OLD_REPORT).read_bytes()).hexdigest()
        == old["report_sha256"]
    )
    catalog = json.loads((ROOT / b.SOURCE).read_text())
    assert catalog["entries"][:-1] == old["swe_incremental_tasks"]["entries"]
    assert data["swe_incremental_tasks"] == catalog
    assert data["terminal_cancel_controls"]["model_calls"] == 0
    assert data["terminal_cancel_controls"]["controls"][1]["passed"] == 6
    entry = catalog["entries"][-1]
    assert entry["official_resolved"] == 0 and entry["apply_exit_code"] == 0
    assert entry["official_test_coverage"]["fail_to_pass"] == {
        "expected": 1,
        "passed": 0,
        "not_passed": 1,
        "missing": 0,
    }
    section = next(
        s for s in data["sections"] if s["id"] == "swe-single-agent-observation"
    )
    assert (
        len(section["tables"][0]["rows"]) == len(section["tables_en"][0]["rows"]) == 15
    )
    assert "26 passed and one failed" in section["scope_en"]
    assert "不代替官方目标测试名评分" in section["scope"]
    assert "目标包编译失败" in section["scope"]
    remaining = next(s for s in data["sections"] if s["id"] == "v15-remaining")
    for key in ("tables", "tables_en"):
        row = next(
            r for t in remaining[key] for r in t["rows"] if r[0] == "SWE-bench-Pro"
        )
        assert "716" in row[1] and "15/731" in row[1]
    assert "原reward=null" in (ROOT / b.REPORT).read_text()
    assert "99.9397%" in (ROOT / b.REPORT).read_text()
