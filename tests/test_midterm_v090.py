import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "reports/swe-native018-continuation-20261026"
TERMINAL = ROOT / "reports/terminal-openssl-arm64-controls-20261026"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_continuation_verifier():
    result = load(FOLDER / "verify.py", "continuation_verify").verify()
    assert result["status"] == "passed"
    assert result["attempted"] + result["unattempted"] == 731


def test_terminal_controls_are_not_model_acceptance():
    result = load(TERMINAL / "verify.py", "terminal_verify").verify()
    assert result["control_trials"] == 2 and result["model_task_attempts"] == 0


@pytest.mark.parametrize(
    "field", ["model_calls", "model_task_attempts", "full_dataset_acceptance"]
)
def test_terminal_rejects_inflated_coverage(tmp_path, field):
    module = load(TERMINAL / "verify.py", "terminal_negative")
    data = json.loads((TERMINAL / "summary.json").read_text())
    data[field] = 1
    raw = json.dumps(data).encode()
    review = json.loads((TERMINAL / "independent-review.json").read_text())
    review["summary_sha256"] = hashlib.sha256(raw).hexdigest()
    (tmp_path / "summary.json").write_bytes(raw)
    (tmp_path / "independent-review.json").write_text(json.dumps(review))
    module.HERE = tmp_path
    with pytest.raises(AssertionError):
        module.verify()


def test_v090_current_projection_preserves_history(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import build_midterm_v090 as b

    snapshot = "reports/vllm-hust-midterm-20261019/中期测试报告-v090.md"
    data = b.build(
        "fixture",
        lambda path: (ROOT / (snapshot if path == b.REPORT else path)).read_bytes(),
    )
    old = json.loads((ROOT / "website-publication/data/midterm-v089.json").read_text())
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
    catalog = json.loads((ROOT / b.SOURCE).read_text())
    assert data["swe_incremental_tasks"] == catalog
    report = b.render_report(
        (ROOT / b.OLD_REPORT).read_text(),
        catalog,
        json.loads((ROOT / b.previous.SOURCE).read_text()),
    )
    actual = (ROOT / snapshot).read_text()
    assert "报告版本：V0.90" in actual
    assert f"{catalog['attempted_unique_tasks']}/731" in actual
    assert "目标包编译失败" in report and "模型任务覆盖仍为0/89" in report
    assert "99.9397%" in report and "报告版本：V0.90" in report
    section = next(
        s for s in data["sections"] if s["id"] == "swe-single-agent-observation"
    )
    assert "not test counts" in section["scope_en"]
    assert len(section["tables"][0]["rows"]) == catalog["attempted_unique_tasks"]
    control = next(s for s in data["sections"] if s["id"] == "terminal-arm64-controls")
    assert control["tables_en"][0]["rows"][0][1] == "0 / 6"
    assert len({s["id"] for s in data["sections"]}) == len(data["sections"])
