import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "reports/terminal-cancel-async-partial-recovery-20261027"


def verifier(root=FOLDER):
    spec = importlib.util.spec_from_file_location(
        "partial_verify", FOLDER / "verify.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = root
    return module


def test_partial_verifier():
    assert verifier().verify() == dict(
        status="passed",
        original_reward=None,
        partial_artifact_reward=0,
        passed=1,
        tests=6,
        new_model_calls=0,
    )


@pytest.mark.parametrize(
    "case",
    [
        "null",
        "complete",
        "calls",
        "retry",
        "original_sha",
        "scope",
        "score",
        "mode",
        "budget",
        "resource",
    ],
)
def test_rejects_semantic_inflation(tmp_path, case):
    root = tmp_path / "evidence"
    shutil.copytree(FOLDER, root)
    s = json.loads((root / "summary.json").read_text())
    if case == "null":
        s["original_attempt"]["reward"] = 0
    elif case == "complete":
        s["original_attempt"]["task_complete"] = True
    elif case == "calls":
        s["original_attempt"]["HTTP200"] = 18
    elif case == "retry":
        s["recovery"]["additional_Agent_attempts"] = 1
    elif case == "original_sha":
        s["artifact_provenance"]["original_filesystem_sha256"] = s[
            "artifact_provenance"
        ]["derived_sha256"]
    elif case == "scope":
        s["coverage"]["completed_answer_scored_tasks"] = 2
    elif case == "score":
        s["recovery"]["reward"] = 1
    elif case == "mode":
        s["artifact_provenance"]["original_mode"] = "0644"
    elif case == "budget":
        s["original_attempt"]["budget"]["turns"] = 200
    else:
        s["recovery"]["cpus"] = 2
    (root / "summary.json").write_text(json.dumps(s))
    with pytest.raises(AssertionError):
        verifier(root).verify()


def test_v093_projection_preserves_all_previous_results(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import build_midterm_v093 as b

    archived = "reports/vllm-hust-midterm-20261019/中期测试报告-v093.md"
    data = b.build(
        "fixture", lambda p: (ROOT / (archived if p == b.REPORT else p)).read_bytes()
    )
    old = json.loads((ROOT / "website-publication/data/midterm-v092.json").read_text())
    for key in (
        "wholechain_reference",
        "m01_method",
        "paired_own_compute",
        "frontierscience_designated_pair",
        "hle_designated_baseline",
        "c16_equal_request_latency",
        "terminal_model_recovery",
        "terminal_arm64_controls",
        "terminal_cancel_controls",
        "swe_incremental_tasks",
    ):
        assert data[key] == old[key]
    assert (
        hashlib.sha256((ROOT / b.OLD_REPORT).read_bytes()).hexdigest()
        == old["report_sha256"]
    )
    assert data["terminal_cancel_partial_recovery"] == json.loads(
        (ROOT / b.SOURCE).read_text()
    )
    section = next(
        s for s in data["sections"] if s["id"] == "terminal-cancel-partial-recovery"
    )
    assert "source" in section["scope_en"] and "源码推断" in section["scope"]
    remaining = next(s for s in data["sections"] if s["id"] == "v15-remaining")
    for key in ("tables", "tables_en"):
        row = next(
            r for t in remaining[key] for r in t["rows"] if r[0] == "TerminalBench 2.1"
        )
        assert "87" in row[1] and "2/89" in row[1]
        assert "null" in row[1]
    report = (ROOT / archived).read_text()
    assert "另88题未执行" not in report
    assert "99.9397%" in report and "原reward=null" in report
