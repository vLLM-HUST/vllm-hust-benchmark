"""Regression checks for new attempts and unchanged prior acceptance evidence."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_midterm_v094 as builder  # noqa: E402


def read(path):
    if path == builder.REPORT:
        path = "reports/vllm-hust-midterm-20261019/中期测试报告-v094.md"
    return (ROOT / path).read_bytes()


def test_current_projection_boundaries():
    value = builder.build("WORKTREE", read)
    assert value["version"] == "V0.94"
    assert value["swe_incremental_tasks"]["attempted_unique_tasks"] == 17
    assert value["swe_incremental_tasks"]["unattempted_tasks"] == 714
    assert value["swe_incremental_tasks"]["recorded_official_resolved"] == 2
    assert value["protonmail_test_source_recovery"]["accepted_functional_score"] is None
    assert value["terminal_cython_controls"]["control_qualification_passed"] is False
    section = next(
        x for x in value["sections"] if x["id"] == "swe-single-agent-observation"
    )
    assert len(section["tables"][0]["rows"]) == 17
    assert section["tables"][0]["rows"][-1][2] == "75成功 + 1拒绝"
    assert "未提交部分补丁" in section["tables"][0]["rows"][-1][-1]
    for key in ("scope", "scope_en"):
        assert "17/731" in section[key] and "canvas.node" in section[key]
    remaining = next(x for x in value["sections"] if x["id"] == "v15-remaining")
    for key in ("tables", "tables_en"):
        swe = next(
            row
            for table in remaining[key]
            for row in table["rows"]
            if row[0] == "SWE-bench-Pro"
        )
        assert "17/731" in swe[1] and "canvas.node" in swe[1]
        terminal = next(
            row
            for table in remaining[key]
            for row in table["rows"]
            if row[0] == "TerminalBench 2.1"
        )
        assert "2/89" in terminal[1] and "87" in terminal[1]


def test_projection_preserves_other_evidence():
    current = builder.build("WORKTREE", read)
    prior = builder.previous.build(
        "WORKTREE", lambda p: read(builder.OLD_REPORT if p == builder.REPORT else p)
    )
    changed = {"swe-single-agent-observation", "v15-remaining"}
    for old in prior["sections"]:
        if old["id"] not in changed:
            assert next(s for s in current["sections"] if s["id"] == old["id"]) == old
    for key, value in prior.items():
        if key not in {
            "version",
            "source_commit",
            "report_sha256",
            "report_url",
            "swe_incremental_tasks",
            "sections",
        }:
            assert current[key] == value


@pytest.mark.parametrize(
    "path,key,value",
    [
        (builder.SOURCE, "attempted_unique_tasks", 18),
        (builder.RECOVERY, "new_Agent_attempts", 1),
        (builder.RECOVERY, "tests_executed", 1),
        (builder.CYTHON, "new_model_task_attempts", 1),
    ],
)
def test_projection_rejects_invented_coverage(path, key, value):
    def changed(p):
        if p != path:
            return read(p)
        data = json.loads(read(p))
        data[key] = value
        return json.dumps(data).encode()

    with pytest.raises(AssertionError):
        builder.build("WORKTREE", changed)


def load_recovery_verifier():
    p = ROOT / "reports/swe-protonmail-test-source-recovery-20261027/verify.py"
    spec = importlib.util.spec_from_file_location("recovery_public", p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recovery_offline():
    assert load_recovery_verifier().verify()["status"] == "passed"


@pytest.mark.parametrize(
    "key,value",
    [
        ("accepted_functional_score", False),
        ("new_model_calls", 1),
        ("released", False),
        ("private_task_answer_patch_or_request_contents_included", True),
    ],
)
def test_recovery_rejects_overstatement(key, value):
    data = json.loads(read(builder.RECOVERY))
    data[key] = value
    with pytest.raises(AssertionError):
        load_recovery_verifier().validate(data)
