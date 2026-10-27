"""Check same-attempt recovery, original history and unaffected acceptance evidence."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
b = importlib.import_module("build_midterm_v095")


def read(path):
    if path == b.REPORT:
        path = "reports/vllm-hust-midterm-20261019/中期测试报告-v095.md"
    return (ROOT / path).read_bytes()


def test_projection():
    data = b.build("WORKTREE", read)
    assert data["version"] == "V0.95"
    assert data["swe_incremental_tasks"]["attempted_unique_tasks"] == 17
    assert data["swe_incremental_tasks"]["recorded_official_resolved"] == 2
    assert (
        data["swe_same_answer_recovery_aggregate"]["positives_after_disclosed_recovery"]
        == 3
    )
    assert data["swe_same_answer_recovery_aggregate"]["unattempted_tasks"] == 714
    for ident in ("swe-single-agent-observation", "swe-test-source-recovery"):
        section = next(s for s in data["sections"] if s["id"] == ident)
        for key in ("scope", "scope_en"):
            assert "3/17" in section[key] and "2/17" in section[key]
    remaining = next(s for s in data["sections"] if s["id"] == "v15-remaining")
    for key in ("tables", "tables_en"):
        rows = {row[0]: row for t in remaining[key] for row in t["rows"]}
        assert "3/17" in rows["SWE-bench-Pro"][1]
        assert "2/89" in rows["TerminalBench 2.1"][1]
        assert "KeyError" in rows["TerminalBench 2.1"][1]
    old = b.previous.build(
        "WORKTREE", lambda p: read(b.OLD_REPORT if p == b.REPORT else p)
    )
    changed = {
        "swe-single-agent-observation",
        "swe-test-source-recovery",
        "terminal-cython-controls",
        "v15-remaining",
    }
    for section in old["sections"]:
        if section["id"] not in changed:
            assert section == next(
                s for s in data["sections"] if s["id"] == section["id"]
            )
    for key in (
        "paired_own_compute",
        "swe_incremental_tasks",
        "protonmail_test_source_recovery",
        "terminal_cython_controls",
    ):
        if key in old:
            assert old[key] == data[key]


def verifier():
    spec = importlib.util.spec_from_file_location(
        "proton_recovery_verify", ROOT / Path(b.RECOVERY).parent / "verify.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_verifier():
    assert verifier().verify()["status"] == "passed"


@pytest.mark.parametrize(
    "path,value",
    [
        (("Agent_attempts_added",), 1),
        (("controller_exit",), 0),
        (("grading_process_exit",), 1),
        (("patch_apply_exit",), 1),
        (("aggregate_distinction", "attempted_tasks"), 18),
        (("aggregate_distinction", "positives_after_disclosed_recovery"), 4),
        (("original_result", "official"), True),
        (("derived_dependency_recovery_result", "F2P", "missing"), 1),
    ],
)
def test_reject_invented_recovery(path, value):
    folder = ROOT / Path(b.RECOVERY).parent
    args = [
        json.loads((folder / name).read_text())
        for name in (
            "summary.json",
            "dependency-build.json",
            "receipt-projections.json",
            "review.json",
        )
    ]
    s = copy.deepcopy(args[0])
    target = s
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(AssertionError):
        verifier().validate(s, *args[1:])
