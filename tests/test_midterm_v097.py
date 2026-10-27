"""Verify V097 projections preserve historical evidence and scoring identities."""

import importlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
b = importlib.import_module("build_midterm_v097")


def read(path):
    return (ROOT / path).read_bytes()


def test_projection_and_original_scopes():
    value = b.build("WORKTREE", read)
    old = b.previous.build(
        "WORKTREE", lambda p: read(b.OLD_REPORT if p == b.REPORT else p)
    )
    assert value["version"] == "V0.97"
    catalog = value["swe_incremental_tasks"]
    assert catalog["entries"][:18] == old["swe_incremental_tasks"]["entries"]
    assert catalog["attempted_unique_tasks"] == 21
    assert catalog["recorded_official_resolved"] == 3
    assert catalog["unattempted_tasks"] == 710
    aggregate = value["swe_same_answer_recovery_aggregate"]
    assert aggregate["recovery_denominator"] == 21
    assert aggregate["positives_after_disclosed_recovery"] == 4
    assert aggregate["additional_Agent_attempts_for_recovery"] == 0
    assert value["terminal_cython_original_model"]["reward"] is None
    assert value["terminal_cython_reconstructed_recovery"]["reward"] == 0
    assert (
        value["terminal_cython_reconstructed_recovery"]["additional_model_attempts"]
        == 0
    )
    assert value["paired_own_compute"] == old["paired_own_compute"]
    assert (
        value["protonmail_dependency_recovery"] == old["protonmail_dependency_recovery"]
    )
    assert value["terminal_cython_single_trace"] == old["terminal_cython_single_trace"]
    for section in old["sections"]:
        if section["id"] not in {
            "swe-single-agent-observation",
            "swe-test-source-recovery",
            "v15-remaining",
            "terminal-cython-controls",
        } and b.terminal_previous.scope() not in section.get("scope", ""):
            assert section == next(
                s for s in value["sections"] if s["id"] == section["id"]
            )
    assert len({s["id"] for s in value["sections"]}) == len(value["sections"])


def test_remaining_and_translated_tables():
    value = b.build("WORKTREE", read)
    section = next(s for s in value["sections"] if s["id"] == "v15-remaining")
    for key in ("tables", "tables_en"):
        rows = {row[0]: row for row in section[key][0]["rows"]}
        assert "21/731" in rows["SWE-bench-Pro"][1]
        assert "3/21" in rows["SWE-bench-Pro"][1]
        assert "4/21" in rows["SWE-bench-Pro"][1]
        assert "710" in rows["SWE-bench-Pro"][2]
        assert "3/89" in rows["TerminalBench 2.1"][1]
        assert "86" in rows["TerminalBench 2.1"][2]
        assert "null" in rows["TerminalBench 2.1"][1]
        assert "10/11" in rows["TerminalBench 2.1"][1]
    swe = next(
        s for s in value["sections"] if s["id"] == "swe-single-agent-observation"
    )
    assert len(swe["tables"][0]["rows"]) == len(swe["tables_en"][0]["rows"]) == 21


def test_report_is_generated_and_evidence_paths_exist():
    expected = b.render_report(
        read(b.OLD_REPORT).decode(),
        json.loads(read(b.SOURCE)),
        json.loads(read(b.previous.SOURCE)),
    )

    def normalize(text):
        return re.sub(r"-{3,}", "---", re.sub(r"\s+", "", text))

    assert normalize(read(b.REPORT).decode()) == normalize(expected)
    value = b.build("WORKTREE", read)
    for section in value["sections"]:
        for item in section.get("evidence", []):
            if item["kind"] == "blob":
                assert (ROOT / item["path"]).is_file(), item["path"]
    assert "报告版本：V0.96" in read(b.OLD_REPORT).decode()
