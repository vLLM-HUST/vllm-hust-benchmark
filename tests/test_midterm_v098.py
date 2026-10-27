"""Keep compilation, empty-patch and positive-control failures distinct."""

import importlib
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
b = importlib.import_module("build_midterm_v098")


def read(path):
    if path == b.REPORT:
        path = "reports/vllm-hust-midterm-20261019/中期测试报告-v098.md"
    return (ROOT / path).read_bytes()


def test_projection_preserves_prior_evidence():
    value = b.build("WORKTREE", read)
    old = b.previous.build(
        "WORKTREE", lambda p: read(b.OLD_REPORT if p == b.REPORT else p)
    )
    catalog = value["swe_incremental_tasks"]
    assert catalog["entries"][:21] == old["swe_incremental_tasks"]["entries"]
    assert (
        catalog["attempted_unique_tasks"],
        catalog["recorded_official_resolved"],
        catalog["unattempted_tasks"],
    ) == (24, 3, 707)
    a = value["swe_same_answer_recovery_aggregate"]
    assert a["recovery_denominator"] == 24
    assert a["positives_after_disclosed_recovery"] == 4
    changed = {
        "version",
        "source_commit",
        "report_sha256",
        "report_url",
        "swe_incremental_tasks",
        "swe_same_answer_recovery_aggregate",
        "sections",
    }
    for key in old:
        if key not in changed:
            assert value[key] == old[key], key
    for section in old["sections"]:
        if section["id"] not in {
            "swe-single-agent-observation",
            "swe-test-source-recovery",
            "v15-remaining",
        }:
            assert section == next(
                s for s in value["sections"] if s["id"] == section["id"]
            )
    assert len({s["id"] for s in value["sections"]}) == len(value["sections"])


def test_scope_and_bilingual_remaining():
    value = b.build("WORKTREE", read)
    a, v, t = value["swe_incremental_tasks"]["entries"][-3:]
    assert a["raw_test_coverage"]["target_tests_executed"] is False
    assert v["request_audit"]["attempts"] == 20 and v["model_calls"] == 19
    assert v["patch_bytes"] == 0 and v["apply_exit_code"] == 128
    assert v["patch_application_attested"] is False
    assert v["application_state_capture_incomplete"] is True
    assert t["raw_test_observation"]["target_tests_executed"] is True
    assert t["official_test_coverage"]["fail_to_pass"]["missing"] == 0
    r = next(s for s in value["sections"] if s["id"] == "v15-remaining")
    for key in ("tables", "tables_en"):
        rows = {row[0]: row for row in r[key][0]["rows"]}
        assert all(x in rows["SWE-bench-Pro"][1] for x in ("24/731", "3/24", "4/24"))
        assert "707" in rows["SWE-bench-Pro"][2]
        assert "3/89" in rows["TerminalBench 2.1"][1]
        assert "break-filter" in rows["TerminalBench 2.1"][2]
    controls = value["terminal_break_filter_controls"]
    assert controls["original_r1"]["reward"] is None
    assert controls["r2"]["controls"]["oracle"]["reward"] == 0
    assert controls["boundary"]["model_task_score"] is None
    assert controls["boundary"]["new_model_task_attempts"] == 0


def test_generated_report_and_paths():
    expected = b.render_report(
        read(b.OLD_REPORT).decode(),
        json.loads(read(b.SOURCE)),
        json.loads(read(b.previous.SOURCE)),
    )

    def normalize(text):
        return re.sub(r"-{3,}", "---", re.sub(r"\s+", "", text))

    assert normalize(expected) == normalize(read(b.REPORT).decode())
    for section in b.build("WORKTREE", read)["sections"]:
        for item in section.get("evidence", []):
            if item["kind"] == "blob":
                assert (ROOT / item["path"]).is_file()
    assert "报告版本：V0.97" in read(b.OLD_REPORT).decode()


def test_public_evidence_recomputes_offline():
    for folder in (b.FOLDER, b.TERMINAL):
        spec = importlib.util.spec_from_file_location(
            "verify_" + Path(folder).name, ROOT / folder / "verify.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.verify()["status"] == "passed"
