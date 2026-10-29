"""Verify V099 scopes without rerunning any experiment."""

import hashlib
import importlib
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
b = importlib.import_module("build_midterm_v099")


def read(path):
    if path == b.REPORT:
        path = path.replace(".md", "-v099.md")
    return (ROOT / path).read_bytes()


def test_prior_structured_results_remain_unchanged():
    value = b.build("WORKTREE", read)
    old = b.previous.build(
        "WORKTREE", lambda p: read(b.OLD_REPORT if p == b.REPORT else p)
    )
    assert (
        value["swe_incremental_tasks"]["entries"][:24]
        == old["swe_incremental_tasks"]["entries"]
    )
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
        current = next(s for s in value["sections"] if s["id"] == section["id"])
        if section["id"] in {
            "swe-single-agent-observation",
            "swe-test-source-recovery",
            "v15-remaining",
        }:
            continue
        for key in section:
            if key not in {"scope", "scope_en"}:
                assert current[key] == section[key], (section["id"], key)
        if not section["id"].startswith("terminal-"):
            assert current == section
    assert len({s["id"] for s in value["sections"]}) == len(value["sections"])
    assert (
        hashlib.sha256(read(b.OLD_REPORT)).hexdigest()
        == "eedc306a432639536c593adf34a15d347f0cf5bf63bbf068b55f17d7b7944353"  # pragma: allowlist secret
    )


def test_current_counts_and_test_execution_boundaries():
    value = b.build("WORKTREE", read)
    catalog = value["swe_incremental_tasks"]
    assert (
        catalog["attempted_unique_tasks"],
        catalog["recorded_official_resolved"],
        catalog["unattempted_tasks"],
    ) == (26, 3, 705)
    aggregate = value["swe_same_answer_recovery_aggregate"]
    assert aggregate["recovery_denominator"] == 26
    assert aggregate["positives_after_disclosed_recovery"] == 4
    assert aggregate["additional_Agent_attempts_for_recovery"] == 0
    library, nav = catalog["entries"][-2:]
    assert library["raw_test_observation"]["target_tests_executed"]
    assert (
        library["raw_test_observation"]["original_image_pytest_reruns"][
            "observed_total_reruns"
        ]
        == 9
    )
    assert library["official_test_coverage"]["fail_to_pass"]["missing"] == 0
    assert nav["raw_test_coverage"]["target_tests_executed"] is False
    assert nav["raw_test_coverage"]["target_test_assertions_failed"] is False
    assert nav["official_test_coverage"]["fail_to_pass"]["missing"] == 2
    remaining = next(s for s in value["sections"] if s["id"] == "v15-remaining")
    for key in ("tables", "tables_en"):
        rows = {r[0]: r for r in remaining[key][0]["rows"]}
        assert all(x in rows["SWE-bench-Pro"][1] for x in ("26/731", "3/26", "4/26"))
        assert "705" in rows["SWE-bench-Pro"][2]
        assert "4/89" in rows["TerminalBench 2.1"][1]
        assert "85" in rows["TerminalBench 2.1"][2]


def test_circuit_http_accounting_and_null_are_separate():
    value = b.build("WORKTREE", read)
    trial = value["terminal_circuit_original_model"]
    accounting = value["terminal_circuit_request_accounting"]
    for key in ("verifier", "reward", "CTRF"):
        assert trial[key] is None
    assert (trial["HTTP_requests"], trial["HTTP_success"], trial["HTTP_rejected"]) == (
        11,
        10,
        1,
    )
    assert (
        trial["trajectory_agent_steps"],
        trial["Harbor_n_episodes"],
        trial["completed_API_latency_entries"],
    ) == (9, 10, 9)
    assert accounting["output_length_fallback"] == 1
    assert accounting["usage_delta"] == {
        "prompt_tokens": 5449,
        "completion_tokens": 4096,
    }
    assert trial["answer_boundary"]["answer_write_command_found"] is False
    assert trial["answer_boundary"]["original_final_answer_file_SHA_unknown"]
    assert trial["answer_boundary"]["new_execution"] == 0
    assert (trial["Terminal_attempts"], trial["Terminal_unattempted"]) == (4, 85)
    controls = value["terminal_circuit_controls"]
    assert controls["results"]["nop"]["passed"] == 2
    for english in (False, True):
        assert "nop 2/3" in b.circuit_note(english)
        assert "nop 1/3" not in b.circuit_note(english)
    assert controls["model_Agent_attempts"] == 0
    assert controls["results"]["oracle"]["reward"] == 1
    assert trial["official_scored"] is False


def test_diagnostic_is_not_replacement_control_or_browser_snapshot():
    value = b.build("WORKTREE", read)
    d = value["terminal_break_filter_byte_diagnostic"]
    assert (d["filter"]["original_output_bytes"], d["filter"]["filtered_bytes"]) == (
        41,
        20,
    )
    assert d["filter"]["browser_session_started"] is False
    assert d["filter"]["original_control_score_recomputed"] is False
    assert d["boundaries"]["new_model_task_attempts"] == 0
    assert d["boundaries"]["original_positive_control_reward"] == 0
    assert (
        value["terminal_break_filter_controls"]["r2"]["controls"]["oracle"]["reward"]
        == 0
    )


def test_generated_report_and_all_evidence_paths():
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


def test_new_public_directories_verify_offline():
    for folder in (b.FOLDER, b.CIRCUIT, b.CONTROLS, b.DIAGNOSTIC):
        spec = importlib.util.spec_from_file_location(
            "verify_" + Path(folder).name, ROOT / folder / "verify.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.verify()["status"] == "passed"
