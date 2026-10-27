"""Offline V100 projection and evidence-boundary regression checks."""

import hashlib
import importlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
b = importlib.import_module("build_midterm_v100")


def read(path):
    return (ROOT / path).read_bytes()


def payload():
    return b.build("WORKTREE", read)


def test_frozen_snapshots_and_unrelated_results_unchanged():
    value = payload()
    old = b.previous.build(
        "WORKTREE", lambda p: read(b.OLD_REPORT if p == b.REPORT else p)
    )
    assert (
        value["swe_incremental_tasks"]["entries"][:26]
        == old["swe_incremental_tasks"]["entries"]
    )
    changed = {
        "version",
        "source_commit",
        "report_sha256",
        "report_url",
        "sections",
        "swe_incremental_tasks",
        "swe_same_answer_recovery_aggregate",
        "m01_method",
        "wholechain_reference",
        "current_summary",
    }
    for key in old:
        if key not in changed:
            assert value[key] == old[key], key
    assert value["m01_method_at_v099"] == old["m01_method"]
    assert value["wholechain_reference_at_v099"] == old["wholechain_reference"]
    for key in (
        "dataset",
        "model",
        "precision",
        "quality",
        "performance",
        "b1_runtime",
    ):
        assert value["current_summary"][key] == old["current_summary"][key]
    assert (
        hashlib.sha256(read(b.OLD_REPORT)).hexdigest()
        == "d264ad42873713ada8c3e7f03cb935b2870ebe4f0abb529dcc27dc7eb8430afe"  # pragma: allowlist secret
    )
    assert len({s["id"] for s in value["sections"]}) == len(value["sections"])


def test_swe_counts_and_missing_tests_are_not_assertion_failures():
    value = payload()
    catalog = value["swe_incremental_tasks"]
    assert (
        catalog["attempted_unique_tasks"],
        catalog["recorded_official_resolved"],
        catalog["unattempted_tasks"],
    ) == (29, 4, 702)
    aggregate = value["swe_same_answer_recovery_aggregate"]
    assert aggregate["recovery_denominator"] == 29
    assert aggregate["positives_after_disclosed_recovery"] == 5
    assert aggregate["additional_Agent_attempts_for_recovery"] == 0
    proton, qute, tuta = catalog["entries"][-3:]
    assert proton["official_test_coverage"]["fail_to_pass"]["missing"] == 4
    assert qute["official_test_coverage"]["fail_to_pass"] == dict(
        expected=1, passed=1, not_passed=0, missing=0
    )
    assert qute["official_test_coverage"]["pass_to_pass"]["passed"] == 9
    assert tuta["official_test_coverage"]["fail_to_pass"]["missing"] == 107
    assert not tuta["raw_test_coverage"]["target_test_assertions_failed"]
    for english in (False, True):
        text = b.note(english)
        assert "TypeError" in text and "No tests found" in text and "XIO" in text
        assert "esbuild-linux-arm64" in text
    remaining = next(s for s in value["sections"] if s["id"] == "v15-remaining")
    for key in ("tables", "tables_en"):
        rows = {r[0]: r for r in remaining[key][0]["rows"]}
        assert all(x in rows["SWE-bench-Pro"][1] for x in ("29/731", "4/29", "5/29"))
        assert "702" in rows["SWE-bench-Pro"][2]
        assert "6/89" in rows["TerminalBench 2.1"][1]
        assert "83" in rows["TerminalBench 2.1"][2]


def test_terminal_null_accounting_and_artifact_boundaries():
    value = payload()
    for name, counts, budget in (
        ("cobol", (23, 22, 23), 900),
        ("codeimage", (35, 34, 35), 1200),
    ):
        trial = value[f"terminal_{name}_original_model"]
        assert (
            trial["HTTP_requests"],
            trial["trajectory_agent_steps"],
            trial["Harbor_n_episodes"],
        ) == counts
        assert trial["completed_API_latency_entries"] == counts[1]
        assert trial["usage_delta"] == {"prompt_tokens": 0, "completion_tokens": 0}
        assert (
            trial["budget"]["agent_timeout_sec"]
            == trial["budget"]["verifier_timeout_sec"]
            == budget
        )
        assert (
            trial["budget"]["harbor_llm_layer_attempts"]
            == trial["budget"]["terminus_query_layer_attempts"]
            == 3
        )
        assert all(trial[k] is None for k in ("verifier", "reward", "CTRF"))
        assert not trial["official_scored"] and trial["output_length_fallback"] == 0
    cobol = value["terminal_cobol_original_model"]["answer_boundary"]
    assert cobol["derived_script_bytes"] == 6098 and cobol["original_file_SHA"] is None
    assert cobol["original_file_mode"] is None and not cobol["recovery_started"]
    image = value["terminal_codeimage_original_model"]
    assert image["answer_boundary"]["original_Agent_autonomously_attempted_OCR"]
    assert not image["answer_boundary"]["operator_injected_OCR_or_answer"]
    assert not image["answer_boundary"][
        "required_answer_artifact_written_in_recorded_commands"
    ]
    assert image["corrected_observation"]["actual_steps_field"] == 34
    count = value["terminal_count_controls"]
    assert count["status"] == "positive_control_failed_dependency_transport"
    assert not count["cause"]["algorithm_or_ARM64_functional_failure_established"]
    assert not count["Agent_admission"]


def test_m01_current_strength_null_and_historical_values_preserved():
    value = payload()
    method = value["m01_method"]
    old = value["m01_method_at_v099"]
    assert method["current"]["passed"] is None
    assert method["current"]["historical_frozen_90_result"] is True
    for key, number in old["current"].items():
        if key != "passed":
            assert method["current"][key] == number
    assert method["paired_own_compute"]["attainment"] is None
    assert method["paired_own_compute"]["historical_empirical_attainment"] is True
    for reference in (
        value["wholechain_reference"],
        value["current_summary"]["reference"],
        method["wholechain_reference"],
        method["paired_own_compute"]["wholechain_reference"],
    ):
        assert reference["reference_qualified"] is None
        assert reference["historical_reference_qualified"] is True
        assert reference["B1"]["eta_target_met"] is None
        assert reference["B1"]["historical_eta_target_met"] is True
        assert reference["B1"]["eta_percent"] == 99.93970959931444
        assert reference["reference_C_TFLOPS_per_chip"] == 7.108506511466548
    strength = value["m01_reference_strength"]
    assert len(strength["remaining_evidence_gaps"]) == 3
    assert strength["eligible_replacement_reference"] is None
    assert not strength["new_measurements_authorized"]
    assert strength["new_quality_windows"] == 0
    assert strength["prospective_candidate_budget"]["wall_limit_seconds"] is None
    assert strength["shared_implementation_allowed"]
    assert not strength["different_algorithm_or_global_optimality_proof_required"]
    for section in value["sections"]:
        for key in ("scope", "scope_en"):
            assert "独立整链参考与配对验收完成。" not in section.get(key, "")
            assert (
                "Independent whole-chain reference and paired assessment are complete."
                not in section.get(key, "")
            )
    verdict = next(s for s in value["sections"] if s["id"] == "m01-verdict")
    assert "历史" in verdict["tables"][0]["headers"][-1]
    assert "Historical" in verdict["tables_en"][0]["headers"][-1]


def load_verifier(folder):
    spec = importlib.util.spec_from_file_location(
        "verify_" + Path(folder).name, ROOT / folder / "verify.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "folder",
    [
        b.FOLDER,
        b.COBOL,
        b.CODEIMAGE,
        b.COBOL_CONTROLS,
        b.CODEIMAGE_CONTROLS,
        b.COUNT,
        b.M01,
    ],
)
def test_new_directories_verify_without_execution(folder):
    assert load_verifier(folder).verify()["status"] == "passed"


@pytest.mark.parametrize("folder", [b.COBOL, b.CODEIMAGE, b.COUNT])
def test_public_byte_mutation_is_rejected(folder, tmp_path, monkeypatch):
    module = load_verifier(folder)
    for path in (ROOT / folder).iterdir():
        if path.is_file():
            (tmp_path / path.name).write_bytes(path.read_bytes())
    raw = (tmp_path / "summary.json").read_bytes()
    (tmp_path / "summary.json").write_bytes(raw + b" ")
    monkeypatch.setattr(module, "HERE", tmp_path)
    with pytest.raises(AssertionError):
        module.verify()


def test_wrong_snapshot_count_rejected():
    bad = json.loads(read(b.SOURCE))
    bad["attempted_unique_tasks"] = 30

    def mutated(path):
        return json.dumps(bad).encode() if path == b.SOURCE else read(path)

    with pytest.raises(AssertionError):
        b.build("WORKTREE", mutated)


def test_report_projection_and_evidence_paths():
    expected = b.render_report(
        read(b.OLD_REPORT).decode(),
        json.loads(read(b.SOURCE)),
        json.loads(read(b.previous.SOURCE)),
    )

    def normalize(s):
        return re.sub(r"-{3,}", "---", re.sub(r"\s+", "", s))

    assert normalize(expected) == normalize(read(b.REPORT).decode())
    for section in payload()["sections"]:
        for evidence in section["evidence"]:
            if evidence["kind"] == "blob":
                assert (ROOT / evidence["path"]).is_file()
    assert "M01通过" not in read(b.REPORT).decode()
