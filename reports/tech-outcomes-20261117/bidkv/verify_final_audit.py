"""Validate the final CPU custody audit and retain diagnostic limitations."""

import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def verify():
    root = Path(__file__).resolve().parent / "final-audit-20261116"
    conclusions = read(root / "audit-conclusions.json")
    assert conclusions["standard_formal_scope_complete"] is True
    assert conclusions["formal_solver_actual"] == 1344
    assert conclusions["same_fixed_native_judge_attempts"] == 720
    assert (
        conclusions["formal_policy_calls"]
        == conclusions["formal_native_preemptions"]
        == 0
    )
    assert conclusions["formal_mechanism_benefit_demonstrated"] is False
    assert conclusions["quality_equivalence_demonstrated"] is False
    assert conclusions["new_model_experiments"] == 0
    assert conclusions["tuning_performed_in_closeout"] is False
    assert conclusions["used_91"] is False
    evidence = read(root / "remote-original-evidence.json")
    status = evidence["status"]
    assert status["phase"] == "completed"
    assert status["model_generation_requests"] == 0
    assert len(status["children"]) == 18
    assert all(
        c["status"] == "completed" and c["actual_returncode"] == 0
        for c in status["children"]
    )
    identity = read(root / "current-input-grader-hash-check.json")["result"]
    assert identity["all_expected_hashes_match"] is True
    assert identity["current_reference_tool_files_verified"] == 98
    assert identity["current_frozen_plan_files_verified"] == 46
    assert identity["all_generation_requests"] == 0
    assert all(x["sha256"] == x["expected_sha256"] for x in identity["files"])
    evo = read(root / "diagnostics/evo02/external-cpu-overlap.json")
    assert evo["possible_shared_cpu_interference"] is True
    assert evo["wall_overlap_seconds"] > 16
    assert evo["all_runs_retained"] is True
    pilot = read(root / "diagnostics/mmlu8/study.json")
    assert pilot["planned"] == 32 and pilot["actual_controller_returncode"] == 0
    assert pilot["closed_pilot_audit_passed"] is True
    assert pilot["pilot_decision_recomputed"]["run_reverse_confirmation"] is False
    assert pilot["pilot_decision_recomputed"] == pilot["stored_pilot_decision"]
    assert (
        "prospective_quality_triage_not_met"
        in pilot["stored_pilot_decision"]["reasons"]
    )
    source = root / "raw_audit81.py.source"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == status["source_sha256"]
    return {
        "current_controlled_sse_replay_receipts": 2064,
        "original_grader_recomputations": 4,
        "actual_cpu_children_returncode_zero": 18,
        "current_frozen_input_and_grader_files": 144,
        "new_model_experiments": 0,
        "evo_cpu_interference_retained": True,
        "mmlu_pilot_failed_quality_gate_retained": True,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
