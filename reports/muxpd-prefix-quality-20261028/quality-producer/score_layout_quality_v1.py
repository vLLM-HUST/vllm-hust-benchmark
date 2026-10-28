"""Explicit provenance bridge around the byte-identical original quality scorer.

No new answer parser, relaxed truncation rule, or replacement quality criterion.
The 24 selected cases are posterior diagnostics and cannot pass the full gate.
"""

import argparse
import copy
import json
from pathlib import Path

import prefill_collective_layout_v1 as layout
import score_quality as original
from probe_control_v1 import audit_terminal_log, inventory
from probe_run_v1 import QUALITY_SOURCE, SOURCE
from quality_control_v1 import FILES
from quality_run_v1 import (
    BUNDLE,
    LAYOUT_SHA,
    PLAN_SHA,
    file_sha,
    select_batches,
    source_manifest,
    validate_layout_admissions,
)


def historical_runs(manifest):
    runs, paths = [], []
    for label in (
        "native-quality-v2-hccl-true-01",
        "native-quality-v2-hccl-true-02",
        "raw-quality-v2-hccl-true-01",
    ):
        source = manifest["historical_runs"][label]
        path = Path(source["path"])
        if file_sha(path) != source["sha256"]:
            raise ValueError("Historical terminal run bytes changed")
        run = json.loads(path.read_text())
        if (
            not run["collection_complete"]
            or not run["shutdown_returned"]
            or run["errors"]
            or run["plan_sha256"] != PLAN_SHA
        ):
            raise ValueError("Historical reference is incomplete or uses another plan")
        for key in ("engine_args", "sources", "runtime_environment", "producer_sha256"):
            if run[key] != source[key]:
                raise ValueError(
                    "Historical provenance receipt differs from copied source evidence"
                )
        runs.append(run)
        paths.append(path)
    if [r["role"] for r in runs] != ["native", "native", "raw"]:
        raise ValueError("Expected original paired native controls and raw reference")
    if (
        runs[0]["process_id"] == runs[1]["process_id"]
        or runs[0]["started_at"] == runs[1]["started_at"]
        or runs[0]["engine_args"] != runs[1]["engine_args"]
    ):
        raise ValueError(
            "Historical native controls must remain distinct identical-config processes"
        )
    # This original provenance check is unchanged, and applies to its original
    # three runs. The deliberately changed new producer is handled explicitly.
    original.compare_provenance(runs)
    return runs, paths


def candidate_provenance(candidate_path, candidate, historical, plan, cases):
    root = candidate_path.parent.parent
    controller = json.loads((root / "controller.json").read_text())
    producer = root / "producer"
    if (
        not controller.get("passed")
        or controller.get("exit_code") != 0
        or controller.get("errors")
        or controller.get("cleanup_required")
        or controller.get("owned_members_after_cleanup") != []
        or controller.get("terminal_run_sha256") != file_sha(candidate_path)
        or not controller.get("finished_at")
    ):
        raise ValueError(
            "Candidate lacks successful terminal controller/owned-process evidence"
        )
    actual = {name: file_sha(producer / name) for name in FILES}
    if actual != controller["files"]:
        raise ValueError("Candidate producer snapshot bytes changed")
    freeze = producer / "freeze-manifest-v1.json"
    if (
        file_sha(freeze) != controller["freeze_manifest_sha256"]
        or json.loads(freeze.read_text())["files"] != actual
    ):
        raise ValueError("Candidate was not produced from reviewed frozen bytes")
    for name in FILES:
        if file_sha(BUNDLE / name) != actual[name]:
            raise ValueError("Use the candidate's frozen scoring bundle")
    if (
        candidate.get("schema") != "muxpd-layout-quality-run-v1"
        or candidate.get("profile") != "raw-prefix"
        or candidate.get("role") != "raw"
        or candidate.get("layout_mode") != "prefix"
        or candidate.get("plan_sha256") != PLAN_SHA
        or not candidate.get("passed")
        or not candidate.get("collection_complete")
        or not candidate.get("layout_restored")
        or not candidate.get("shutdown_returned")
        or candidate.get("errors")
        or candidate.get("failure_cleanup")
        or candidate.get("terminal_maintenance", {}).get("aborted_internal_ids")
        or candidate.get("producer_sha256") != actual["quality_run_v1.py"]
        or candidate.get("original_producer_sha256") != actual["run_quality.py"]
        or candidate.get("original_scorer_sha256") != actual["score_quality.py"]
        or candidate.get("source_manifest_sha256") != actual["source-manifest-v1.json"]
        or candidate.get("layout_module_sha256") != LAYOUT_SHA
    ):
        raise ValueError("Candidate lifecycle/module/producer identity failed")
    audit = audit_terminal_log(root / "process.log")
    if not audit["passed"] or audit != controller["terminal_process_log_audit"]:
        raise ValueError("Candidate terminal log failed independent audit")
    for when in ("before", "after"):
        if (
            inventory((root / f"npu-{when}.txt").read_text())
            != controller["expected_physical"]
        ):
            raise ValueError("Candidate terminal physical idle guard failed")
    raw = historical[2]
    if (
        candidate["engine_args"] != raw["engine_args"]
        or candidate["runtime_environment"] != raw["runtime_environment"]
    ):
        raise ValueError("Candidate changed original runtime/sampling environment")
    for module in ("vllm", "vllm_ascend"):
        if candidate["sources"][module] != raw["sources"][module]:
            raise ValueError("Candidate changed original core/Ascend source")
    plugin = candidate["sources"]["vllm_muxpd"]
    if (
        plugin["commit"] != SOURCE
        or plugin["status"]
        or plugin["changed_files"]
        or plugin["sha256"] != raw["sources"]["vllm_muxpd"]["sha256"]
    ):
        raise ValueError("Candidate plugin source differs from clean 6c identity")
    bridge = candidate["source_equivalence"]
    if (
        bridge["observer_source"] != SOURCE
        or bridge["quality_runtime_source"] != QUALITY_SOURCE
        or bridge["existing_runtime_diff"]
        or len(bridge["existing_runtime_files"]) != 20
    ):
        raise ValueError(
            "Candidate lacks original 20-runtime-file equivalence evidence"
        )
    for guard in (
        candidate["source_after_startup"],
        candidate["sources_after_shutdown"],
        controller["source_preflight"],
    ):
        if guard["sources"] != candidate["sources"] or guard["equivalence"] != bridge:
            raise ValueError("Source changed across candidate lifetime")
    batches = select_batches(plan, candidate["scope"])
    selected = [c for b in batches for c in b["case_ids"]]
    if (
        candidate["selected_batch_ids"] != [b["batch_id"] for b in batches]
        or controller["selected_batch_ids"] != candidate["selected_batch_ids"]
        or controller["scope"] != candidate["scope"]
        or candidate["selected_case_ids"] != selected
        or len(candidate["outputs"]) != len(selected)
        or {o["case_id"] for o in candidate["outputs"]} != set(selected)
    ):
        raise ValueError("Candidate complete batch/case scope differs")
    layout.validate_receipts(
        candidate["layout_installation"],
        LAYOUT_SHA,
        mode="prefix",
        expected_treated_cohorts=0,
        expected_completed=0,
    )
    receipt_path = candidate_path.parent / "scheduler-receipts.jsonl"
    layout_evidence = validate_layout_admissions(
        candidate["layout_final"], candidate["admissions"], receipt_path, batches, cases
    )
    if layout_evidence != candidate["layout_admission_validation"]:
        raise ValueError(
            "Candidate layout summary differs from independent receipt validation"
        )
    layout.validate_uninstall_receipts(
        candidate["layout_uninstall"], LAYOUT_SHA, mode="prefix"
    )
    return batches, {
        "original_compare_provenance_applied_to": "original native1/native2/raw only; unchanged function",
        "new_producer_intentionally_differs": True,
        "original_producer_sha256": raw["producer_sha256"],
        "candidate_producer_sha256": candidate["producer_sha256"],
        "external_prefix_module_sha256": LAYOUT_SHA,
        "old_plugin_commit": QUALITY_SOURCE,
        "candidate_plugin_commit": SOURCE,
        "original_runtime_files_unchanged": bridge,
        "core_ascend_engine_args_runtime_environment_exactly_equal": True,
        "candidate_physical_devices": controller["expected_physical"],
        "historical_logical_visible_devices": raw["runtime_environment"]["environment"][
            "ASCEND_RT_VISIBLE_DEVICES"
        ],
        "physical_device_note": "Current controller physical allocation is explicit. Equality of logical visibility does not claim equality of historical physical cards.",
        "terminal_log": audit,
        "layout_validation": layout_evidence,
        "controller_sha256": file_sha(root / "controller.json"),
    }


def restrict_outputs(run, selected):
    value = dict(run)
    value["outputs"] = [row for row in run["outputs"] if row["case_id"] in selected]
    return value


def assess(candidate_path):
    manifest = source_manifest()
    plan, cases, extract = original.load_frozen(BUNDLE / "frozen-v2")
    historical, paths = historical_runs(manifest)
    full_historical = [original.score(run, plan, cases, extract) for run in historical]
    candidate = json.loads(candidate_path.read_text())
    batches, provenance = candidate_provenance(
        candidate_path, candidate, historical, plan, cases
    )
    selected_ids = {case_id for batch in batches for case_id in batch["case_ids"]}
    selected_cases = [case for case in cases if case["case_id"] in selected_ids]
    # Projection exists only for explicit posterior subset diagnostics. Full
    # qualification calls the original score/evaluate on all original 200 rows.
    references = [
        original.score(
            restrict_outputs(run, selected_ids), plan, selected_cases, extract
        )
        for run in historical
    ]
    scored = original.score(candidate, plan, selected_cases, extract)
    coverage_plan = copy.deepcopy(plan)
    coverage_plan["batches"] = batches
    coverage = original.verify_mixed_coverage(candidate_path, candidate, coverage_plan)
    full = candidate["scope"] == "full"
    result = {
        "schema": "muxpd-layout-quality-assessment-v1",
        "scope": candidate["scope"],
        "plan_sha256": PLAN_SHA,
        "selected_batch_ids": candidate["selected_batch_ids"],
        "selected_case_count": len(selected_cases),
        "provenance_bridge": provenance,
        "original_scorer_sha256": file_sha(BUNDLE / "score_quality.py"),
        "wrapper_sha256": file_sha(__file__),
        "run_sources": [
            {"path": str(p), "sha256": file_sha(p)} for p in [*paths, candidate_path]
        ],
        "historical_full_scores": full_historical,
        "historical_completion_note": "The original native controls each have nine length-truncated answers. Original score marks these wrong/incomplete; full strict evaluation remains unable to pass its unchanged native-completeness condition with these controls.",
        "scores": [*references[:2], scored],
        "raw_reference_score": references[2],
        "mixed_coverage": coverage,
        "full_frozen_plan_passed": None,
        "full_quality_gate_attempted": full,
        "claim_scope": (
            "Original fixed200 quality rule, external prefix treatment explicitly identified."
            if full
            else "Posterior 24-case diagnostic selected from old losses; no new/full quality gate or performance claim."
        ),
    }
    if full:
        gate = original.evaluate(references[0], references[1], scored)
        result["original_answer_quality_evaluation"] = gate
        result["full_frozen_plan_passed"] = bool(
            gate["bounded_answer_quality_gate_passed"] and coverage["passed"]
        )
    else:
        stable = {
            a["case_id"]
            for a, b in zip(references[0]["rows"], references[1]["rows"], strict=True)
            if a["correct"] and b["correct"]
        }
        result["diagnostic_new_wrong_on_both_native_correct"] = [
            r["case_id"]
            for r in scored["rows"]
            if r["case_id"] in stable and not r["correct"]
        ]
        result["diagnostic_candidate_vs_native1"] = original.paired(
            references[0], scored
        )
        result["diagnostic_candidate_vs_native2"] = original.paired(
            references[1], scored
        )
    result["candidate_vs_original_raw"] = original.paired(references[2], scored)
    for item in result["scores"]:
        item["accuracy_exact_binomial_95_ci"] = original.cp(
            item["correct"], item["total"]
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess(args.candidate.resolve())
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "scope": result["scope"],
                "correct_counts": [s["correct"] for s in result["scores"]],
                "full_frozen_plan_passed": result["full_frozen_plan_passed"],
            }
        )
    )


if __name__ == "__main__":
    main()
