"""Score frozen gold MCQ answers; never infer semantic quality from text equality."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def load_frozen(directory):
    plan = json.loads((directory / "plan.json").read_text())
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if sha(canonical(body)) != plan["plan_sha256"]:
        raise ValueError("Frozen plan digest mismatch")
    for name, digest in plan["source_files"].items():
        if sha((directory / name).read_bytes()) != digest:
            raise ValueError(f"Frozen input digest mismatch: {name}")
    cases = [json.loads(line) for line in (directory / plan["case_file"]).read_text().splitlines()]
    if len(cases) != plan["case_count"] or len({row["case_id"] for row in cases}) != len(cases):
        raise ValueError("Frozen case cardinality mismatch")
    spec = importlib.util.spec_from_file_location("frozen_canonical_adapter", directory / "canonical-adapter.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return plan, cases, module.extract_answer


def score(run, plan, cases, extract):
    if run["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("Run used a different plan")
    outputs = {}
    for row in run["outputs"]:
        if row["case_id"] in outputs:
            raise ValueError("Duplicate output case ID")
        outputs[row["case_id"]] = row
    if set(outputs) - {row["case_id"] for row in cases}:
        raise ValueError("Unknown output case ID")
    rows = []
    for case in cases:
        output = outputs.get(case["case_id"], {})
        complete = bool(output.get("finished") and not output.get("error")
                        and output.get("finish_reason") == "stop")
        answer = extract(output.get("text", "")) if complete else None
        valid = answer is not None and 0 <= ord(answer) - 65 < case["option_count"]
        rows.append({"case_id": case["case_id"], "gold": case["answer"],
                     "answer": answer, "complete": complete, "valid_answer": valid,
                     "correct": bool(complete and valid and answer == case["answer"]),
                     "finish_reason": output.get("finish_reason"),
                     "output_tokens": len(output.get("token_ids", []))})
    complete = bool(all(row["complete"] for row in rows) and run.get("collection_complete")
                    and run.get("shutdown_returned") and not run.get("errors"))
    correct = sum(row["correct"] for row in rows)
    return {"rows": rows, "correct": correct, "total": len(rows),
            "accuracy": correct / len(rows), "all_complete": complete,
            "invalid_answers": sum(not row["valid_answer"] for row in rows),
            "incomplete": [row["case_id"] for row in rows if not row["complete"]]}


def cp(success, n, alpha=0.05):
    """Two-sided Clopper-Pearson, with explicit empty-denominator behavior."""
    from scipy.stats import beta

    if n == 0:
        return None
    return [0.0 if success == 0 else float(beta.ppf(alpha / 2, success, n - success + 1)),
            1.0 if success == n else float(beta.ppf(1 - alpha / 2, success + 1, n - success))]


def paired(reference, candidate):
    from scipy.stats import binomtest

    transitions = {name: [] for name in ("correct_correct", "correct_wrong", "wrong_correct", "wrong_wrong")}
    for first, second in zip(reference["rows"], candidate["rows"], strict=True):
        assert first["case_id"] == second["case_id"]
        key = ("correct" if first["correct"] else "wrong") + "_" + ("correct" if second["correct"] else "wrong")
        transitions[key].append(first["case_id"])
    loss, gain = len(transitions["correct_wrong"]), len(transitions["wrong_correct"])
    n = reference["total"]
    # 97.5% binomial intervals for each discordant probability give a
    # conservative simultaneous >=95% interval for gain-minus-loss via union bound.
    gain_ci, loss_ci = cp(gain, n, .025), cp(loss, n, .025)
    return {"transitions": transitions, "counts": {k: len(v) for k, v in transitions.items()},
            "accuracy_difference": (gain - loss) / n,
            "paired_difference_conservative_95_ci": [gain_ci[0] - loss_ci[1], gain_ci[1] - loss_ci[0]],
            "mcnemar_exact_two_sided_p_diagnostic_only": float(binomtest(gain, gain + loss, .5).pvalue) if gain + loss else 1.0}


def evaluate(first, second, candidate):
    from scipy.stats import beta

    stable_correct = {a["case_id"] for a, b in zip(first["rows"], second["rows"], strict=True)
                      if a["correct"] and b["correct"]}
    new_wrong = [row["case_id"] for row in candidate["rows"]
                 if row["case_id"] in stable_correct and not row["correct"]]
    n, failures = len(stable_correct), len(new_wrong)
    upper = (None if n == 0 else 1.0 if failures == n else
             float(beta.ppf(.95, failures + 1, n - failures)))
    checks = {"both_native_controls_complete": first["all_complete"] and second["all_complete"],
              "candidate_complete": candidate["all_complete"],
              "no_new_wrong_on_both_native_correct": not new_wrong,
              "nonempty_stable_correct_control_set": bool(stable_correct),
              "correct_count_at_least_lower_native": candidate["correct"] >= min(first["correct"], second["correct"])}
    return {"bounded_answer_quality_gate_passed": all(checks.values()), "checks": checks,
            "stable_native_correct_cases": sorted(stable_correct), "new_wrong_case_ids": new_wrong,
            "stable_correct_new_error_probability_one_sided_95_upper": upper,
            "candidate_vs_native1": paired(first, candidate), "candidate_vs_native2": paired(second, candidate),
            "native2_vs_native1": paired(first, second),
            "limitation": "Fixed selected subset; confidence assumes exchangeable Bernoulli cases and is not a population guarantee. Nonsignificance never establishes equivalence. Mixed-P/D coverage is assessed separately from actual scheduler receipts."}


def compare_provenance(runs):
    ignored = {"worker_cls", "enable_chunked_prefill", "scheduler_cls", "batch_admission_policy", "batch_admission_policy_config"}
    common = [{key: value for key, value in run["engine_args"].items() if key not in ignored} for run in runs]
    if any(item != common[0] for item in common[1:]):
        raise ValueError("Common native/candidate runtime settings differ")
    for field in ("producer_sha256", "runtime_environment"):
        if any(not run.get(field) for run in runs) or any(run[field] != runs[0][field] for run in runs[1:]):
            raise ValueError(f"Missing or inconsistent {field}")
    for module in ("vllm", "vllm_ascend", "vllm_muxpd"):
        identities = [run.get("sources", {}).get(module, {}) for run in runs]
        if any(not item.get("commit") for item in identities):
            raise ValueError(f"Missing source identity: {module}")
        if len({(item["commit"], item["sha256"], item["status"], item.get("diff_sha256"),
                 canonical(item.get("changed_files"))) for item in identities}) != 1:
            raise ValueError(f"Runtime source identities differ: {module}")
        for item in identities:
            if item["status"] and (not item.get("changed_files") or "diff_bytes_utf8" not in item
                                   or sha(item["diff_bytes_utf8"].encode()) != item.get("diff_sha256")):
                raise ValueError(f"Dirty source lacks exact diff/file identities: {module}")


def verify_mixed_coverage(run_path, run, plan):
    path = run_path.parent / "scheduler-receipts.jsonl"
    if not path.exists() or sha(path.read_bytes()) != run.get("scheduler_receipts_sha256"):
        return {"passed": False, "reason": "Missing or mismatched scheduler receipt"}
    scheduled, counts = {}, {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        key = (row["batch_id"], row["global_step"])
        if row["event"] == "schedule":
            if key in scheduled:
                raise ValueError("Duplicate schedule receipt")
            scheduled[key] = row
        elif row["event"] == "retirement":
            before = scheduled.pop(key)
            after = row.get("prefill_layers_completed")
            mixed = bool(before["pending_prefill_ids"] and after is not None
                         and after > before["prefill_layers_before"] and row["decode_retired_ids"])
            if mixed:
                counts[row["batch_id"]] = counts.get(row["batch_id"], 0) + 1
        else:
            raise ValueError("Unknown scheduler receipt type")
    required = {row["batch_id"] for row in plan["batches"] if len(row["case_ids"]) > 1}
    missing = sorted(required - counts.keys())
    claimed = {int(key): value for key, value in run.get("mixed_coverage", {}).get("steps_per_batch", {}).items()}
    return {"passed": not missing and not scheduled and counts == claimed,
            "steps_per_batch": counts, "uncovered_batches": missing,
            "unretired_schedule_count": len(scheduled), "matches_run_summary": counts == claimed,
            "receipt_sha256": sha(path.read_bytes())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--native1", type=Path, required=True)
    parser.add_argument("--native2", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--raw-reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan, cases, extract = load_frozen(args.frozen)
    paths = [args.native1, args.native2, args.candidate]
    runs = [json.loads(path.read_text()) for path in paths]
    if [run["role"] for run in runs[:2]] != ["native", "native"] or runs[2]["role"] not in ("raw", "group"):
        raise ValueError("Expected two native controls and one named candidate")
    if runs[0]["process_id"] == runs[1]["process_id"] or runs[0]["started_at"] == runs[1]["started_at"]:
        raise ValueError("Native controls must come from two fresh processes")
    if runs[0]["engine_args"] != runs[1]["engine_args"]:
        raise ValueError("Native control launch arguments differ")
    compare_provenance(runs)
    scored = [score(run, plan, cases, extract) for run in runs]
    result = evaluate(*scored)
    coverage = verify_mixed_coverage(args.candidate, runs[2], plan)
    result["mixed_coverage_gate"] = coverage
    result["full_frozen_plan_passed"] = result["bounded_answer_quality_gate_passed"] and coverage["passed"]
    if runs[2]["role"] == "group":
        if args.raw_reference is None:
            parser.error("group comparison also requires --raw-reference")
        raw_run = json.loads(args.raw_reference.read_text())
        if raw_run["role"] != "raw":
            raise ValueError("Raw reference has the wrong role")
        compare_provenance([*runs, raw_run])
        result["group_vs_raw"] = paired(score(raw_run, plan, cases, extract), scored[2])
        result["raw_reference_source"] = {"path": str(args.raw_reference), "sha256": sha(args.raw_reference.read_bytes())}
    for value in scored:
        value["accuracy_exact_binomial_95_ci"] = cp(value["correct"], value["total"])
    result.update({"plan_sha256": plan["plan_sha256"], "scores": scored,
                   "run_sources": [{"path": str(path), "sha256": sha(path.read_bytes())} for path in paths],
                   "scorer_sha256": sha(Path(__file__).read_bytes()),
                   "candidate_mixed_coverage": runs[2].get("mixed_coverage", {})})
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"full_frozen_plan_passed": result["full_frozen_plan_passed"],
                      "bounded_answer_quality_gate_passed": result["bounded_answer_quality_gate_passed"],
                      "correct_counts": [value["correct"] for value in scored],
                      "new_wrong_case_ids": result["new_wrong_case_ids"]}, indent=2))


if __name__ == "__main__":
    main()
