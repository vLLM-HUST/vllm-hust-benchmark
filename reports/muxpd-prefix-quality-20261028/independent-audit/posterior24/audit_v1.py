"""Independent CPU audit of a terminal 24-case run; never reads the live full run.

Does not import/call quality_run_v1, quality_control_v1, score_layout_quality_v1,
or score_quality. Only the exact frozen canonical answer extractor is reused.
No model, torch, NPU inventory, process control, or external mutation is used.
"""

import hashlib
import importlib.util
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

from tokenizers import Tokenizer

BASE = Path("/root/workspace/muxpd-quality-layout-20261028")
RUN = BASE / "quality-runs/prefix-quality-losses-01"
OUT = Path(__file__).resolve().parent
OLD = Path("/root/workspace/muxpd-followup-20261025/quality")
PLAN_SHA = "dd3e1f305475ec556dc824947f26d5fb5743242e82b5099155c95b8954bb2e65"
LAYOUT_SHA = "6431c4c0abdd869ef17e7122f681c457ad2b4d5741f86c6a0653e9fca44e02fc"
FREEZE_SHA = "07a796c568e8cd69ae7157b3386269b1a624a94b6b8393bba232a5cec68a9345"
ASSESSMENT_SHA = "40804616e0fd02668514933272913a035b3f3a6886329788aa2536b257e12c60"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def need(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    sources = []

    def bind(path):
        sources.append({"path": str(path), "sha256": sha(path)})
        return read(path)

    controller = bind(RUN / "controller.json")
    run = bind(RUN / "results/run.json")
    need(controller["finished_at"] and run["finished_at"], "Only terminal run is auditable")
    need(controller["passed"] and controller["exit_code"] == 0 and not controller["errors"], "Controller failed")
    need(controller.get("owned_members_after_cleanup") == [], "Owned process group not explicitly empty")
    need(not controller.get("cleanup_required") and not controller.get("timeout") and not controller.get("forced_termination"), "Run required forced cleanup")
    need(controller["terminal_run_sha256"] == sha(RUN / "results/run.json"), "Terminal child bytes changed")
    need(all(run[k] for k in ["passed", "collection_complete", "layout_restored", "shutdown_returned"]) and not run["errors"], "Child lifecycle failed")
    need(not any(k.startswith("failure_") or k == "layout_failure_uninstall" for k in run), "Failure cleanup present")
    need(run["terminal_maintenance"]["aborted_internal_ids"] == [], "Normal completion cancelled requests")
    need(datetime.fromisoformat(controller["finished_at"]) >= datetime.fromisoformat(run["finished_at"]), "Controller precedes child terminal state")
    log = RUN / "process.log"
    text = log.read_text()
    need("Traceback (most recent call last):" not in text, "Terminal traceback")
    need("RuntimeError: MuxPD graph method changed externally:" not in text, "Terminal graph ownership error")
    need(sha(log) == controller["terminal_process_log_audit"]["sha256"], "Terminal process log changed")
    sources.append({"path": str(log), "sha256": sha(log)})
    for when in ("before", "after"):
        path = RUN / f"npu-{when}.txt"
        value = path.read_text()
        physical = set(re.findall(r"^\|\s*(\d+)\s+910B2\s*\|", value, re.MULTILINE))
        idle = set(re.findall(r"No running processes found in NPU (\d+)\b", value))
        need(physical == idle == {"0", "2", "3", "4"}, "Saved NPU idle/inventory mismatch")
        need(controller[f"idle_physical_{when}"] == ["0", "2", "3", "4"], "Controller idle claim differs")
        sources.append({"path": str(path), "sha256": sha(path)})
    freeze = bind(RUN / "producer/freeze-manifest-v1.json")
    need(sha(RUN / "producer/freeze-manifest-v1.json") == FREEZE_SHA == controller["freeze_manifest_sha256"], "Unexpected freeze")
    need(freeze["files"] == controller["files"], "Freeze and controller snapshots differ")
    for name, expected in freeze["files"].items():
        need(sha(RUN / "producer" / name) == expected, f"Frozen producer differs: {name}")
    need(run["producer_sha256"] == freeze["files"]["quality_run_v1.py"], "Producer identity mismatch")
    need(run["original_scorer_sha256"] == sha(OLD / "score_quality.py") == freeze["files"]["score_quality.py"], "Original scorer bytes changed")
    manifest = bind(RUN / "producer/source-manifest-v1.json")
    for item in manifest["copied_files"]:
        copied = RUN / "producer" / item["destination"]
        need(sha(copied) == item["sha256"] == sha(Path(item["source"])), "Source copy mismatch")

    plan = bind(RUN / "producer/frozen-v2/plan.json")
    body = {k: v for k, v in plan.items() if k != "plan_sha256"}
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    need(hashlib.sha256(encoded).hexdigest() == PLAN_SHA == plan["plan_sha256"] == run["plan_sha256"], "Original plan digest mismatch")
    need(sha(RUN / "producer/frozen-v2/plan.json") == sha(OLD / "frozen-v2/plan.json"), "Original plan not byte-identical")
    for name, expected in plan["source_files"].items():
        need(sha(RUN / "producer/frozen-v2" / name) == expected == sha(OLD / "frozen-v2" / name), "Frozen source file differs")
    cases = [json.loads(line) for line in (RUN / "producer/frozen-v2/cases.jsonl").read_text().splitlines()]
    need(len(cases) == 200 and len(plan["batches"]) == 25, "Original population changed")
    need(plan["sampling"] == {"temperature": 0, "seed": 42, "max_tokens": 8192, "ignore_eos": False, "detokenize": False}, "Original sampling changed")
    need(plan["maximum_wall_seconds"] == 10800 and plan["maximum_engine_steps"] == 215300, "Original work bounds changed")
    batches = [b for b in plan["batches"] if b["batch_id"] in [3, 6, 14]]
    selected = [c for b in batches for c in b["case_ids"]]
    need(len(selected) == 24 and run["selected_case_ids"] == selected and run["selected_batch_ids"] == [3, 6, 14], "Incomplete/repacked selected batches")
    need(run["scope"] == "loss-batches" and controller["scope"] == "loss-batches", "Wrong posterior diagnostic scope")
    by_id = {c["case_id"]: c for c in cases}
    adapter = RUN / "producer/frozen-v2/canonical-adapter.py"
    spec = importlib.util.spec_from_file_location("independent_frozen_answer_adapter", adapter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = plan["tokenizer_files"]["tokenizer.json"]
    for tokenfile in plan["tokenizer_files"].values():
        need(sha(Path(tokenfile["path"])) == tokenfile["sha256"], "Tokenizer source differs")
    decoder = Tokenizer.from_file(tokenizer["path"])

    historical = []
    for label in ("native-quality-v2-hccl-true-01", "native-quality-v2-hccl-true-02", "raw-quality-v2-hccl-true-01"):
        identity = manifest["historical_runs"][label]
        path = Path(identity["path"])
        need(sha(path) == identity["sha256"], "Historical run hash mismatch")
        historical.append(bind(path))
    need(historical[0]["process_id"] != historical[1]["process_id"] and historical[0]["started_at"] != historical[1]["started_at"], "Native processes not distinct")
    for native in historical[:2]:
        need(native["role"] == "native" and native["collection_complete"] and native["shutdown_returned"] and not native["errors"], "Native collection failed")
    need(run["engine_args"] == historical[2]["engine_args"] and run["runtime_environment"] == historical[2]["runtime_environment"], "Raw runtime changed")
    need(historical[0]["sources"] == historical[1]["sources"] == historical[2]["sources"], "Historical provenance differs")
    for name in ("vllm", "vllm_ascend"):
        need(run["sources"][name] == historical[2]["sources"][name], "Core/Ascend changed")
    for key in ("source_after_startup", "sources_after_shutdown"):
        need(run[key]["sources"] == run["sources"], "Source changed during run")
    need(controller["source_preflight"]["sources"] == run["sources"], "Preflight source differs")
    equivalence = run["source_equivalence"]
    source = Path(equivalence["source"])
    new = "6c606f789450a45ef35afee398b0a3e516caa1a1"
    old = "f7ed21e183b27d2141176f3e7a5c8a5ff4ede07d"
    need(run["sources"]["vllm_muxpd"]["commit"] == new and not run["sources"]["vllm_muxpd"]["status"], "Plugin identity changed")
    existing = subprocess.check_output(["git", "-C", str(source), "ls-tree", "-r", "--name-only", old, "--", "src"], text=True).splitlines()
    need(len(existing) == 20 and existing == equivalence["existing_runtime_files"], "Original runtime-file list differs")
    changed = subprocess.check_output(["git", "-C", str(source), "diff", "--name-only", old, new, "--", *existing], text=True)
    need(not changed and not equivalence["existing_runtime_diff"], "Original f7 runtime bytes changed")

    def independently_score(value):
        outputs = {r["case_id"]: r for r in value["outputs"]}
        need(len(outputs) == len(value["outputs"]), "Duplicate output ID")
        rows = {}
        for case_id in selected:
            output = outputs[case_id]
            tokens = output["token_ids"]
            need(type(tokens) is list and all(type(t) is int and t >= 0 for t in tokens), "Malformed output tokens")
            need(decoder.decode(tokens, skip_special_tokens=True) == output["text"], "Saved text differs from complete token sequence")
            complete = bool(output.get("finished") and not output.get("error") and output.get("finish_reason") == "stop")
            answer = module.extract_answer(output["text"]) if complete else None
            valid = answer is not None and 0 <= ord(answer) - 65 < by_id[case_id]["option_count"]
            rows[case_id] = {"answer": answer, "gold": by_id[case_id]["answer"], "complete": complete,
                             "valid_answer": valid, "correct": bool(complete and valid and answer == by_id[case_id]["answer"]),
                             "finish_reason": output["finish_reason"], "output_tokens": len(tokens),
                             "tokens_sha256": digest(tokens), "text_sha256": hashlib.sha256(output["text"].encode()).hexdigest()}
        return rows, outputs

    scored = [independently_score(x) for x in [*historical, run]]
    need({o["case_id"] for o in run["outputs"]} == set(selected) and len(run["outputs"]) == 24, "Candidate not exactly24 answers")
    need(all(row["complete"] for row in scored[3][0].values()), "Candidate contains truncated/incomplete answer")
    detailed = []
    byte_equal = [0, 0, 0]
    for case_id in selected:
        comparisons = []
        tokens = scored[3][1][case_id]["token_ids"]
        for index in range(3):
            ref = scored[index][1][case_id]["token_ids"]
            same = ref == tokens
            byte_equal[index] += int(same)
            common = next((i for i, (a, b) in enumerate(zip(ref, tokens)) if a != b), min(len(ref), len(tokens)))
            comparisons.append({"tokens_equal": same, "common_prefix_tokens": common,
                                "first_difference_token_number_one_based": None if same else common + 1,
                                "text_equal": scored[index][1][case_id]["text"] == scored[3][1][case_id]["text"]})
        detailed.append({"case_id": case_id, "native1": scored[0][0][case_id], "native2": scored[1][0][case_id],
                         "old_raw": scored[2][0][case_id], "prefix": scored[3][0][case_id], "output_comparisons": comparisons})
    stable = {c for c in selected if scored[0][0][c]["correct"] and scored[1][0][c]["correct"]}
    new_wrong = [c for c in selected if c in stable and not scored[3][0][c]["correct"]]
    need(not new_wrong, "New wrong on both-native-correct selected case")
    counts = [sum(r["correct"] for r in rows.values()) for rows, _ in scored]
    need(counts == [22, 22, 19, 22], "Independent answer counts differ")

    receipts = RUN / "results/scheduler-receipts.jsonl"
    need(sha(receipts) == run["scheduler_receipts_sha256"], "Scheduler receipt bytes changed")
    sources.append({"path": str(receipts), "sha256": sha(receipts)})
    pending, mixed, p_schedules = {}, {}, []
    for line in receipts.read_text().splitlines():
        row = json.loads(line)
        key = row["batch_id"], row["global_step"]
        need(row["batch_id"] in [3, 6, 14], "Receipt outside selected batches")
        if row["event"] == "schedule":
            need(key not in pending, "Duplicate schedule")
            pending[key] = row
            need(not row["preempted_req_ids"], "Unexpected preemption")
            if row["new_prefill_ids"]:
                p_schedules.append(row)
        else:
            need(row["event"] == "retirement" and key in pending, "Unmatched retirement")
            before = pending.pop(key)
            after = row["prefill_layers_completed"]
            real_mixed = bool(before["pending_prefill_ids"] and after is not None and after > before["prefill_layers_before"] and row["decode_retired_ids"])
            need(real_mixed == row["mixed_pd_progress_and_retirement"], "Incorrect claimed mixed step")
            if real_mixed:
                mixed[row["batch_id"]] = mixed.get(row["batch_id"], 0) + 1
    need(not pending and mixed == {3: 10, 6: 10, 14: 10}, "Mixed progress/retirement coverage differs")
    need(mixed == {int(k): v for k, v in run["mixed_coverage"]["steps_per_batch"].items()}, "Mixed summary differs")
    need(len(run["admissions"]) == len(p_schedules) == 6, "Expected exactly seed+peer in three batches")
    binding = {}
    for i, admission in enumerate(run["admissions"]):
        batch = batches[i // 2]
        members = batch["case_ids"][:1] if i % 2 == 0 else batch["case_ids"][1:]
        need([b["case_id"] for b in admission["bindings"]] == members and admission["batch_id"] == batch["batch_id"], "Original batch admission order changed")
        need(admission["stage"] == ("seed" if i % 2 == 0 else "peers"), "Seed/peer stage changed")
        if i % 2:
            need(admission["seed_output_tokens"] == 1 and admission["seed_still_live"], "Peer release missed exact live seed token1")
        binding.update({b["internal"]: b["case_id"] for b in admission["bindings"]})
    for key in ("layout_installation", "layout_final", "layout_uninstall", "worker_after"):
        need(len(run[key]) == 4 and {r["rank"] for r in run[key]} == set(range(4)), "Missing/duplicate TP rank")
    calls = 0
    for row in run["layout_installation"]:
        need(row["module_sha256"] == LAYOUT_SHA and row["installed_hook_count"] == 3 and row["all_hooks_intact"] and not row["cohorts"] and not row["errors"], "Bad layout installation")
    for row in run["layout_final"]:
        need(row["module_sha256"] == LAYOUT_SHA and row["mode"] == "prefix" and row["installed_hook_count"] == 3 and row["all_hooks_intact"], "Layout identity/ownership mismatch")
        need(len(row["cohorts"]) == 6 and row["treated_cohorts"] == 3 and row["active_cohort"] is None and not row["errors"] and not row["in_collective"], "Layout terminal state differs")
        for i, cohort in enumerate(row["cohorts"]):
            schedule = p_schedules[i]
            p_counts = {rid: schedule["scheduled_tokens"][rid] for rid in schedule["new_prefill_ids"]}
            d_counts = {rid: count for rid, count in schedule["scheduled_tokens"].items() if rid not in p_counts}
            need(cohort["admission_p_counts"] == p_counts and cohort["admission_d_counts"] == d_counts, "Layout does not match actual scheduler admission")
            need(cohort["status"] == "closed" and cohort["clear_count"] == 1 and cohort["error"] is None, "Cohort not naturally cleared once")
            need(cohort["d_rows"] == i % 2 and cohort["p_rows"] == sum(p_counts.values()), "Expected actual seedD0/peerD1 geometry differs")
            need(cohort["batch"]["packed_tokens_and_positions_verified"], "Actual P layout not verified")
            for actual in cohort["batch"]["rows"]:
                case = by_id[binding[actual["request_id"]]]
                need(actual["prompt_sha256"] == digest(case["prompt_token_ids"]), "Worker actual prompt fingerprint differs from frozen original case")
                need(actual["prompt_tokens"] == len(case["prompt_token_ids"]) == p_counts[actual["request_id"]] and actual["computed_tokens"] == actual["output_tokens"] == 0, "P is not complete fresh prompt")
            if not d_counts:
                need(not cohort["calls"] and cohort["zero_d_bypass_calls"] > 0, "D0 did not bypass")
            else:
                need(cohort["calls"] and not cohort["zero_d_bypass_calls"], "D1 not treated")
            for ordinal, call in enumerate(cohort["calls"], 1):
                calls += 1
                need(call["ordinal"] == ordinal and call["status"] == "complete" and call["error"] is None, "Incomplete/misordered call")
                need(call["existing_collective_calls"] == 1 and call["return_original_input"] is True, "Single-call alias contract failed")
                need(call["input_shape"] == [cohort["p_rows"], 2048] and call["collective_payload_shape"] == [cohort["p_rows"] + 1, 2048], "Actual collective payload differs")
                need(call["prefix_rows"] == call["dummy_rows"] == call["copyback_offset"] == 1 and call["suffix_rows"] == 0 and call["dtype"] == "torch.bfloat16", "Prefix/dtype/copyback changed")
                need(call["router_call_delta"] == {"caller": 0, "decode": 0, "prefill": 1}, "Route count changed")
                route = call["route_before"]
                need(route["active_lane"] == "prefill" and route["ranks"] == [0, 1, 2, 3] and route["device_group_python_id"] != route["selected_device_group_python_id"], "P sibling route not isolated")
    for row in run["layout_uninstall"]:
        need(row["ok"] and row["idle"] and row["all_originals_restored"] and row["restored_count"] == row["installed_hook_count"] == 3 and not row["allow_incomplete"] and not row["error"] and not row["collection_errors"], "Normal three-slot uninstall failed")
    for row in run["worker_after"]:
        need(not any(row["requests"].values()) and not any(row["input_num_reqs"].values()) and not any(row["input_req_id_to_index"].values()), "Worker retains requests")
        need(not any(rid is not None for lane in row["input_rows"].values() for rid in lane), "Worker input slots not empty")
        need(not row["prefill_task_present"] and row["prefill_cohort_id"] is None and not row["awaiting_sample"] and not row["pending_lanes"], "Worker retains P/iteration ownership")
    final = run["final_state"]
    need(all(not final[k] for k in final if k not in ["free_blocks", "pool_blocks"]), "Scheduler not empty")
    need(final["free_blocks"] == run["idle_baseline"]["free_blocks"], "KV blocks not restored")

    # Cross-check the wrapper only after independently deriving every answer and
    # lifecycle result above. It is an output to audit, never the oracle.
    assessment = bind(RUN / "assessment.json")
    need(sha(RUN / "assessment.json") == ASSESSMENT_SHA, "Assessment identity changed")
    need(assessment["full_frozen_plan_passed"] is None and assessment["full_quality_gate_attempted"] is False, "Posterior subset was promoted to full gate")
    for index, independent in zip((0, 1, 3), assessment["scores"], strict=True):
        need(independent["correct"] == counts[index], "Wrapper count differs")
        for reported in independent["rows"]:
            ours = scored[index][0][reported["case_id"]]
            need(all(reported[k] == ours[k] for k in ["answer", "gold", "complete", "correct", "valid_answer", "finish_reason", "output_tokens"]), "Wrapper per-case score differs")
    targets = [next(row for row in detailed if row["case_id"] == c) for c in ["mmlu-pro-03766", "mmlu-pro-10672", "mmlu-pro-01741"]]
    summary = {"schema": "muxpd-prefix-losses-independent-audit-v1", "passed": True,
               "scope": "Posterior original batches3/6/14,24 complete answers. No full200/current-main/performance qualification.",
               "independence": "Independent JSON/provenance/lifecycle/payload checks and scoring loop. Does not import any quality driver/controller/scorer/wrapper. Uses only the SHA-pinned original canonical extract_answer and original tokenizer.",
               "correct_counts_native1_native2_raw_prefix": counts,
               "candidate_natural_stop_count": 24, "candidate_truncation_count": 0,
               "new_wrong_on_both_native_correct": new_wrong,
               "exact_full_token_sequence_equal_to_native1_native2_raw": byte_equal,
               "candidate_full_answers_not_globally_byte_identical_to_native": byte_equal[0] < 24 or byte_equal[1] < 24,
               "regression_targets": [{"case_id": r["case_id"], **r["prefix"]} for r in targets],
               "mixed_steps_per_batch": mixed, "total_recorded_treated_calls_all_ranks": calls,
               "completed_cohorts_per_rank": 6, "treated_cohorts_per_rank": 3,
               "normal_three_slot_uninstall": True, "owned_processes_empty": True,
               "saved_physical_idle_before_after": [0, 2, 3, 4],
               "runtime20_f7_to_6c_unchanged": True, "default_main_or_package_qualification_claimed": False,
               "full_frozen_plan_passed": None,
               "historical_control_limitation": "Original full200 native controls each have9 length truncations; no completion/budget rule is changed here.",
               "assessment_sha256": ASSESSMENT_SHA, "audit_producer_sha256": sha(Path(__file__))}
    for name, value in [("summary.json", summary), ("rows.json", detailed), ("source-identities.json", sources)]:
        with (OUT / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
