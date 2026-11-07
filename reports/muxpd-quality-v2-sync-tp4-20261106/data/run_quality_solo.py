"""Single-in-flight variant of the frozen run_quality.py (sha256 389ab14e...).

Imports the frozen module unchanged and reuses validate/enqueue_bound/install_receipts/
runtime_environment/source_identities/write. Only the admission loop differs: each case is
enqueued alone and drained before the next one is enqueued, so at most one request is in flight
(asserted on every engine step). Bounds are explicit arguments because the frozen batched
maximum_engine_steps (215300) cannot cover one-token-per-step solo decoding.
"""
import argparse
import hashlib
import json
import os
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import run_quality as frozen
from score_quality import load_frozen

FROZEN_DRIVER_SHA256 = "389ab14e591130d2bc44a31b93f31ecde6a44112fb8d36662d15505abe7359cf"


def run(args, plan, cases, engine_args):
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if any(os.environ.get(k) for k in ("MUXPD_NUMERIC_OUTPUT", "MUXPD_PRESSURE_TRACE_DIR")):
        raise ValueError("Remove unrelated numerical/pressure observers from this diagnostic")
    report = {"schema": "muxpd-mmlu-pro-quality-run-solo-v1", "role": args.role,
              "admission": "single in-flight: enqueue one case, drain it, then the next (cases.jsonl order)",
              "plan_sha256": plan["plan_sha256"], "process_id": os.getpid(),
              "started_at": datetime.now(UTC).isoformat(), "engine_args": engine_args,
              "outputs": [], "admissions": [], "errors": [],
              "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "frozen_driver_sha256": hashlib.sha256(Path(frozen.__file__).read_bytes()).hexdigest(),
              "bounds": {"maximum_engine_steps": args.max_steps, "maximum_wall_seconds": args.max_wall_seconds},
              "runtime_environment": frozen.runtime_environment(), "max_in_flight_observed": 0,
              "collection_complete": False, "shutdown_returned": False,
              "scope": "Single-request reference outputs; no timing or population-equivalence claim"}
    if report["frozen_driver_sha256"] != FROZEN_DRIVER_SHA256:
        raise ValueError("Imported run_quality.py differs from the frozen driver")
    frozen.write(args.output / "run.json", report)
    llm = None
    coverage, context = {}, {"global_step": 0, "batch_id": -1}
    try:
        from tokenizers import Tokenizer
        from vllm import LLM, SamplingParams
        from vllm.v1.engine.core_client import InprocClient

        tokfile = plan["tokenizer_files"]["tokenizer.json"]
        if hashlib.sha256(Path(tokfile["path"]).read_bytes()).hexdigest() != tokfile["sha256"]:
            raise ValueError("Tokenizer decoder differs from frozen source")
        decoder = Tokenizer.from_file(tokfile["path"])
        llm = LLM(**engine_args)
        if not isinstance(llm.llm_engine.engine_core, InprocClient):
            raise TypeError("Actual coordinator is not InprocClient")
        core = llm.llm_engine.engine_core.engine_core
        report["sources"] = frozen.source_identities()
        cfg = core.vllm_config
        report["effective"] = {"max_model_len": cfg.model_config.max_model_len,
                               "max_num_seqs": cfg.scheduler_config.max_num_seqs,
                               "max_num_batched_tokens": cfg.scheduler_config.max_num_batched_tokens,
                               "enable_chunked_prefill": cfg.scheduler_config.enable_chunked_prefill,
                               "tensor_parallel_size": cfg.parallel_config.tensor_parallel_size}
        if report["effective"] != {k: engine_args[k] for k in report["effective"]}:
            raise RuntimeError("Effective runtime differs from frozen launch arguments")
        report["scheduler_class"] = type(core.scheduler).__module__ + "." + type(core.scheduler).__name__
        report["engine_coordinator"] = type(llm.llm_engine.engine_core).__name__
        params = SamplingParams(**plan["sampling"])
        started = time.monotonic()
        with (args.output / "scheduler-receipts.jsonl").open("x", buffering=1) as receipts:
            frozen.install_receipts(core.scheduler, receipts, context, coverage)
            for index, case in enumerate(cases):
                context["batch_id"] = index
                mappings = frozen.enqueue_bound(llm, [case], params)
                admitted = {mappings[0]["external"]: case}
                report["admissions"].append({**context, "stage": "solo", "bindings": mappings})
                finished = set()
                case_started = time.monotonic()
                while llm.llm_engine.has_unfinished_requests():
                    if context["global_step"] >= args.max_steps or time.monotonic() - started > args.max_wall_seconds:
                        raise RuntimeError("Frozen bounded-work limit exceeded")
                    in_flight = len(core.scheduler.requests)
                    report["max_in_flight_observed"] = max(report["max_in_flight_observed"], in_flight)
                    if in_flight > 1:
                        raise RuntimeError("More than one request in flight")
                    context["global_step"] += 1
                    for output in llm.llm_engine.step():
                        if not output.finished:
                            continue
                        if output.request_id not in admitted or output.request_id in finished or len(output.outputs) != 1:
                            raise RuntimeError("Unknown, duplicate or multi-sample final output")
                        finished.add(output.request_id)
                        item = output.outputs[0]
                        if list(output.prompt_token_ids) != case["prompt_token_ids"]:
                            raise RuntimeError("Returned prompt differs from frozen token IDs")
                        tokens = list(map(int, item.token_ids))
                        report["outputs"].append({"case_id": case["case_id"], "request_id": output.request_id,
                                                  "finished": True, "token_ids": tokens,
                                                  "text": decoder.decode(tokens, skip_special_tokens=True),
                                                  "finish_reason": item.finish_reason, "stop_reason": item.stop_reason,
                                                  "case_wall_seconds": round(time.monotonic() - case_started, 3)})
                if finished != set(admitted):
                    raise RuntimeError("Case ended without its request returning")
                if index % 10 == 9:
                    frozen.write(args.output / "run.json", report)
                    print(f"solo {index + 1}/{len(cases)} steps={context['global_step']} "
                          f"elapsed={time.monotonic() - started:.0f}s", flush=True)
        report["collection_complete"] = len(report["outputs"]) == plan["case_count"]
    except BaseException:  # noqa: BLE001
        report["errors"].append(traceback.format_exc())
    finally:
        if llm is not None:
            try:
                llm.llm_engine.engine_core.shutdown()
                report["shutdown_returned"] = True
            except BaseException:  # noqa: BLE001
                report["errors"].append(traceback.format_exc())
        report["finished_at"] = datetime.now(UTC).isoformat()
        receipt_path = args.output / "scheduler-receipts.jsonl"
        if receipt_path.exists():
            report["scheduler_receipts_sha256"] = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        frozen.write(args.output / "run.json", report)
    complete = report["collection_complete"] and report["shutdown_returned"] and not report["errors"]
    print(json.dumps({"collection_complete": complete, "outputs": len(report["outputs"]), "errors": report["errors"]}))
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("dry-run", "run"))
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--engine-args", type=Path, required=True)
    parser.add_argument("--role", choices=("native", "raw", "group"), required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--max-wall-seconds", type=int, required=True)
    parser.add_argument("--admission", choices=("solo", "batched"), default="solo")
    parser.add_argument("--tp-override", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan, cases, _ = load_frozen(args.frozen)
    if plan["plan_sha256"] != args.plan_sha256:
        raise ValueError("Explicit plan identity mismatch")
    if args.tp_override is not None:  # after the digest check; recorded in run.json via engine_args
        plan["runtime"]["tensor_parallel_size"] = args.tp_override
    engine_args = frozen.validate(args.role, json.loads(args.engine_args.read_text()), plan)
    if args.command == "dry-run":
        print(json.dumps({"validated": True, "role": args.role, "case_count": len(cases),
                          "tp": engine_args["tensor_parallel_size"], "plan_sha256": plan["plan_sha256"],
                          "frozen_driver_sha256": hashlib.sha256(Path(frozen.__file__).read_bytes()).hexdigest()}))
        return 0
    if args.output is None:
        parser.error("run requires a new --output directory")
    if args.admission == "batched":  # unchanged frozen seed-first batched collection
        return frozen.run(args, plan, cases, engine_args)
    return run(args, plan, cases, engine_args)


if __name__ == "__main__":
    raise SystemExit(main())
