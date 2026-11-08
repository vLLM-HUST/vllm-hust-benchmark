"""Retry a failed original NatureBench validity judge, never rerun the Agent.

Run in the qualified trusted controller environment, with saved task output and
original data mounted read-only. Default verifies and prints a plan; --execute
uses the unchanged original collector, prompt,4096-token policy and parser.
"""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from urllib.request import Request, urlopen

from prepare_naturebench_native_runtime import verify_runtime
from run_naturebench_original import endpoint
from vllm_hust_workload_data.naturebench import original_cpu_packages


def plan(args):
    source = verify_runtime(args.author_root, args.runtime)
    packages = list(original_cpu_packages(args.data_root, task_ids=[args.task_id]))
    saved = args.saved_task.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("fresh judge recovery output required")
    for root in (
        saved,
        args.data_root.resolve(),
        args.runtime.resolve(),
        args.author_root.resolve(),
    ):
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("recovery output must be separate from all inputs")
    verdict = json.loads((saved / "judge_verdict.json").read_text("utf-8"))
    if verdict.get("is_valid") is not None:
        raise ValueError("existing boolean judge result must be reused, not retried")
    if not args.model.strip():
        raise ValueError("actual served judge model required")
    return {
        "source_id": "NATUREBENCH",
        "task_id": args.task_id,
        "source": packages[0]["source"],
        "runtime_source": str(source),
        "judge_endpoint": endpoint(args.endpoint),
        "model": args.model,
        "status": "VERIFIED_FAILED_JUDGE_RECOVERY_PLAN_NOT_EXECUTED",
        "new_agent_calls": 0,
        "source_checkbox_eligible": False,
        "original_verdict_sha256": hashlib.sha256(
            (saved / "judge_verdict.json").read_bytes()
        ).hexdigest(),
    }


def execute(args, prepared):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    receipt = dict(prepared)
    sys.path.insert(0, prepared["runtime_source"])
    original = importlib.import_module("judge_core.runner")
    policy = importlib.import_module("judge_core.policy")
    for module in (original, policy):
        if (
            not Path(module.__file__)
            .resolve()
            .is_relative_to(Path(prepared["runtime_source"]).resolve())
        ):
            raise ValueError("loaded judge module is outside verified runtime")
    problem = args.data_root / "tasks" / args.task_id / "problem"
    inputs = original.collect_judge_inputs_with_context(
        args.saved_task, problem, agent_name="claude", task_name=args.task_id
    )
    canonical = json.dumps(inputs, sort_keys=True, ensure_ascii=False).encode()
    prompt = original.build_user_prompt_with_context(inputs)
    receipt.update(
        original_input_sha256=hashlib.sha256(canonical).hexdigest(),
        original_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        original_output_budget=policy.MAX_JUDGE_OUTPUT_TOKENS,
    )
    key = os.environ.get("JUDGE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["x-api-key"] = key
    body = {
        "model": args.model,
        "system": policy.JUDGE_SYSTEM,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": policy.MAX_JUDGE_OUTPUT_TOKENS,
    }
    try:
        with urlopen(
            Request(
                prepared["judge_endpoint"] + "/v1/messages/count_tokens",
                data=json.dumps(body).encode(),
                headers=headers,
            ),
            timeout=120,
        ) as response:
            count = json.load(response)
        with urlopen(
            Request(prepared["judge_endpoint"] + "/v1/models", headers=headers),
            timeout=30,
        ) as response:
            models = json.load(response)
        model = next(row for row in models["data"] if row["id"] == args.model)
        receipt.update(
            input_tokens=count["input_tokens"],
            served_max_context=model["max_model_len"],
        )
        if (
            count["input_tokens"] + policy.MAX_JUDGE_OUTPUT_TOKENS
            > model["max_model_len"]
        ):
            raise ValueError(
                "full original evidence exceeds served context; no trimming allowed"
            )
        verdict = original.judge_task(
            args.task_id,
            args.saved_task,
            agent_name="claude",
            model=args.model,
            base_url=prepared["judge_endpoint"],
            api_key=key or "local-runtime-not-a-secret",
            task_problem_dir=problem,
        )
        (output / "original-verdict.private.json").write_text(
            json.dumps(verdict, indent=2), encoding="utf-8"
        )
        after = original.collect_judge_inputs_with_context(
            args.saved_task, problem, agent_name="claude", task_name=args.task_id
        )
        if json.dumps(after, sort_keys=True, ensure_ascii=False).encode() != canonical:
            raise ValueError("saved judge evidence changed during recovery")
        if (
            hashlib.sha256(
                (args.saved_task / "judge_verdict.json").read_bytes()
            ).hexdigest()
            != prepared["original_verdict_sha256"]
        ):
            raise ValueError("original verdict changed during recovery")
        receipt.update(
            status="ORIGINAL_JUDGE_RECOVERY_RETURNED", is_valid=verdict.get("is_valid")
        )
        if type(verdict.get("is_valid")) is not bool or not verdict.get("reason"):
            raise ValueError("original judge did not return a valid boolean verdict")
        verify_runtime(args.author_root, args.runtime)
        list(original_cpu_packages(args.data_root, task_ids=[args.task_id]))
        return 0
    finally:
        (output / "recovery-receipt.json").write_text(
            json.dumps(receipt, indent=2), encoding="utf-8"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("author-root", "runtime", "data-root", "saved-task", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    prepared = plan(args)
    if args.execute:
        raise SystemExit(execute(args, prepared))
    print(json.dumps(prepared, indent=2))


if __name__ == "__main__":
    main()
