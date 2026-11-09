"""Delegate original CPU scientific tasks to the verified original Agent harness.

Default: verify inputs and print an offline plan. Execution requires --execute,
Docker, an already running original evaluator and explicit Agent/judge endpoints.
No question reconstruction, host generated-code fallback or skip-judge mode.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit
from urllib.request import urlopen

from prepare_naturebench_native_runtime import verify_runtime
from vllm_hust_workload_data.naturebench import original_cpu_packages


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "configs/workload-native/naturebench-original-release.json"


def endpoint(value):
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("explicit HTTP endpoint without embedded credentials required")
    parsed.port  # Validate malformed port syntax without making a connection.
    return value.rstrip("/")


def plan(args):
    context_tokens = getattr(args, "model_context_tokens", None)
    if context_tokens is not None and context_tokens <= 0:
        raise ValueError("positive actual served model context required")
    source = verify_runtime(args.author_root, args.runtime)
    runtime_profile = json.loads(
        (args.runtime / "runtime-profile.json").read_text("utf-8")
    )
    release = json.loads(RELEASE.read_text("utf-8"))
    tasks = args.task_id or release["official_cpu_tasks"]
    if (
        not tasks
        or len(set(tasks)) != len(tasks)
        or set(tasks) - set(release["official_cpu_tasks"])
    ):
        raise ValueError("select distinct original CPU task IDs; no replacement tasks")
    if not args.model.strip() or not (args.judge_model or args.model).strip():
        raise ValueError("explicit Agent and judge models required")
    if not 1 <= args.eval_port <= 65535:
        raise ValueError("valid original evaluator port required")
    data = args.data_root.resolve(strict=True)
    output = args.output.resolve()
    for root in [args.author_root.resolve(), args.runtime.resolve(), data]:
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError(
                "output must be separate from source,runtime and task data"
            )
    if output.exists():
        raise FileExistsError(
            "fresh output required; inspect prior native runs,do not restart"
        )
    packages = list(original_cpu_packages(data, task_ids=tasks))
    if {item["task_id"] for item in packages} != set(tasks):
        raise ValueError("complete original selected task packages required")
    mapping = (args.eval_mapping or source / "eval_env_mapping.json").resolve(
        strict=True
    )
    command = [
        sys.executable,
        str(source / "run_naturebench.py"),
        "--tasks",
        str(output / "selected-original-taskset.txt"),
        "--data-dir",
        str(data),
        "--skip-download",
        "--build-task-images",
        "--agent",
        "claude",
        "--model",
        args.model,
        "--mode",
        "base",
        "--timeout",
        "14400",
        "--setup-timeout",
        "14400",
        "--max-workers",
        "1",
        "--eval-port",
        str(args.eval_port),
        "--eval-env-mapping",
        str(mapping),
        "--proxy-mode",
        "none",
        "--out-dir",
        str(output / "native"),
    ]
    control_token_file = getattr(args, "eval_control_token_file", None)
    if control_token_file is not None:
        if control_token_file.is_symlink() or not control_token_file.is_file():
            raise ValueError(
                "existing regular private evaluator control-token file required"
            )
        control_token_file = control_token_file.resolve(strict=True)
        if control_token_file.stat().st_size == 0:
            raise ValueError("nonempty evaluator control-token file required")
        # The author explicitly supports forwarding solve.py options after --.
        # Pass only the file path, never its credential content, in the command.
        command.extend(["--", "--eval-control-token-file", str(control_token_file)])
    return {
        "source_id": "NATUREBENCH",
        "status": "VERIFIED_ORIGINAL_CPU_NATIVE_PLAN_NOT_EXECUTED",
        "code_revision": release["code_revision"],
        "dataset_revision": release["dataset_revision"],
        "task_ids": tasks,
        "original_selected_packages_verified": len(packages),
        "agent": "claude",
        "agent_model": args.model,
        "judge_model": args.judge_model or args.model,
        "caller_declared_served_context_tokens": context_tokens,
        "context_capacity_discovered_from_endpoint": False,
        "runtime_context_capacity_forwarded": runtime_profile[
            "network_profile"
        ].endswith("-context"),
        "agent_endpoint": endpoint(args.endpoint),
        "judge_endpoint": endpoint(args.judge_endpoint or args.endpoint),
        "task_budget_seconds": 14400,
        "setup_budget_seconds": 14400,
        "skip_judge": False,
        "native_command": command,
        "eval_control_token_file": (
            str(control_token_file) if control_token_file else None
        ),
        "model_calls_by_plan": 0,
        "source_checkbox_eligible": False,
    }


def execute(args, prepared):
    required = ["ANTHROPIC_API_KEY", "JUDGE_API_KEY"]
    if any(not os.environ.get(key, "").strip() for key in required):
        raise ValueError(
            "set explicit ANTHROPIC_API_KEY and JUDGE_API_KEY; never borrow default credentials"
        )
    if prepared["caller_declared_served_context_tokens"] is None:
        raise ValueError(
            "declare --model-context-tokens from actual service configuration before execution"
        )
    if not prepared["runtime_context_capacity_forwarded"]:
        raise ValueError(
            "prepare original-context or owned-hostnet-context runtime so the original task container receives the declared capacity"
        )
    # Only a trusted harness runs here; generated scientific programs stay in
    # the original task Docker containers. Do not fall back to host execution.
    subprocess.run(
        ["docker", "info", "--format", "{{.ID}}"], check=True, capture_output=True
    )
    with urlopen(f"http://127.0.0.1:{args.eval_port}/health", timeout=10) as response:
        if response.status != 200 or json.load(response).get("status") != "ok":
            raise ValueError("start the unchanged original evaluator before execution")
    environment = dict(os.environ)
    for key in [
        "JUDGE_OPENAI_BASE_URL",
        "JUDGE_ANTHROPIC_BASE_URL",
        "JUDGE_OPENAI_API_KEY",
        "JUDGE_ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_API_BASE",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW",
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
        "DISABLE_COMPACT",
        "CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT",
    ]:
        environment.pop(key, None)
    environment.update(
        ANTHROPIC_BASE_URL=prepared["agent_endpoint"],
        ANTHROPIC_MODEL=prepared["agent_model"],
        JUDGE_BASE_URL=prepared["judge_endpoint"],
        JUDGE_MODEL=prepared["judge_model"],
        CLAUDE_CODE_MAX_CONTEXT_TOKENS=str(
            prepared["caller_declared_served_context_tokens"]
        ),
    )
    output = args.output.resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    with (output / "selected-original-taskset.txt").open(
        "x", encoding="utf-8"
    ) as stream:
        stream.write("\n".join(prepared["task_ids"]) + "\n")
    receipt = dict(prepared, status="ORIGINAL_HARNESS_STARTED_NOT_COMPLETED")
    receipt_path = output / "execution-public.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    try:
        with (
            (output / "native-stdout-private.log").open("xb") as stdout,
            (output / "native-stderr-private.log").open("xb") as stderr,
        ):
            result = subprocess.run(
                prepared["native_command"],
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        receipt.update(
            status="ORIGINAL_HARNESS_TERMINAL_REVIEW_NATIVE_RESULTS",
            native_exit_code=result.returncode,
            native_exit_zero_is_not_source_completion=True,
        )
        return result.returncode
    except BaseException as error:
        receipt.update(
            status="ORIGINAL_HARNESS_INTERRUPTED_INSPECT_BEFORE_RESTART",
            error_type=type(error).__name__,
        )
        raise
    finally:
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--model-context-tokens",
        type=int,
        help="Actual combined input/output context limit of deployed service; required by execute, not an output-budget override",
    )
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-endpoint")
    parser.add_argument(
        "--task-id",
        action="append",
        help="Original CPU ID; default is complete CPU3 group",
    )
    parser.add_argument("--eval-port", type=int, default=8321)
    parser.add_argument("--eval-mapping", type=Path)
    parser.add_argument(
        "--eval-control-token-file",
        type=Path,
        help="Existing private token file matching the already running original evaluator; contents are never printed",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    prepared = plan(args)
    if args.execute:
        raise SystemExit(execute(args, prepared))
    print(json.dumps(prepared, indent=2))


if __name__ == "__main__":
    main()
