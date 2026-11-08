"""Run the original ResearchAgent with an independent native Messages profile."""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys

from prepare_mlagentbench_messages_agent import PROFILE, verify_messages_port
from run_mlagentbench_local_agent import check_isolation, original_arguments
import run_mlagentbench_original_eval as original_eval


def score(args):
    """Reuse original saved-submission grading, changing only port verification."""
    previous = original_eval.verify_port
    try:
        original_eval.verify_port = verify_messages_port
        return original_eval.run(args)
    finally:
        original_eval.verify_port = previous


def run(args):
    report = dict(
        source_id="MLAGENTBENCH",
        status="PLAN_ONLY_NO_MODEL_OR_AGENT_EXECUTION",
        profile=PROFILE,
        native_agent_tasks_run=0,
        source_checkbox_eligible=False,
    )
    if not args.execute:
        return report
    if not args.model or args.queue is None or args.work_root is None:
        raise ValueError(
            "model, queue and fresh work root required for Agent execution"
        )
    if not Path("/.dockerenv").is_file():
        raise ValueError("dedicated isolated container required")
    check_isolation(Path("/proc/self/status").read_text())
    verify_messages_port(args.port_root)
    if args.queue.is_symlink() or not args.queue.resolve(strict=True).is_dir():
        raise ValueError("dedicated private Messages queue required")
    root = args.port_root.resolve()
    outputs = [args.work_root.resolve(), args.log_root.resolve()]
    if outputs[0].is_relative_to(outputs[1]) or outputs[1].is_relative_to(outputs[0]):
        raise ValueError("independent work/log roots required")
    for path in outputs:
        if (
            path.exists()
            or path.is_symlink()
            or path.is_relative_to(root)
            or root.is_relative_to(path)
        ):
            raise ValueError("fresh outputs outside original/ported software required")
    old_path, old_argv = list(sys.path), sys.argv
    saved = {
        key: os.environ.get(key)
        for key in [
            "MLAGENTBENCH_LOCAL_MODEL",
            "MLAGENTBENCH_LOCAL_API_BASE",
            "MLAGENTBENCH_MESSAGES_QUEUE",
        ]
    }
    try:
        sys.path.insert(0, str(root))
        os.environ["MLAGENTBENCH_LOCAL_MODEL"] = args.model
        os.environ["MLAGENTBENCH_LOCAL_API_BASE"] = "http://127.0.0.1:14001/v1"
        os.environ["MLAGENTBENCH_MESSAGES_QUEUE"] = str(args.queue.resolve())
        sys.argv = original_arguments(args)
        runpy.run_module("MLAgentBench.runner", run_name="__main__", alter_sys=True)
    finally:
        sys.path[:] = old_path
        sys.argv = old_argv
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    report.update(
        status="ORIGINAL_RUNNER_RETURNED_REQUIRES_TRACE_AND_ORIGINAL_GRADING",
        native_agent_tasks_run=1,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--queue", type=Path)
    parser.add_argument(
        "--score-output",
        type=Path,
        help="Grade saved original final snapshot without new model calls",
    )
    parser.add_argument("--task", default="debug")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.score_output is not None:
        args.output = args.score_output
        result = score(args)
    else:
        result = run(args)
    print(json.dumps(result, indent=2))
    if result.get("scored_outputs_valid") is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
