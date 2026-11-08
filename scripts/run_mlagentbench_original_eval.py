"""Run original saved-submission scoring, never rerun Agent or training.

The upstream eval module misses the typing.Dict annotation import. Supply only
that name to its execution namespace; retain original source and scoring rules.
"""

import argparse
import json
import math
from pathlib import Path
import runpy
import sys
from typing import Dict

from run_mlagentbench_local_agent import check_isolation, verify_port


def scored_outputs_valid(payload):
    if not isinstance(payload, dict) or not payload:
        return False
    for result in payload.values():
        if not isinstance(result, dict):
            return False
        values = result.get("score")
        final = result.get("final_score")
        if (
            not isinstance(values, list)
            or not values
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
            or type(final) not in (int, float)
            or not math.isfinite(final)
            or final != values[-1]
        ):
            return False
    return True


def run(args):
    report = {
        "source_id": "MLAGENTBENCH",
        "status": "PLAN_ONLY_NO_SCORING",
        "model_calls": 0,
    }
    if not args.execute:
        return report
    if not Path("/.dockerenv").is_file():
        raise ValueError("dedicated isolated container required")
    check_isolation(Path("/proc/self/status").read_text())
    verify_port(args.port_root)
    logs = args.log_root.resolve(strict=True)
    for name in [
        "env_log/trace.json",
        "env_log/overall_time.txt",
        "log",
        "agent_log/main_log",
    ]:
        p = logs / name
        if (
            not p.is_file()
            or p.is_symlink()
            or not p.resolve(strict=True).is_relative_to(logs)
        ):
            raise ValueError("contained original trace and captured logs required")
    final_snapshot = logs / "env_log/traces/step_final_files"
    if (
        not final_snapshot.is_dir()
        or final_snapshot.is_symlink()
        or not final_snapshot.resolve(strict=True).is_relative_to(logs)
    ):
        raise ValueError("terminal original final snapshot required before grading")
    output = args.output.resolve()
    if (
        output.exists()
        or args.output.is_symlink()
        or any(output.is_relative_to(root) for root in [logs, args.port_root.resolve()])
    ):
        raise ValueError(
            "fresh private output separate from original logs/source required"
        )
    with output.open("x", encoding="utf-8"):
        pass
    saved_argv = sys.argv
    sys.path.insert(0, str(args.port_root.resolve()))
    try:
        sys.argv = [
            "MLAgentBench.eval",
            "--task",
            args.task,
            "--log-folder",
            str(logs),
            "--output-file",
            str(output),
        ]
        runpy.run_module(
            "MLAgentBench.eval", run_name="__main__", init_globals={"Dict": Dict}
        )
    finally:
        sys.argv = saved_argv
    valid = scored_outputs_valid(json.loads(output.read_text("utf-8")))
    return dict(
        report,
        status="ORIGINAL_SAVED_SUBMISSION_SCORE_VALID"
        if valid
        else "ORIGINAL_EMPTY_OR_INVALID_SCORE_NOT_DELIVERED",
        scored_outputs_valid=valid,
        annotation_compatibility="typing.Dict namespace binding only; source/scoring unchanged",
        output=str(output),
        source_checkbox_eligible=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    result = run(parser.parse_args())
    print(json.dumps(result, indent=2))
    if result.get("scored_outputs_valid") is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
