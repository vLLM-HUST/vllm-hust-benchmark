"""Run original default ResearchAgent through a private queue inside isolation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from appworld_model_transport import model_proxy
from prepare_mlagentbench_local_agent import PINS, ROOT, port_runner


def check_isolation(status):
    fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
    if (
        int(fields.get("CapEff", "1").strip(), 16) != 0
        or fields.get("NoNewPrivs", "0").strip() != "1"
    ):
        raise ValueError(
            "dedicated capability-dropped no-new-privileges container required"
        )


def original_arguments(args):
    return [
        "MLAgentBench.runner",
        "--task",
        "debug",
        "--agent-type",
        "ResearchAgent",
        "--log-dir",
        str(args.log_root),
        "--work-dir",
        str(args.work_root),
        "--llm-name",
        args.model,
        "--fast-llm-name",
        args.model,
        "--edit-script-llm-name",
        args.model,
        "--max-steps",
        "50",
        "--agent-max-steps",
        "50",
        "--max-time",
        "18000",
        "--edit-script-llm-max-tokens",
        "4000",
        "--max-retries",
        "5",
        "--max-steps-in-context",
        "3",
        "--max-observation-steps-in-context",
        "3",
    ]


def verify_port(root):
    root = root.resolve(strict=True)
    pins = json.loads(PINS.read_text("utf-8"))
    expected = {item["path"]: item for item in pins["files"]}
    for name, item in expected.items():
        path = root / name
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError("contained original/ported software required")
        raw = path.read_bytes()
        if name in {"MLAgentBench/LLM.py", "MLAgentBench/runner.py"}:
            backup = root / ("originals/" + name + ".source")
            if backup.is_symlink() or not backup.resolve(strict=True).is_relative_to(
                root
            ):
                raise ValueError("contained retained original required")
            original = backup.read_bytes()
            if hashlib.sha256(original).hexdigest() != item["sha256"]:
                raise ValueError("retained original source differs")
            target = (
                port_runner(original)
                if name.endswith("runner.py")
                else (ROOT / "scripts/mlagentbench_local_llm_binding.py")
                .read_text("utf-8")
                .encode()
            )
            if raw != target:
                raise ValueError("runtime port differs from disclosed implementation")
        elif hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError("original task/tool/Agent source differs")
    for name in ["mlagentbench_local_chat.py", "appworld_model_transport.py"]:
        path = root / name
        if (
            path.is_symlink()
            or not path.resolve(strict=True).is_relative_to(root)
            or path.read_bytes()
            != (ROOT / "scripts" / name).read_text("utf-8").encode()
        ):
            raise ValueError("public queue/local transport software differs")
    actual_python = {path.relative_to(root).as_posix() for path in root.rglob("*.py")}
    expected_python = {name for name in expected if name.endswith(".py")} | {
        "mlagentbench_local_chat.py",
        "appworld_model_transport.py",
    }
    if actual_python != expected_python:
        raise ValueError("unexpected Python source in prepared Agent workspace")


def run(args):
    report = {
        "source_id": "MLAGENTBENCH",
        "status": "PLAN_ONLY_NO_AGENT_OR_MODEL_EXECUTION",
        "native_agent_tasks_run": 0,
        "source_checkbox_eligible": False,
    }
    if not args.execute:
        return report
    if not Path("/.dockerenv").is_file():
        raise ValueError(
            "execution is allowed only in the dedicated isolated container"
        )
    check_isolation(Path("/proc/self/status").read_text())
    verify_port(args.port_root)
    for output in [args.work_root, args.log_root]:
        if output.exists() or output.is_symlink():
            raise ValueError(
                "fresh task work and log paths required; no implicit resume"
            )
        if output.resolve().is_relative_to(args.port_root.resolve()):
            raise ValueError("task output must not overwrite original Agent software")
    original_argv = sys.argv
    sys.path.insert(0, str(args.port_root.resolve()))
    os.environ["MLAGENTBENCH_LOCAL_MODEL"] = args.model
    try:
        with model_proxy(args.queue, port=14001, timeout=600):
            os.environ["MLAGENTBENCH_LOCAL_API_BASE"] = "http://127.0.0.1:14001/v1"
            sys.argv = original_arguments(args)
            runpy.run_module("MLAgentBench.runner", run_name="__main__", alter_sys=True)
    finally:
        sys.argv = original_argv
    report.update(
        status="ORIGINAL_AGENT_RUNNER_RETURNED_REQUIRES_TRACE_AND_ORIGINAL_SCORE_AUDIT",
        native_agent_tasks_run=1,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port-root", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--execute", action="store_true")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
