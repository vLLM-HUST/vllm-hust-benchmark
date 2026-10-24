"""Backend-independent SWE-bench Pro V1 orchestration around pinned upstream code.

Model serving and Controller policies belong to their own repositories. This
module does not start/reconfigure a model or require a Controller installation.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import importlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import threading
import time
import traceback

from .swe_bench_pro_inputs import (
    LOCK_PATH,
    digest,
    grade_saved_batch,
    prepare,
    read,
    save,
    validate_grading_sources,
)


class AgentWallBudgetExceeded(BaseException):
    """End the frozen run even if the Agent recovers ordinary tool errors."""


def _wall_budget_expired(signum, frame):
    raise AgentWallBudgetExceeded("Frozen Agent wall budget reached")


def validate(configuration):
    c = configuration
    if c.get("schema") != "swe-bench-pro-experiment-v1":
        raise ValueError("Unsupported SWE-bench Pro configuration schema")
    if not c.get("run_id"):
        raise ValueError("Explicit unique run_id required")
    if type(c.get("image_registry_insecure", False)) is not bool:
        raise ValueError("image_registry_insecure must be an explicit boolean")
    batch = c["batch"]
    if batch["arrival_time"] != 0 or batch["client_concurrency_limit"] is not None:
        raise ValueError("Finite batch must arrive at zero without client slots")
    for key, value in [
        ("batch.size", batch["size"]),
        ("agent.step_limit", c["agent"]["step_limit"]),
        ("agent.wall_budget_seconds", c["agent"]["wall_budget_seconds"]),
        ("agent.max_tokens", c["agent"]["max_tokens"]),
        ("agent.tool_timeout_seconds", c["agent"]["tool_timeout_seconds"]),
    ]:
        if type(value) is not int or value <= 0:
            raise ValueError("Explicit positive integer required: " + key)
    copies = batch.get("independent_agents_per_task", 1)
    if type(copies) is not int or copies <= 0:
        raise ValueError("Independent copies must be a positive integer")
    if batch.get("program_instances", batch["size"] * copies) != batch["size"] * copies:
        raise ValueError("Task count and program count differ")
    ids = batch.get("instance_ids")
    if ids is not None and (len(ids) != batch["size"] or len(set(ids)) != len(ids)):
        raise ValueError("Explicit task ids must match batch size, without duplicates")
    if c["agent"]["automatic_retries"] != 0:
        raise ValueError("Automatic model retries must be zero")
    instructions = c["agent"].get("task_instructions")
    if instructions is not None and (
        not isinstance(instructions, str) or not instructions.strip()
    ):
        raise ValueError("Optional task_instructions must be a nonempty string")
    timeouts = c["agent"]["http_timeouts"]
    if set(timeouts) != {"connect", "read", "write", "pool"}:
        raise ValueError("Record all four HTTP timeout phases explicitly")
    for key, value in timeouts.items():
        if value is None and key == "read":
            continue
        if type(value) not in (int, float) or value <= 0:
            raise ValueError("Invalid HTTP timeout: " + key)
    if not c["model"]["api_base"].startswith(("http://", "https://")):
        raise ValueError("Explicit OpenAI-compatible API base required")
    if not c["model"]["model_name"]:
        raise ValueError("Explicit served model name required")
    allowed = c.get("container", {}).get("allowed_initial_untracked", [])
    if not isinstance(allowed, list) or any(
        not isinstance(name, str)
        or not name.strip("/")
        or name.startswith("/")
        or any(part in (".", "..", ".git") for part in name.split("/"))
        or any(char in name for char in "\n\r*?[")
        for name in allowed
    ):
        raise ValueError("Initial untracked allowlist requires exact relative paths")
    for name in ("dataset_parquet", "agent_source", "official_v1_source", "docker"):
        if not Path(c["paths"][name]).is_absolute():
            raise ValueError("Absolute runtime path required: " + name)
    return c


def hooks(c):
    target = c.get("integration", {}).get("hook")
    if target is None:
        return None
    module, name = target.split(":", 1)
    return getattr(importlib.import_module(module), name)()


def call_hook(hook, method, *args):
    if hook is not None and hasattr(hook, method):
        return getattr(hook, method)(*args)
    return None


def validate_initial_repository(before, base_commit, allowed_untracked=()):
    lines = before["output"].splitlines()
    if (
        before["returncode"]
        or not lines
        or lines[0] != base_commit
        or any(
            line not in {"?? " + name for name in allowed_untracked}
            for line in lines[1:]
        )
    ):
        raise ValueError("Task container must start at the clean official base commit")


def verify_inputs(c, folder):
    """Accept existing private material only after revalidating pinned sources."""
    import pyarrow.parquet as parquet

    folder = Path(folder)
    manifest = read(folder / "input-selection.json")
    lock = read(LOCK_PATH)
    if (
        manifest["status"] != "PREPARED"
        or digest(folder / "cases-private.jsonl") != manifest["cases_sha256"]
    ):
        raise ValueError("Prepared input identity mismatch")
    if digest(c["paths"]["dataset_parquet"]) != lock["dataset"]["parquet_sha256"]:
        raise ValueError("Frozen V1 parquet changed")
    for relative, expected in lock["agent_source_files"].items():
        if digest(Path(c["paths"]["agent_source"]) / relative) != expected:
            raise ValueError("Pinned official Agent source changed: " + relative)
    rows = [
        json.loads(x) for x in (folder / "cases-private.jsonl").read_text().splitlines()
    ]
    if len(rows) != c["batch"]["size"] or len(rows) != len(manifest["tasks"]):
        raise ValueError("Prepared task count differs from requested batch")
    requested = c["batch"].get("instance_ids")
    if requested and sorted(requested) != sorted(x["instance_id"] for x in rows):
        raise ValueError("Prepared task ids differ from requested selection")
    all_rows = {
        x["instance_id"]: x
        for x in parquet.read_table(c["paths"]["dataset_parquet"]).to_pylist()
    }
    os.environ["DOCKER_HOST"] = c["paths"]["docker_host"]
    for row, task in zip(rows, manifest["tasks"]):
        if (
            all_rows.get(row["instance_id"]) != row
            or row["instance_id"] != task["instance_id"]
        ):
            raise ValueError("Prepared row is not the frozen dataset row")
        inspected = json.loads(
            subprocess.check_output(
                [c["paths"]["docker"], "image", "inspect", task["image_id"]]
            )
        )[0]
        if inspected["Architecture"] != "arm64" or inspected["Id"] != task["image_id"]:
            raise ValueError("Immutable ARM image mismatch")
    validate_grading_sources(c["paths"]["official_v1_source"], rows, lock)
    return {
        "tasks": len(rows),
        "cases_sha256": manifest["cases_sha256"],
        "dataset_sha256": lock["dataset"]["parquet_sha256"],
        "official_sources": "PASS",
    }


def adopt_inputs(c, existing, output):
    import shutil

    proof = verify_inputs(c, existing)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("cases-private.jsonl", "input-selection.json"):
        shutil.copyfile(Path(existing) / name, output / name)
    save(output / "reuse-verification.json", proof)
    return proof


def create_task_environment(c, image, factory):
    """Keep the sandbox alive through Agent budget, preparation and cleanup.

    The 40-minute historical minimum remains for short runs. The extra ten
    minutes cover the existing preparation barrier and evidence collection;
    they do not extend the Agent's own wall-clock budget.
    """
    wall = c["agent"]["wall_budget_seconds"]
    if type(wall) is not int or wall <= 0:
        raise ValueError("Positive Agent wall budget required for container lifetime")
    return factory(
        image=image["image_id"],
        cwd="/app",
        executable=c["paths"]["docker"],
        timeout=c["agent"]["tool_timeout_seconds"],
        container_timeout=f"{max(2400, wall + 600)}s",
        run_args=[
            "--rm",
            "--entrypoint",
            "",
            "--network",
            c["container"]["network"],
            "--cpus",
            str(c["container"]["cpus"]),
            "--memory",
            c["container"]["memory"],
        ],
    )


def _agent_task(c, row, image, folder, barrier):
    folder = Path(folder)
    folder.mkdir(exist_ok=False)
    os.environ.update(
        DOCKER_HOST=c["paths"]["docker_host"],
        MSWEA_SILENT_STARTUP="1",
        MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT="1",
        MSWEA_GLOBAL_CONFIG_DIR=str(folder / "agent-config"),
        LITELLM_LOCAL_MODEL_COST_MAP="True",
    )
    sys.path.insert(0, str(Path(c["paths"]["agent_source"]) / "src"))
    import httpx
    import yaml
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.docker import DockerEnvironment
    from minisweagent.models.litellm_model import LitellmModel
    from minisweagent.run.utils.save import save_traj
    from .swe_bench_pro_timing import instrument
    from functools import partial

    cfg = yaml.safe_load(
        (
            Path(c["paths"]["agent_source"])
            / "src/minisweagent/config/extra/swebench.yaml"
        ).read_text()
    )
    agent_cfg = {
        k: v.replace("/testbed", "/app") if isinstance(v, str) else v
        for k, v in cfg["agent"].items()
    }
    agent_cfg.update(step_limit=c["agent"]["step_limit"], cost_limit=0)
    formatter = runpy.run_path(
        str(
            Path(c["paths"]["official_v1_source"])
            / "helper_code/create_problem_statement.py"
        )
    )
    task = formatter["create_problem_statement"](row)
    if c["agent"].get("task_instructions"):
        task += "\n\n" + c["agent"]["task_instructions"]
    (folder / "task-private.txt").write_text(task)
    program = c["run_id"] + ":" + folder.name
    hook = hooks(c)
    env = agent = None
    began = None
    status, result, error, patch = "PREPARATION_FAILED", "", None, ""
    stop = threading.Event()
    observer = None
    try:
        env = create_task_environment(c, image, DockerEnvironment)
        (folder / "container-id.txt").write_text(env.container_id)
        before = env.execute("git rev-parse HEAD && git status --porcelain")
        save(folder / "code-before.json", before)
        validate_initial_repository(
            before,
            row["base_commit"],
            c["container"].get("allowed_initial_untracked", []),
        )
        inspected = json.loads(
            subprocess.check_output([c["paths"]["docker"], "inspect", env.container_id])
        )[0]
        if inspected["Mounts"] or inspected["Image"] != image["image_id"]:
            raise ValueError("Independent immutable task container required")
        save(folder / "container-inspect-private.json", inspected)
        api_key = (
            os.environ[c["model"]["api_key_env"]]
            if c["model"].get("api_key_env")
            else "EMPTY"
        )
        kwargs = {
            "api_base": c["model"]["api_base"],
            "api_key": api_key,
            "temperature": c["agent"]["temperature"],
            "seed": c["agent"]["seed"],
            "max_tokens": c["agent"]["max_tokens"],
            "timeout": c["agent"]["http_timeouts"]["connect"],
            "num_retries": 0,
            "extra_headers": {"X-Session-ID": program},
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": c["agent"]["thinking"]}
            },
        }
        model = LitellmModel(
            model_name=c["model"]["model_name"],
            cost_tracking="ignore_errors",
            model_kwargs=kwargs,
        )
        model.query = partial(
            model.query, timeout=httpx.Timeout(**c["agent"]["http_timeouts"])
        )
        # Credentials remain usable by the model but are redacted when saving config.
        call_hook(hook, "before_program", c, folder, program)
        instrument(model, env, folder, program)
        agent = DefaultAgent(model, env, **agent_cfg)

        def snapshot(path, **values):
            copied_agent = copy.copy(agent)
            copied_model = copy.copy(model)
            safe_kwargs = dict(model.config.model_kwargs)
            safe_kwargs["api_key"] = "REDACTED"  # pragma: allowlist secret
            copied_model.config = dataclasses.replace(
                model.config, model_kwargs=safe_kwargs
            )
            copied_agent.model = copied_model
            save_traj(copied_agent, path, print_path=False, **values)

        save(
            folder / "http-client-configuration.json",
            {"timeouts": c["agent"]["http_timeouts"], "num_retries": 0},
        )
        save(
            folder / "ready.json",
            {"instance_id": row["instance_id"], "ready_monotonic": time.monotonic()},
        )
        barrier.wait(timeout=180)
        began = time.monotonic()
        save(
            folder / "arrival.json",
            {
                "scheduled_arrival_seconds": 0,
                "epoch": time.time(),
                "actual_start_monotonic": began,
            },
        )

        def observe():
            previous = -1
            while not stop.wait(2):
                try:
                    if agent.model.n_calls != previous:
                        previous = agent.model.n_calls
                        snapshot(folder / "trajectory-live-private.json")
                        save(
                            folder / "progress-private.json",
                            {
                                "model_calls": previous,
                                "elapsed_seconds": time.monotonic() - began,
                            },
                        )
                        print("AGENT_PROGRESS", program, previous, flush=True)
                except Exception:
                    (folder / "observer-error.txt").write_text(traceback.format_exc())

        signal.signal(signal.SIGALRM, _wall_budget_expired)
        signal.alarm(c["agent"]["wall_budget_seconds"])
        observer = threading.Thread(target=observe, daemon=True)
        observer.start()
        try:
            status, result = agent.run(task)
        except (Exception, AgentWallBudgetExceeded) as exc:
            status, error = type(exc).__name__, str(exc)
            (folder / "agent-exception-private.txt").write_text(traceback.format_exc())
        finally:
            signal.alarm(0)
        if status == "Submitted":
            patch = result
        else:
            extraction = env.execute("git diff --binary HEAD")
            save(folder / "partial-extraction-private.json", extraction)
            patch = extraction["output"] if extraction["returncode"] == 0 else ""
        save(
            folder / "working-tree-status-private.json",
            env.execute("git status --porcelain"),
        )
    except Exception as exc:
        error = str(exc)
        (folder / "failure-private.txt").write_text(traceback.format_exc())
        try:
            barrier.abort()
        except Exception:
            pass
    finally:
        signal.alarm(0)
        stop.set()
        if observer:
            observer.join(timeout=5)
        ended = time.monotonic()
        if agent:
            try:
                snapshot(
                    folder / "trajectory-private.json",
                    exit_status=status,
                    result=result,
                )
            except Exception:
                (folder / "trajectory-save-error.txt").write_text(
                    traceback.format_exc()
                )
        (folder / "patch-private.diff").write_text(patch)
        report = {
            "instance_id": row["instance_id"],
            "program_id": program,
            "status": status,
            "error": error,
            "agent_submitted": status == "Submitted",
            "started_monotonic": began,
            "finished_monotonic": ended,
            "elapsed_seconds": ended - began if began is not None else None,
            "model_calls": model.n_calls if agent else 0,
            "runner_module": __file__,
            "runner_sha256": digest(__file__),
            "controller_modules_loaded": [
                name
                for name in sys.modules
                if name.startswith("thunderagent_controller_mod")
            ],
            "patch_bytes": len(patch.encode()),
            "patch_sha256": digest(folder / "patch-private.diff"),
        }
        try:
            call_hook(hook, "after_program", c, folder, program, report)
        except Exception as exc:
            report["integration_error"] = str(exc)
        if env and env.container_id:
            removed = subprocess.run(
                [c["paths"]["docker"], "rm", "-f", env.container_id],
                capture_output=True,
            )
            report["container_cleanup_exit_code"] = removed.returncode
            env.container_id = None
        save(folder / "report-private.json", report)
        print("AGENT_TERMINAL", json.dumps(report), flush=True)


def run(c, inputs, output):
    validate(c)
    proof = verify_inputs(c, inputs)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "submission-intent.json").open("x").write(
        json.dumps({"run_id": c["run_id"], "epoch": time.time()})
    )
    save(output / "settings-private.json", c)
    save(output / "input-verification.json", proof)
    save(
        output / "runner-provenance.json",
        {
            "module": __file__,
            "source_sha256": digest(__file__),
            "benchmark_version": importlib.metadata.version("vllm-hust-benchmark")
            if _installed()
            else "source-checkout",
        },
    )
    hook = hooks(c)
    call_hook(hook, "before_run", c, output)
    rows = [
        json.loads(x)
        for x in (Path(inputs) / "cases-private.jsonl").read_text().splitlines()
    ]
    images = read(Path(inputs) / "input-selection.json")["tasks"]
    expanded = [
        (row, image)
        for _ in range(c["batch"].get("independent_agents_per_task", 1))
        for row, image in zip(rows, images)
    ]
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(len(expanded) + 1)
    processes = [
        context.Process(
            target=_agent_task, args=(c, row, image, output / f"task-{i:03d}", barrier)
        )
        for i, (row, image) in enumerate(expanded)
    ]
    try:
        for process in processes:
            process.start()
        barrier.wait(timeout=180)
        save(
            output / "batch-start.json",
            {
                "epoch": time.time(),
                "program_instances": len(expanded),
                "arrival_time": 0,
                "client_concurrency_limit": None,
            },
        )
        deadline = time.monotonic() + c["agent"]["wall_budget_seconds"] + 180
        while any(p.is_alive() for p in processes) and time.monotonic() < deadline:
            time.sleep(1)
        if any(p.is_alive() for p in processes):
            raise TimeoutError("Agent batch drain deadline exceeded")
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
        # Only remove containers whose immutable ids were created by this run.
        for identifier in output.glob("task-*/container-id.txt"):
            container_id = identifier.read_text().strip()
            if container_id:
                subprocess.run(
                    [c["paths"]["docker"], "rm", "-f", container_id],
                    capture_output=True,
                )
        call_hook(hook, "after_run", c, output)
    reports = [
        read(output / f"task-{i:03d}/report-private.json") for i in range(len(expanded))
    ]
    summary = {
        "run_id": c["run_id"],
        "program_instances": len(reports),
        "submitted": sum(x["agent_submitted"] for x in reports),
        "tasks": reports,
        "grading_status": "DEFERRED",
        "quality_score": None,
    }
    save(output / "batch-report.json", summary)
    return summary


def _installed():
    try:
        importlib.metadata.version("vllm-hust-benchmark")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=["check", "prepare", "verify-inputs", "run", "grade", "summarize"],
    )
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--reuse-inputs", type=Path)
    parser.add_argument("--batch-directory", type=Path)
    args = parser.parse_args(argv)
    c = validate(read(args.settings))
    if args.action == "check":
        result = {
            "configuration": "PASS",
            "program_instances": c["batch"]["size"]
            * c["batch"].get("independent_agents_per_task", 1),
        }
    elif args.action == "prepare":
        result = (
            adopt_inputs(c, args.reuse_inputs, args.output)
            if args.reuse_inputs
            else prepare(c, args.output)
        )
    elif args.action == "verify-inputs":
        if args.inputs is None:
            parser.error("--inputs is required")
        result = verify_inputs(c, args.inputs)
    elif args.action == "run":
        if args.inputs is None:
            parser.error("--inputs is required")
        result = run(c, args.inputs, args.output)
    elif args.action == "grade":
        if args.inputs is None or args.batch_directory is None:
            parser.error("--inputs and --batch-directory are required")
        verify_inputs(c, args.inputs)
        result = grade_saved_batch(c, args.inputs, args.batch_directory, args.output)
    else:
        result = read(args.output / "batch-report.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
