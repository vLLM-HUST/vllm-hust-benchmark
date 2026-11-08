"""Run pinned original OSWorld tasks, Qwen3VL Agent and single-task grader.

Default is a source-verified offline plan. No questions, setup steps, screenshots,
references, metrics or generation budgets are reconstructed. Model calls use the
author-supported OpenAI backend with the actual explicitly served model name.
"""

import argparse
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from check_osworld_author_client import PROFILE, run as check_client
from osworld_cdp_wait_compat import cdp_wait_playwright
from osworld_owned_qemu_provider import OwnedQemuProvider, validate_profile
from osworld_qwen3vl_reset_compat import reset_compatible_agent
from osworld_sentinel_description_compat import sentinel_description_compatible_agent
from osworld_screenshot_wait_compat import (
    screenshot_wait_requests,
    setup_readiness_requests,
)
from prepare_jsonschemabench_author_source import verify_source

ROOT = Path(__file__).resolve().parents[1]


def plan(args):
    source = verify_source(args.author_root, PROFILE)
    pins = json.loads(PROFILE.read_text("utf-8"))
    identities = {row["path"]: row["sha256"] for row in pins["files"]}
    catalog = json.loads(
        (source / "evaluation_examples/test_all.json").read_text("utf-8")
    )
    tasks = []
    for domain, names in catalog.items():
        for name in names:
            if not all(
                isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+", value)
                for value in [domain, name]
            ):
                raise ValueError("contained original domain and case IDs required")
            relative = f"evaluation_examples/examples/{domain}/{name}.json"
            raw = (source / relative).read_bytes()
            if (
                hashlib.sha256(raw).hexdigest() != identities[relative]
                or json.loads(raw)["id"] != name
            ):
                raise ValueError("original task identity differs")
            tasks.append(
                {
                    "domain": domain,
                    "id": name,
                    "path": relative,
                    "sha256": identities[relative],
                }
            )
    if args.case_index is not None:
        if type(args.case_index) is not int or not 0 <= args.case_index < min(
            3, len(tasks)
        ):
            raise ValueError(
                "case-index selects only the fixed first three original cases"
            )
        selected = [tasks[args.case_index]]
    elif args.all_tasks:
        selected = tasks
    else:
        if type(args.limit) is not int or not 1 <= args.limit <= len(tasks):
            raise ValueError("positive bounded original-task prefix required")
        selected = tasks[: args.limit]
    parsed = urlsplit(args.api_base)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") != "/v1"
    ):
        raise ValueError(
            "explicit OpenAI-compatible /v1 endpoint without embedded credentials required"
        )
    parsed.port
    if not isinstance(args.model, str) or not args.model.strip():
        raise ValueError("actual served model name required")
    output = args.output.resolve()
    if (
        args.output.is_symlink()
        or output.exists()
        or any(
            output.is_relative_to(root) or root.is_relative_to(output)
            for root in [source, ROOT]
        )
    ):
        raise ValueError(
            "fresh output outside original source and benchmark checkout required"
        )
    owned = None
    if args.provider == "owned-qemu":
        if args.owned_qemu_profile is None:
            raise ValueError("explicit owned-QEMU profile required")
        owned = json.loads(args.owned_qemu_profile.read_text("utf-8"))
        validate_profile(owned)
        if any(task not in tasks[:3] for task in selected):
            raise ValueError(
                "owned-QEMU qualification is limited to the fixed first three cases"
            )
    elif args.owned_qemu_profile is not None or args.slow_arm_tcg:
        raise ValueError("owned runtime compatibility requires provider=owned-qemu")
    elif not args.vm_path:
        raise ValueError(
            "explicit disposable author VM path required; no implicit VM allocation"
        )
    return {
        "source_id": "OSWORLD",
        "status": "VERIFIED_ORIGINAL_TASK_PLAN_NOT_EXECUTED",
        "source_revision": pins["revision"],
        "source_root": str(source),
        "tasks": selected,
        "api_base": args.api_base.rstrip("/"),
        "model": args.model,
        "provider": args.provider,
        "owned_profile": owned,
        "slow_arm_tcg": args.slow_arm_tcg,
        "max_steps": 15,
        "max_tokens": 32768,
        "reset_signature_compatibility": True,
        "native_sentinel_description_compatibility": True,
        "native_tasks": 0,
        "source_checkbox_eligible": False,
    }


def execute(args, prepared):
    check_client(Path(prepared["source_root"]), "check-imports")
    key = (
        os.environ[args.api_key_env]
        if args.api_key_env
        else "local-runtime-not-a-secret"
    )  # pragma: allowlist secret - explicit no-auth placeholder
    with urlopen(
        Request(
            prepared["api_base"] + "/models", headers={"Authorization": "Bearer " + key}
        ),
        timeout=30,
    ) as response:
        models = json.load(response)["data"]
    if not any(row["id"] == args.model for row in models):
        raise ValueError("actual endpoint must expose the selected model ID")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    report = dict(prepared, status="ORIGINAL_TASKS_RUNNING", completed_tasks=[])
    report_path = output / "execution-public.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    old_path = list(sys.path)
    saved_env = {
        name: os.environ.get(name) for name in ["OPENAI_BASE_URL", "OPENAI_API_KEY"]
    }
    sys.path.insert(0, prepared["source_root"])
    try:
        import desktop_env.desktop_env as environment
        import desktop_env.controllers.python as controller
        import desktop_env.controllers.setup as setup
        import desktop_env.evaluators.getters.chrome as chrome_getters
        from mm_agents.qwen3vl_agent import Qwen3VLAgent
        import lib_run_single
    except BaseException as error:
        report.update(
            status="ORIGINAL_CLIENT_IMPORT_FAILED", error_type=type(error).__name__
        )
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        sys.path[:] = old_path
        raise
    saved_factory, saved_controller_requests, saved_setup_requests = (
        environment.create_vm_manager_and_provider,
        controller.requests,
        setup.requests,
    )
    saved_playwright = chrome_getters.sync_playwright
    try:
        os.environ["OPENAI_BASE_URL"], os.environ["OPENAI_API_KEY"] = (
            prepared["api_base"],
            key,
        )
        profile = prepared["owned_profile"]
        if profile:

            def factory(provider_name, region, use_proxy=False):
                if provider_name != "docker" or use_proxy:
                    raise ValueError(
                        "owned provider only supports the declared local nonproxy profile"
                    )
                return None, OwnedQemuProvider(profile)

            environment.create_vm_manager_and_provider = factory
            if args.slow_arm_tcg:
                chrome_getters.sync_playwright = cdp_wait_playwright(
                    saved_playwright,
                    endpoint=f"http://127.0.0.1:{profile['ports']['chromium']}",
                )
                base = f"http://127.0.0.1:{profile['ports']['server']}"
                controller.requests = screenshot_wait_requests(
                    controller.requests, controller_base=base
                )
                setup.requests = setup_readiness_requests(
                    setup.requests, controller_base=base, screenshot_only=True
                )
        env = environment.DesktopEnv(
            provider_name="docker",
            path_to_vm="operator-owned-qemu" if profile else args.vm_path,
            snapshot_name=profile["snapshot_name"] if profile else "init_state",
            action_space="pyautogui",
            cache_dir=str(output / "cache"),
            screen_size=(1920, 1080),
            headless=True,
            require_a11y_tree=False,
            os_type="Ubuntu",
        )
        try:
            agent = reset_compatible_agent(
                sentinel_description_compatible_agent(Qwen3VLAgent)
            )(
                model=args.model,
                max_tokens=32768,
                action_space="pyautogui",
                observation_type="screenshot",
                api_backend="openai",
            )
            original_args = SimpleNamespace(
                result_dir=str(output),
                model=args.model,
                action_space="pyautogui",
                observation_type="screenshot",
                sleep_after_execution=0.0,
                max_steps=15,
                max_tokens=32768,
            )
            for task in prepared["tasks"]:
                example = json.loads(
                    (Path(prepared["source_root"]) / task["path"]).read_text("utf-8")
                )
                destination = output / task["domain"] / task["id"]
                destination.mkdir(parents=True, exist_ok=False)
                scores = []
                lib_run_single.run_single_example(
                    agent,
                    env,
                    example,
                    15,
                    example["instruction"],
                    original_args,
                    str(destination),
                    scores,
                )
                if len(scores) != 1 or not math.isfinite(float(scores[0])):
                    raise ValueError(
                        "original finite task score required, not just runner return"
                    )
                report["completed_tasks"].append(
                    {
                        "id": task["id"],
                        "domain": task["domain"],
                        "original_score": float(scores[0]),
                    }
                )
                report["native_tasks"] = len(report["completed_tasks"])
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            report["status"] = (
                "ORIGINAL_TASK_RUNNERS_AND_GRADERS_RETURNED_REQUIRES_DELIVERY_AUDIT"
            )
        finally:
            env.close()
    except BaseException as error:
        report.update(
            status="ORIGINAL_EXECUTION_FAILED_RETAIN_PRIVATE_OUTPUTS",
            error_type=type(error).__name__,
        )
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        (
            environment.create_vm_manager_and_provider,
            controller.requests,
            setup.requests,
        ) = saved_factory, saved_controller_requests, saved_setup_requests
        sys.path[:] = old_path
        chrome_getters.sync_playwright = saved_playwright
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return report


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-key-env")
    parser.add_argument(
        "--provider", choices=["docker", "owned-qemu"], default="docker"
    )
    parser.add_argument("--vm-path")
    parser.add_argument("--owned-qemu-profile", type=Path)
    parser.add_argument("--slow-arm-tcg", action="store_true")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, default=3)
    selection.add_argument("--case-index", type=int)
    selection.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    prepared = plan(args)
    print(json.dumps(execute(args, prepared) if args.execute else prepared, indent=2))


if __name__ == "__main__":
    main()
