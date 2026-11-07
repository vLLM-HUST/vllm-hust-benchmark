"""Run unchanged WebArena tasks through the author BrowserGym/AgentLab client.

Source and site preparation are separate. No website, model or Agent is started
without --execute. A finished task is not whole-source delivery qualification.
"""

import argparse
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import importlib.metadata as metadata
import json
import os
from pathlib import Path

from appworld_model_transport import model_proxy
from prepare_jsonschemabench_author_source import verify_source
from run_miniwob_agentlab import loopback_url, terminal_summary


PINS = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/webarena-author-source.json"
)
SITES = (
    "REDDIT",
    "SHOPPING",
    "SHOPPING_ADMIN",
    "GITLAB",
    "WIKIPEDIA",
    "MAP",
    "HOMEPAGE",
)
VERSIONS = {
    "agentlab": "0.4.2",
    "browsergym-webarena": "0.14.2",
    "libwebarena": "0.0.4",
}


def original_task(author_root, task_id):
    verify_source(author_root, PINS)
    tasks = json.loads(
        (Path(author_root) / "config_files/test.raw.json").read_text("utf-8")
    )
    if type(task_id) is not int or not 0 <= task_id < len(tasks):
        raise ValueError("original task ID outside the pinned source")
    task = tasks[task_id]
    if task["task_id"] != task_id:
        raise ValueError("original task ordering changed; review the source profile")
    return task


def site_configuration(path, required):
    configured = json.loads(Path(path).read_text("utf-8"))
    if not isinstance(configured, dict) or set(configured) - set(SITES):
        raise ValueError("site map must contain only original uppercase site names")
    needed = {site.upper() for site in required}
    if not needed <= set(configured) or not needed <= set(SITES):
        raise ValueError(
            "all sites required by the selected original task must be configured"
        )
    for value in configured.values():
        if not isinstance(value, str):
            raise ValueError("site URLs must be strings")
        loopback_url(value)
    # The original module asserts all seven variables exist at import time.
    # Unused sites are explicitly unavailable, never mapped onto another site.
    missing = sorted(set(SITES) - set(configured))
    return {
        name: configured.get(name, f"http://127.0.0.1:1/unconfigured/{name.lower()}")
        for name in SITES
    }, missing


def environment_bindings(sites):
    """Bind both original WebArena and official BrowserGym namespaces."""
    return {**sites, **{f"WA_{name}": url for name, url in sites.items()}}


@contextmanager
def browser_timeout_profile(spec, timeout_ms):
    """Use BrowserEnv's supported timeout option without changing task/scoring."""
    original = spec.kwargs
    try:
        if timeout_ms is not None:
            spec.kwargs = {**original, "timeout": timeout_ms}
        yield
    finally:
        spec.kwargs = original


def execute(args):
    if (
        args.author_root is None
        or args.sites is None
        or args.output is None
        or not args.model
        or not args.model.strip()
    ):
        raise ValueError(
            "original source, site map, fresh output and actual model required"
        )
    if (args.endpoint is None) == (args.queue is None):
        raise ValueError("choose one loopback endpoint or private model queue")
    if (
        args.max_steps <= 0
        or (args.max_output_tokens is not None and args.max_output_tokens <= 0)
        or args.seed < 0
    ):
        raise ValueError(
            "positive budgets required; omitted output budget preserves author default"
        )
    timeout_ms = getattr(args, "browser_timeout_ms", None)
    if timeout_ms is not None and (type(timeout_ms) is not int or timeout_ms <= 0):
        raise ValueError("positive browser timeout in milliseconds required")
    task = original_task(args.author_root, args.task_id)
    sites, unavailable = site_configuration(args.sites, task["sites"])
    root = args.author_root.resolve(strict=True)
    output = args.output.resolve()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or output.is_relative_to(root)
        or root.is_relative_to(output)
    ):
        raise ValueError("fresh output outside original source required")
    if args.queue is not None:
        queue = args.queue.resolve(strict=True)
        if (
            args.queue.is_symlink()
            or not queue.is_dir()
            or queue.is_relative_to(output)
            or output.is_relative_to(queue)
            or queue.is_relative_to(root)
            or root.is_relative_to(queue)
        ):
            raise ValueError(
                "existing dedicated private queue separate from outputs and source required"
            )
    if args.endpoint is not None and not args.endpoint.rstrip("/").endswith("/v1"):
        raise ValueError("explicit loopback endpoint ending in /v1 required")
    actual = {name: metadata.version(name) for name in VERSIONS}
    if actual != VERSIONS:
        raise ValueError("qualified author framework versions required")
    distribution = metadata.distribution("libwebarena")
    packaged = json.loads(
        distribution.locate_file("webarena/test.raw.json").read_text("utf-8")
    )
    if (
        not isinstance(packaged, list)
        or args.task_id >= len(packaged)
        or packaged[args.task_id] != task
    ):
        raise ValueError("packaged task differs from the complete pinned original task")
    with ExitStack() as stack:
        endpoint = args.endpoint
        if args.queue is not None:
            port = stack.enter_context(
                model_proxy(args.queue, port=0, max_request_bytes=64 * 1024 * 1024)
            )
            endpoint = f"http://127.0.0.1:{port}/v1"
        os.environ.update(environment_bindings(sites))
        os.environ["VLLM_API_URL"] = endpoint
        os.environ["OPENAI_API_KEY"] = (
            "local-loopback"  # pragma: allowlist secret - local placeholder
        )
        import browsergym.webarena  # noqa: F401
        import gymnasium
        from agentlab.agents.generic_agent.agent_configs import FLAGS_CUSTOM
        from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
        from agentlab.llm.chat_api import SelfHostedModelArgs
        from browsergym.experiments.loop import EnvArgs, ExpArgs

        name = f"webarena.{args.task_id}"
        gym_spec = gymnasium.spec("browsergym/" + name)
        stack.enter_context(browser_timeout_profile(gym_spec, timeout_ms))
        model_options = {"model_name": args.model, "backend": "vllm"}
        if args.max_output_tokens is not None:
            model_options["max_new_tokens"] = args.max_output_tokens
        run = ExpArgs(
            agent_args=GenericAgentArgs(
                chat_model_args=SelfHostedModelArgs(**model_options),
                flags=deepcopy(FLAGS_CUSTOM),
            ),
            env_args=EnvArgs(
                task_name=name, task_seed=args.seed, max_steps=args.max_steps
            ),
        )
        output.mkdir(parents=True, exist_ok=False)
        run.prepare(output)
        run.run()
        if run.err_msg:
            raise RuntimeError(run.err_msg)
        return {
            "status": "NATIVE_TASK_FINISHED",
            "source_id": "WEBARENA",
            "task_id": args.task_id,
            "result_dir": str(run.exp_dir),
            "model": args.model,
            "max_steps": args.max_steps,
            "task_seed": args.seed,
            "max_output_tokens": args.max_output_tokens,
            "browser_timeout_ms": timeout_ms,
            "unconfigured_sites": unavailable,
            "task_and_scoring_code_changed": False,
            "framework_versions": actual,
            "scoring_provider": "Unmodified libwebarena0.0.4 via BrowserGym-WebArena0.14.2",
            "source_checkbox_eligible": False,
            **terminal_summary(run.exp_dir),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--author-root", type=Path)
    parser.add_argument(
        "--sites",
        type=Path,
        help="JSON map of uppercase site names to owned loopback URLs",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model")
    transport = parser.add_mutually_exclusive_group()
    transport.add_argument("--endpoint", type=loopback_url)
    transport.add_argument("--queue", type=Path)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument(
        "--browser-timeout-ms",
        type=int,
        help="Explicit BrowserEnv page/action wait profile; omitted preserves the author default. Does not change step or token budgets.",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "PLAN_ONLY",
                    "source_id": "WEBARENA",
                    "task_id": args.task_id,
                    "source_checkbox_eligible": False,
                }
            )
        )
        return
    print(json.dumps(execute(args)))


if __name__ == "__main__":
    main()
