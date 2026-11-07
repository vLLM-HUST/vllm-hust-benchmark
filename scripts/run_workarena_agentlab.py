"""Plan or execute the fixed original WorkArena L1 slice on a dedicated instance."""

import argparse
from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

from plan_workarena_l1_sample import original_l1_profile
from run_miniwob_agentlab import loopback_url, terminal_summary


def require_dedicated_instance(env):
    names = ("SNOW_INSTANCE_URL", "SNOW_INSTANCE_UNAME", "SNOW_INSTANCE_PWD")
    if not all(env.get(name, "").strip() for name in names):
        raise ValueError(
            "explicit dedicated instance environment required; no pool fallback"
        )
    url = urlsplit(env[names[0]])
    if (
        url.scheme != "https"
        or not re.fullmatch(r"[a-z0-9-]+\.service-now\.com", url.netloc)
        or url.path not in {"", "/"}
        or url.query
        or url.fragment
    ):
        raise ValueError("expected a dedicated HTTPS ServiceNow instance origin")


def verify_selected_sample(inventory, selected):
    data = json.loads(Path(inventory).read_text("utf-8"))
    expected = data["delivery_inventory"]["WORKARENA"][
        "fixed_l1_sample_preflight_20261107"
    ]["selected"]
    observed = [
        {key: row[key] for key in ("task_id", "seed", "category")} for row in selected
    ]
    if observed != expected:
        raise ValueError(
            "original task or seed differs from the fixed delivery inventory"
        )


def build_experiments(model):
    if importlib.metadata.version("agentlab") != "0.4.2":
        raise ValueError("requires the qualified AgentLab 0.4.2 client")
    from agentlab.agents.generic_agent.agent_configs import FLAGS_CUSTOM
    from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
    from agentlab.llm.chat_api import SelfHostedModelArgs
    from browsergym.experiments.loop import ExpArgs

    _, _, selected, benchmark, env_args, action_profile = original_l1_profile()
    runs = []
    for args in env_args:
        agent = GenericAgentArgs(
            chat_model_args=SelfHostedModelArgs(model_name=model, backend="vllm"),
            flags=deepcopy(FLAGS_CUSTOM),
        )
        agent.set_benchmark(benchmark, demo_mode=False)
        if any(
            getattr(agent.flags.action.action_set, key) != value
            for key, value in action_profile.items()
        ):
            raise ValueError("AgentLab altered original WorkArena action profile")
        runs.append(ExpArgs(agent_args=agent, env_args=args))
    return selected, runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", type=loopback_url)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inventory", type=Path, help="unique delivery inventory JSON")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-dedicated-instance", action="store_true")
    args = parser.parse_args()
    if not args.model.strip():
        parser.error("actual nonempty model ID required")
    if args.execute:
        if (
            not args.confirm_dedicated_instance
            or not args.endpoint
            or not args.endpoint.rstrip("/").endswith("/v1")
            or args.output is None
            or args.output.exists()
            or args.output.is_symlink()
            or args.inventory is None
        ):
            parser.error(
                "dedicated-instance confirmation, inventory, explicit loopback /v1 and fresh output required"
            )
        require_dedicated_instance(os.environ)
    elif args.endpoint or args.output or args.confirm_dedicated_instance:
        parser.error("runtime arguments require --execute")
    selected, runs = build_experiments(args.model)
    if args.inventory is not None:
        verify_selected_sample(args.inventory, selected)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "ORIGINAL_WORKARENA_AGENTLAB_CONFIG_BOUND_NOT_EXECUTED",
                    "selected": selected,
                    "original_max_steps": [run.env_args.max_steps for run in runs],
                    "model_or_site_calls": 0,
                    "source_checkbox_eligible": False,
                },
                indent=2,
            )
        )
        return
    os.environ["VLLM_API_URL"] = args.endpoint
    os.environ["OPENAI_API_KEY"] = (
        "local-loopback"  # pragma: allowlist secret - local client placeholder
    )
    args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    for row, run in zip(selected, runs, strict=True):
        run.prepare(args.output)
        run.run()
        if run.err_msg:
            raise RuntimeError(
                "WorkArena run failed; inspect private experiment output"
            )
        print(json.dumps({**row, **terminal_summary(run.exp_dir)}), flush=True)


if __name__ == "__main__":
    main()
