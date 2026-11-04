"""Run fixed original MiniWoB++ tasks with the AgentLab vLLM client."""

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlsplit


TASKS = ("miniwob.click-dialog", "miniwob.multi-layouts")


def loopback_url(value):
    url = urlsplit(value)
    if (
        url.scheme != "http"
        or url.hostname != "127.0.0.1"
        or not url.port
        or url.username is not None
        or url.query
        or url.fragment
    ):
        raise argparse.ArgumentTypeError("explicit IPv4 loopback HTTP URL required")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--site", type=loopback_url, required=True)
    parser.add_argument("--endpoint", type=loopback_url, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--max-output-tokens", type=int, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if (
        not args.site.endswith("/miniwob/")
        or not args.endpoint.rstrip("/").endswith("/v1")
        or not args.model.strip()
        or min(args.max_steps, args.max_output_tokens) <= 0
    ):
        parser.error("valid site, vLLM endpoint, model and positive budgets required")
    if not args.execute:
        print(json.dumps({"status": "PLAN_ONLY", "task": args.task, "seed": 42}))
        return

    os.environ["MINIWOB_URL"] = args.site
    os.environ["VLLM_API_URL"] = args.endpoint
    import browsergym.miniwob  # noqa: F401
    from agentlab.agents.generic_agent.agent_configs import FLAGS_CUSTOM
    from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
    from agentlab.llm.chat_api import SelfHostedModelArgs
    from browsergym.experiments.loop import EnvArgs, ExpArgs

    args.output.mkdir(parents=True, exist_ok=False)
    run = ExpArgs(
        agent_args=GenericAgentArgs(
            chat_model_args=SelfHostedModelArgs(
                model_name=args.model,
                backend="vllm",
                max_new_tokens=args.max_output_tokens,
            ),
            flags=FLAGS_CUSTOM,
        ),
        env_args=EnvArgs(task_name=args.task, task_seed=42, max_steps=args.max_steps),
    )
    run.prepare(args.output)
    run.run()
    if run.err_msg:
        raise RuntimeError(run.err_msg)
    print(json.dumps({"status": "FINISHED", "result_dir": str(run.exp_dir)}))


if __name__ == "__main__":
    main()
