"""Run fixed original MiniWoB++ tasks with the AgentLab vLLM client."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
from urllib.parse import urlsplit


TASKS = ("miniwob.click-dialog", "miniwob.multi-layouts")
SAVED_PROFILES = {
    TASKS[0]: (
        "d4f9370ad801820cea1f1855d8476747d7769703d9a540a8c4d5ba537724f8ea",  # pragma: allowlist secret - saved summary SHA256
        1024,
        (1, 1.0, True, False),
    ),
    TASKS[1]: (
        "e2722e4ffc6bb1c0ca504010bdb95be5b607b98d58d039154544556085e58a63",  # pragma: allowlist secret - saved summary SHA256
        4096,
        (20, 0.0, False, True),
    ),
}


@contextmanager
def response_metadata(client_type, destination):
    original = client_type.create

    def capture(client, *args, **kwargs):
        result = original(client, *args, **kwargs)
        image_parts = sum(
            part.get("type") == "image_url"
            for message in kwargs.get("messages", [])
            if isinstance(message, dict) and isinstance(message.get("content"), list)
            for part in message["content"]
            if isinstance(part, dict)
        )
        for choice in result.choices:
            row = {
                "finish_reason": choice.finish_reason,
                "content_is_null": choice.message.content is None,
                "completion_tokens": getattr(result.usage, "completion_tokens", None),
                "request_image_parts": image_parts,
            }
            with destination.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
        return result

    client_type.create = capture
    try:
        yield
    finally:
        client_type.create = original


def terminal_summary(result_dir):
    summary = json.loads((Path(result_dir) / "summary_info.json").read_text("utf-8"))
    reward = summary.get("cum_reward")
    steps = summary.get("n_steps")
    terminated = summary.get("terminated")
    truncated = summary.get("truncated")
    if (
        summary.get("err_msg") is not None
        or type(steps) is not int
        or steps < 1
        or type(reward) not in (int, float)
        or not math.isfinite(reward)
        or type(terminated) is not bool
        or type(truncated) is not bool
        or not (terminated or truncated)
    ):
        raise ValueError("missing or invalid BrowserGym terminal summary")
    return {
        "steps": steps,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
    }


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
    parser.add_argument("--site", type=loopback_url)
    parser.add_argument("--endpoint", type=loopback_url)
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--agent-profile",
        choices=("axtree", "visual"),
        default="axtree",
        help="visual uses the unmodified author VisualAgent coordinate/screenshot defaults",
    )
    parser.add_argument("--record-response-metadata", action="store_true")
    parser.add_argument("--verify-saved", type=Path, metavar="RESULT_DIR")
    args = parser.parse_args()
    runtime_options = (
        args.site,
        args.endpoint,
        args.model,
        args.output,
        args.max_steps,
        args.max_output_tokens,
    )
    if args.verify_saved is not None:
        if (
            args.execute
            or args.agent_profile != "axtree"
            or args.record_response_metadata
            or any(v is not None for v in runtime_options)
        ):
            parser.error("--verify-saved cannot be combined with runtime options")
        expected_hash, budget, expected_terminal = SAVED_PROFILES[args.task]
        actual_hash = hashlib.sha256(
            (args.verify_saved / "summary_info.json").read_bytes()
        ).hexdigest()
        if actual_hash != expected_hash:
            parser.error("saved summary does not match the pinned original evidence")
        summary = terminal_summary(args.verify_saved)
        if tuple(summary.values()) != expected_terminal:
            parser.error("saved terminal outcome differs from the pinned receipt")
        print(
            json.dumps(
                {
                    "status": "SAVED_SUMMARY_VERIFIED",
                    "task": args.task,
                    "sha256": actual_hash,
                    "max_output_tokens": budget,
                    **summary,
                }
            )
        )
        return
    if (
        any(v is None for v in runtime_options)
        or not args.site.endswith("/miniwob/")
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
    os.environ["OPENAI_API_KEY"] = (
        "local-loopback"  # pragma: allowlist secret - unauthenticated loopback placeholder
    )
    import browsergym.miniwob  # noqa: F401
    from agentlab.agents.generic_agent.agent_configs import FLAGS_CUSTOM
    from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
    from agentlab.llm.chat_api import SelfHostedModelArgs
    from browsergym.experiments.loop import EnvArgs, ExpArgs

    flags = FLAGS_CUSTOM
    agent_type = GenericAgentArgs
    expected_actions = ["bid"]
    if args.agent_profile == "visual":
        from agentlab.agents.visual_agent.agent_configs import DEFAULT_PROMPT_FLAGS
        from agentlab.agents.visual_agent.visual_agent import VisualAgentArgs

        flags = DEFAULT_PROMPT_FLAGS
        agent_type = VisualAgentArgs
        expected_actions = ["coord"]
        if not flags.obs.use_screenshot or flags.obs.use_som:
            raise RuntimeError("author screenshot-only visual profile changed")
    action_set = flags.action.action_set
    if list(action_set.subsets) != expected_actions or action_set.multiaction:
        raise RuntimeError("MiniWoB++ requires the fixed single browser-action profile")

    args.output.mkdir(parents=True, exist_ok=False)
    run = ExpArgs(
        agent_args=agent_type(
            chat_model_args=SelfHostedModelArgs(
                model_name=args.model,
                backend="vllm",
                max_new_tokens=args.max_output_tokens,
                vision_support=args.agent_profile == "visual",
            ),
            flags=flags,
        ),
        env_args=EnvArgs(task_name=args.task, task_seed=42, max_steps=args.max_steps),
    )
    run.prepare(args.output)
    if args.record_response_metadata:
        from openai.resources.chat.completions import Completions

        with response_metadata(Completions, args.output / "response-metadata.jsonl"):
            run.run()
    else:
        run.run()
    if run.err_msg:
        raise RuntimeError(run.err_msg)
    print(
        json.dumps(
            {
                "status": "FINISHED",
                "result_dir": str(run.exp_dir),
                "task": args.task,
                "seed": 42,
                "model": args.model,
                "agent_profile": args.agent_profile,
                "max_steps": args.max_steps,
                "max_output_tokens": args.max_output_tokens,
                **terminal_summary(run.exp_dir),
            }
        )
    )


if __name__ == "__main__":
    main()
