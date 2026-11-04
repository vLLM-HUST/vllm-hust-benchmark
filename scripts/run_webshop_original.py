"""Run fixed original WebShop text sessions with a local chat model."""

import argparse
import json
import math
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit


def run_session(env, session, choose, max_steps):
    observation, _ = env.reset(session=session)
    steps = []
    for _ in range(max_steps):
        available = env.get_available_actions()
        action = choose(observation, available)
        match = re.fullmatch(r"(search|click)\[([^\[\]\r\n]+)\]", action or "")
        if (
            not match
            or (match[1] == "search" and not available["has_search_bar"])
            or (match[1] == "click" and match[2] not in available["clickables"])
        ):
            steps.append({"action": action, "invalid": True})
            continue
        observation, reward, done, _ = env.step(action)
        if not math.isfinite(reward):
            raise ValueError("nonfinite author reward")
        steps.append({"action": action, "reward": reward, "done": done})
        if done:
            return {
                "session": session,
                "steps": steps,
                "terminated": True,
                "reward": reward,
            }
    return {"session": session, "steps": steps, "terminated": False, "reward": None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--author-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--qwen-no-thinking", action="store_true")
    p.add_argument("--retain-history", action="store_true")
    p.add_argument("--author-episode-budget", action="store_true")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    url = urlsplit(args.endpoint)
    if (
        (url.scheme, url.hostname, url.path) != ("http", "127.0.0.1", "/v1")
        or not url.port
        or url.username
        or url.query
        or url.fragment
    ):
        p.error("explicit loopback /v1 endpoint required")
    max_steps = 100 if args.author_episode_budget else 15
    if not args.execute:
        print(
            json.dumps(
                {"status": "PLAN_ONLY", "sessions": [0, 1, 2], "max_steps": max_steps}
            )
        )
        return
    from openai import OpenAI

    sys.path.insert(0, str(args.author_root.resolve(strict=True)))
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

    client = OpenAI(
        base_url=args.endpoint, api_key="local-loopback", max_retries=0, timeout=60
    )
    env = WebAgentTextEnv(observation_mode="text", num_products=1000, human_goals=True)
    history = []

    def choose(observation, available):
        template_options = (
            {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
            if args.qwen_no_thinking
            else {}
        )
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                *history,
                {
                    "role": "user",
                    "content": f"{observation}\nAvailable actions: {available}\nReturn one action only: search[query] or click[element].",
                },
            ],
            max_tokens=1024,
            temperature=0,
            **template_options,
        )
        action = response.choices[0].message.content
        if args.retain_history and action is not None:
            history.extend(
                [
                    {
                        "role": "user",
                        "content": f"{observation}\nAvailable actions: {available}",
                    },
                    {"role": "assistant", "content": action},
                ]
            )
        return action

    with args.output.open("x", encoding="utf-8") as output:
        for session in (0, 1, 2):
            history.clear()
            result = run_session(env, session, choose, max_steps)
            if args.author_episode_budget:
                # Publisher baseline returns zero if the 100-action episode expires.
                result["episode_score"] = (
                    result["reward"] if result["terminated"] else 0.0
                )
            output.write(json.dumps(result) + "\n")
            output.flush()


if __name__ == "__main__":
    main()
