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
    if not args.execute:
        print(
            json.dumps({"status": "PLAN_ONLY", "sessions": [0, 1, 2], "max_steps": 15})
        )
        return
    from openai import OpenAI

    sys.path.insert(0, str(args.author_root.resolve(strict=True)))
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

    client = OpenAI(
        base_url=args.endpoint, api_key="local-loopback", max_retries=0, timeout=60
    )
    env = WebAgentTextEnv(observation_mode="text", num_products=1000, human_goals=True)

    def choose(observation, available):
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {
                    "role": "user",
                    "content": f"{observation}\nAvailable actions: {available}\nReturn one action only: search[query] or click[element].",
                }
            ],
            max_tokens=1024,
            temperature=0,
        )
        return response.choices[0].message.content

    with args.output.open("x", encoding="utf-8") as output:
        for session in (0, 1, 2):
            output.write(json.dumps(run_session(env, session, choose, 15)) + "\n")
            output.flush()


if __name__ == "__main__":
    main()
