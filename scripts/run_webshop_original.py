"""Run fixed original WebShop text sessions with a local chat model."""

import argparse
import hashlib
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


def author_episode_score(result):
    if result["terminated"]:
        return result["reward"]
    if len(result["steps"]) != 100:
        raise ValueError("cannot score an incomplete author episode")
    return 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--author-root", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--endpoint")
    p.add_argument("--model")
    p.add_argument("--verify-saved-author-budget", type=Path)
    p.add_argument("--qwen-no-thinking", action="store_true")
    p.add_argument("--retain-history", action="store_true")
    p.add_argument("--author-episode-budget", action="store_true")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if args.verify_saved_author_budget:
        if any(
            (
                args.author_root,
                args.output,
                args.endpoint,
                args.model,
                args.execute,
                args.qwen_no_thinking,
                args.retain_history,
                args.author_episode_budget,
            )
        ):
            p.error("saved verification cannot be combined with run options")
        raw = args.verify_saved_author_budget.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if (
            digest != "15b8bc1ec3714a97050e4b9ac2690f2674db9c4ccdd8648cbff6014d017876d3"
        ):  # pragma: allowlist secret - saved result SHA256
            p.error("saved output does not match the pinned author-budget run")
        rows = [json.loads(line) for line in raw.splitlines()]
        if [row["session"] for row in rows] != [0, 1, 2] or any(
            row["terminated"]
            or row["reward"] is not None
            or any(step.get("invalid") or step.get("done") for step in row["steps"])
            or author_episode_score(row) != row["episode_score"]
            for row in rows
        ):
            p.error("invalid saved author-budget episode outcome")
        print(
            json.dumps(
                {
                    "status": "SAVED_EPISODES_VERIFIED",
                    "sha256": digest,
                    "episode_scores": [row["episode_score"] for row in rows],
                }
            )
        )
        return
    if not all((args.author_root, args.output, args.endpoint, args.model)):
        p.error(
            "--author-root, --output, --endpoint and --model are required for a run"
        )
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
        base_url=args.endpoint,
        api_key="local-loopback",  # pragma: allowlist secret - noncredential for validated loopback
        max_retries=0,
        timeout=60,
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
                result["episode_score"] = author_episode_score(result)
            output.write(json.dumps(result) + "\n")
            output.flush()


if __name__ == "__main__":
    main()
