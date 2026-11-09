"""Run three original ResearchCodeBench annotations through its author harness."""

import argparse
import asyncio
import json
from pathlib import Path
import sys
from types import ModuleType
from urllib.request import Request, urlopen

from appworld_model_transport import model_proxy

MODEL = "Qwen/Qwen3.5-9B"
COHORT = ["Diff-Transformer", "GPS", "LEN"]


class QueueClient:
    """Implement the provider protocol consumed by the original prompt builder."""

    name = "QWEN3_5_9B_LOCAL"

    def __init__(self, port):
        self.url = f"http://127.0.0.1:{port}/v1/chat/completions"

    async def run(self, *, llm_type, user_message, num_completions, temperature):
        if llm_type is not self or num_completions != 1:
            raise ValueError("the fixed local profile requires one completion")
        return await asyncio.to_thread(self.request, user_message, temperature)

    def request(self, prompt, temperature):
        body = json.dumps(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4096,
                "temperature": temperature,
            }
        ).encode()
        req = Request(self.url, data=body, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=900) as response:
            reply = json.load(response)
        if reply["model"] != MODEL:
            raise ValueError("unexpected model identity")
        return [reply["choices"][0]["message"]["content"]]


def cohort(pset):
    return [
        (p.folder_name, f.rel_path, s.name, s.start_line)
        for p in pset.problems
        for f in p.problem_files
        for s in f.snippets
    ]


def select(author, pset_class):
    pset = pset_class.parse_pset(str(author / "pset"), selected_problems=COHORT)
    if [p.folder_name for p in pset.problems] != COHORT:
        raise ValueError("pinned paper order changed")
    for problem in pset.problems:
        if len(problem.problem_files) != 1 or not problem.problem_files[0].snippets:
            raise ValueError("expected one annotated file per paper")
        problem.problem_files[0].snippets = problem.problem_files[0].snippets[:1]
    return pset


def main(args):
    # Only the provider adapter is replaced. Original prompt and scorer are loaded unchanged.
    provider = ModuleType("core.async_chat_clients")
    provider.AsyncChatClients = QueueClient
    sys.modules[provider.__name__] = provider
    sys.path.insert(0, str(args.author))
    from core.annotation.models.pset import PSet

    fixed = select(args.author, PSet)
    if args.phase == "plan":
        print(json.dumps({"cohort": cohort(fixed), "model_calls": 0}))
        return
    raw = args.output / "predictions.private.json"
    if args.phase == "generate":
        if args.queue is None:
            raise ValueError("an independent private queue is required")
        with model_proxy(args.queue, port=0, timeout=900, expected_model=MODEL) as port:
            client = QueueClient(port)
            asyncio.run(fixed.solve_sequentially([client], 1, 0.0, client))
        with raw.open("x") as stream:
            stream.write(fixed.model_dump_json())
        return
    generated = PSet.model_validate_json(raw.read_text())
    if cohort(generated) != cohort(fixed):
        raise ValueError("saved predictions differ from fixed sample")
    generated.test_all(str(args.author / "pset"), "/tmp/scoring", timeout_seconds=60)
    records = []
    for problem in generated.problems:
        (snippet,) = problem.problem_files[0].snippets
        (completion,) = snippet.predictions[QueueClient.name].completions
        result = completion.test_result
        if result is None:
            raise ValueError("missing author score")
        records.append(
            {
                "paper": problem.folder_name,
                "snippet": snippet.name,
                "original_exit_code": result.exit_code,
                "original_passed": result.passed,
            }
        )
    with (args.output / "author-scored.private.json").open("x") as stream:
        stream.write(generated.model_dump_json())
    with (args.output / "original-score-public.json").open("x") as stream:
        json.dump({"model": MODEL, "tasks": records}, stream, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "generate", "score"))
    parser.add_argument("--author", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queue", type=Path)
    main(parser.parse_args())
