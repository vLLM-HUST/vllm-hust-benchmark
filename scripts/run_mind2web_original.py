"""Run the pinned author evaluator on externally prepared complete original tasks.

Caller owns model access and exclusive resource coordination. Default is inert.
This adapter does not download data, change tasks or replace the author scorer.
"""

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
from urllib.parse import urlsplit

from prepare_jsonschemabench_author_source import verify_source

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("test_task", "test_website", "test_domain")


class RestrictedScores(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("collections", "defaultdict"): collections.defaultdict,
            ("builtins", "dict"): dict,
            ("builtins", "float"): float,
            ("builtins", "int"): int,
        }
        if (module, name) not in allowed:
            raise ValueError("unreviewed candidate pickle global")
        return allowed[module, name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("author-root", "tasks-root", "tokenizer", "scores", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    endpoint = urlsplit(args.endpoint)
    if (
        endpoint.scheme not in {"http", "https"}
        or not endpoint.hostname
        or endpoint.username
        or endpoint.password
        or endpoint.query
        or endpoint.fragment
    ):
        parser.error("HTTP model endpoint without embedded credentials required")
    if args.output.resolve().is_relative_to(ROOT):
        parser.error("raw native outputs must remain outside the checkout")
    if not args.execute:
        print(
            json.dumps(
                {
                    "executed": False,
                    "splits": SPLITS,
                    "max_new_tokens": 50,
                    "top_k": 50,
                    "disable_thinking": args.disable_thinking,
                }
            )
        )
        return
    author = verify_source(
        args.author_root, ROOT / "configs/workload-native/mind2web-author-source.json"
    )
    protocol = json.loads(
        (ROOT / "configs/workload-native/mind2web-original-protocol.json").read_text(
            "utf-8"
        )
    )
    expected = next(
        item
        for item in protocol["official_assets"]
        if item["path"] == "scores_all_data.pkl"
    )
    with args.scores.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected["sha256"]:
            raise ValueError("original publisher candidate identity mismatch")
        stream.seek(0)
        scores = RestrictedScores(stream).load()
    sys.path.insert(0, str(author / "src/action_prediction"))
    from dataloader import get_data_split, MultiChoiceDataset
    from metric import ActionEvaluatorMultiChoice
    from transformers import AutoTokenizer
    import openai

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True
    )
    openai.api_base = args.endpoint
    openai.api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(123)

    class ServedModel:
        calls = 0

        def generate(self, prompt, max_new_tokens=50):
            if max_new_tokens != 50:
                raise ValueError("original generation budget changed")
            parameters = (
                {"chat_template_kwargs": {"enable_thinking": False}}
                if args.disable_thinking
                else {}
            )
            response = openai.ChatCompletion.create(
                model=args.model,
                messages=prompt,
                max_tokens=50,
                temperature=0,
                request_timeout=1800,
                **parameters,
            )
            self.calls += 1
            with (args.output / f"response-{self.calls:05d}.private.json").open(
                "x"
            ) as stream:
                json.dump(response, stream)
            return [choice["message"]["content"] for choice in response["choices"]]

    model = ServedModel()
    for split in SPLITS:
        name = f"{split}-original-first3.private.json"
        tasks = json.loads((args.tasks_root / name).read_text("utf-8"))
        if (
            not isinstance(tasks, list)
            or len(tasks) != 3
            or any(not task.get("actions") for task in tasks)
        ):
            raise ValueError("three complete original tasks per split required")
        data = get_data_split(str(args.tasks_root), name, candidate_results=scores)
        dataset = MultiChoiceDataset(
            data, tokenizer, neg_ratio=0.2, num_candidates=5, max_context_len=512
        )
        prompt = json.loads(
            (author / "src/action_prediction/llm_prompt.json").read_text("utf-8")
        )
        result = ActionEvaluatorMultiChoice(tokenizer).evaluate_dataset_llm(
            dataset,
            model,
            output_path=str(args.output),
            name=split,
            prompt_template=prompt,
            top_k=50,
        )
        print(json.dumps({"split": split, "official_result": result}), flush=True)


if __name__ == "__main__":
    main()
