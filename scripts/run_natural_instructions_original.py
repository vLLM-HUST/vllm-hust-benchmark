"""Fixed original NI tasks with upstream collator/scorer; no observation or analysis.

Run standalone in the documented Python 3.9 / Transformers 4.18 environment.
Data, upstream sources and all generated outputs remain outside this repository.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlparse
from urllib.request import Request, urlopen

REVISION = "55a365637381ce7f3748fa2eac7aef1a113bbb82"  # pragma: allowlist secret - public revision
COLLATOR_SHA = "d170d2230bba4a1f250ee5f3ecf16131f0e334855602115d95e4d16befdb7a29"  # pragma: allowlist secret - upstream code checksum
TASKS = {
    "task1575_amazon_reviews_multi_sentiment_classification.json": (
        "bbea80b443d94d18b53d016e18f01c50928488f9c9de30964c59880db5ffd6b0",  # pragma: allowlist secret - public data checksum
        102,
    ),  # pragma: allowlist secret - public data checksum
    "task778_pawsx_english_french_translation.json": (
        "7291f56113deffa27fbfc149167293249287c9a36b80d66b8bbecc815b97b50a",  # pragma: allowlist secret - public data checksum
        250,
    ),  # pragma: allowlist secret - public data checksum
    "task1290_xsum_summarization.json": (
        "932e33f4ed3e23587076d938f9e088ae6c0a1b09a88468b493286c085cfeca41",  # pragma: allowlist secret - public data checksum
        6493,
    ),  # pragma: allowlist secret - public data checksum
}
SCORER_FILES = {
    "evaluation.py": "aac89279abb16c8e9acfaac90ad612f51d5ca27ae75259cb9524276b5dd7dbb3",  # pragma: allowlist secret - original scorer checksum
    "rouge/__init__.py": "6d23d061b8eee523e6f1f2e7a7a4fa667751cef3f54c63cae65d54ed53af4cfe",  # pragma: allowlist secret - original scorer checksum
    "rouge/rouge_scorer.py": "d466ebff966664ccd77589635042634f6c152320a95b807978db63dbb5423708",  # pragma: allowlist secret - original scorer checksum
    "rouge/scoring.py": "7ed8af87ee7e402eaf8fe3386f19e11f9fe316a00ff741afa439ed2f1cbb0c91",  # pragma: allowlist secret - original scorer checksum
    "rouge/tokenize.py": "65e5173bc32c5a99a4266a10c49300f500ef35e1e68317a70c272904a567e42a",  # pragma: allowlist secret - original scorer checksum
    "rouge/tokenizers.py": "5e8a2f10e5bf3d8c35ef05237ec9a5a79441ab88445a937804017696ed6f3d54",  # pragma: allowlist secret - original scorer checksum
    "gpt2/vocab.json": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",  # pragma: allowlist secret - OpenAI GPT2 asset checksum
    "gpt2/merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",  # pragma: allowlist secret - OpenAI GPT2 asset checksum
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_endpoint(value):
    parsed = urlparse(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1"}
        or parsed.path != "/v1"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("only a local /v1 endpoint without credentials is supported")
    return value


def selected_indices(name, full_small):
    count = TASKS[name][1]
    if full_small and count > 250:
        raise ValueError(
            "full-small requires explicitly selected tasks of at most 250 rows"
        )
    return range(count if full_small else 3)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def post(url, body):
    request = Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=300) as response:
        return json.load(response)


class OriginalModelTokenizer:
    """Only actual token IDs from the serving model; never truncate a task."""

    def __init__(self, endpoint, model):
        self.url, self.model = endpoint.removesuffix("/v1") + "/tokenize", model

    def __call__(self, text):
        result = post(
            self.url, {"model": self.model, "prompt": text, "add_special_tokens": False}
        )
        if not isinstance(result.get("tokens"), list) or not result["tokens"]:
            raise ValueError("original model tokenizer returned no real token IDs")
        return {"input_ids": result["tokens"]}

    def decode(self, *args, **kwargs):
        raise ValueError("original task would require truncation; no fallback")


def original_scorer(root):
    root = root.resolve()
    for relative, expected in SCORER_FILES.items():
        if digest(root / relative) != expected:
            raise ValueError("original scoring source or GPT2 asset identity mismatch")
    config = json.loads(
        (root / "gpt2/tokenizer_config.json").read_text(encoding="utf-8")
    )
    if config != {"tokenizer_class": "GPT2Tokenizer", "model_max_length": 100000}:
        raise ValueError("use the documented local GPT2 class-selection configuration")
    previous = Path.cwd()
    sys.path.insert(0, str(root))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        os.chdir(root)
        module = load_module("ni_fixed_original_evaluation", root / "evaluation.py")
        if (
            Path(module.rouge_scorer.__file__).resolve()
            != (root / "rouge/rouge_scorer.py").resolve()
        ):
            raise ValueError(
                "the imported ROUGE module is not the pinned original source"
            )
        return module
    finally:
        os.chdir(previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--collator-source", required=True, type=Path)
    parser.add_argument("--scorer-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--task-files", nargs="+", choices=list(TASKS), default=list(TASKS)
    )
    parser.add_argument("--full-small", action="store_true")
    parser.add_argument("--endpoint", default="http://127.0.0.1:19466/v1")
    parser.add_argument("--model", default="workload-qwen35-27b-128k-nonthinking")
    args = parser.parse_args()
    endpoint = validate_endpoint(args.endpoint.rstrip("/"))
    if len(set(args.task_files)) != len(args.task_files):
        parser.error("select distinct original task files")
    if digest(args.collator_source) != COLLATOR_SHA:
        raise ValueError("original collator identity mismatch")
    import transformers

    if transformers.__version__ != "4.18.0":
        raise ValueError("use the upstream Transformers 4.18.0 collator environment")
    upstream = load_module("ni_fixed_original_collator", args.collator_source)
    collator = upstream.DataCollatorForNI(
        tokenizer=OriginalModelTokenizer(endpoint, args.model),
        text_only=True,
        max_source_length=128000,
        add_task_name=False,
        add_task_definition=True,
        num_pos_examples=0,
        num_neg_examples=0,
        tk_instruct=False,
    )
    scorer = original_scorer(args.scorer_root) if args.scorer_root else None
    tasks = {}
    for name in args.task_files:
        if digest(args.data_root / name) != TASKS[name][0]:
            raise ValueError("original task identity mismatch")
        task = json.loads((args.data_root / name).read_text(encoding="utf-8"))
        if len(task["Instances"]) != TASKS[name][1]:
            raise ValueError("original instance count mismatch")
        if len({row["id"] for row in task["Instances"]}) != TASKS[name][1]:
            raise ValueError("original instance identifiers are not distinct")
        selected_indices(name, args.full_small)
        tasks[name] = task
    output = args.output.resolve()
    if output == args.data_root.resolve() or output.is_relative_to(
        args.data_root.resolve()
    ):
        raise ValueError("output must be outside the original data directory")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    receipt = {
        "source_id": "NATURAL_INSTRUCTIONS",
        "revision": REVISION,
        "helper_sha256": digest(__file__),
        "model": args.model,
        "temperature": 0,
        "max_tokens": 1024,
        "selection": "full_selected_small_task_files"
        if args.full_small
        else "first_three_per_task",
        "status": "RUNNING",
        "samples": [],
        "official_scores": [],
        "performance_claim": False,
        "underlying_dataset_equivalence": False,
    }

    def save():
        path = output / "receipt.json"
        path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        path.chmod(0o600)

    save()
    for name, task in tasks.items():
        predictions, references = [], []
        for index in selected_indices(name, args.full_small):
            instance = task["Instances"][index]
            row = {
                **{k: v for k, v in task.items() if k != "Instances"},
                "Task": name.removesuffix(".json"),
                "Instance": instance,
            }
            prompt = collator([row])["inputs"][0]
            messages = [{"role": "user", "content": prompt}]
            count = post(
                endpoint.removesuffix("/v1") + "/tokenize",
                {
                    "model": args.model,
                    "messages": messages,
                    "add_generation_prompt": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )["count"]
            if count + 1024 > 131072:
                raise ValueError("original full input exceeds declared serving budget")
            payload = {
                "model": args.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 1024,
                "n": 1,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            result = post(endpoint + "/chat/completions", payload)
            choice = result["choices"][0]
            if (
                choice["finish_reason"] not in {"stop", "length"}
                or not isinstance(choice["message"].get("content"), str)
                or not choice["message"]["content"]
                or result["usage"]["prompt_tokens"] != count
            ):
                raise ValueError("response integrity validation failed")
            private = output / (
                hashlib.sha256((name + ":" + instance["id"]).encode()).hexdigest()
                + ".private.json"
            )
            private.write_text(
                json.dumps(
                    {"request": payload, "response": result}, ensure_ascii=False
                ),
                encoding="utf-8",
            )
            private.chmod(0o600)
            receipt["samples"].append(
                {
                    "task_file": name,
                    "instance_id": instance["id"],
                    "index": index,
                    "finish_reason": choice["finish_reason"],
                    "usage": result["usage"],
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "private_result_sha256": digest(private),
                }
            )
            predictions.append(choice["message"]["content"])
            references.append(instance["output"])
            save()
        if scorer:
            receipt["official_scores"].append(
                {
                    "task_file": name,
                    "instances": len(predictions),
                    "xlingual": "translation" in name,
                    "metrics": scorer.compute_metrics(
                        predictions, references, xlingual="translation" in name
                    ),
                }
            )
        if digest(args.data_root / name) != TASKS[name][0]:
            raise ValueError("original task changed during execution")
        save()
    receipt["status"] = (
        "ORIGINAL_SELECTED_TASKS_EXECUTED_AND_SCORED"
        if scorer
        else "ORIGINAL_SELECTED_TASKS_GENERATED_SCORING_NOT_RUN"
    )
    receipt["all_normal_stop"] = all(
        s["finish_reason"] == "stop" for s in receipt["samples"]
    )
    save()
    print(json.dumps({"status": receipt["status"], "samples": len(receipt["samples"])}))


if __name__ == "__main__":
    main()
