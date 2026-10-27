"""Run selected original BBH tasks with publisher prompts and a pinned reference harness.

Use ``workload-data prepare-bbh-original`` first. All 27 original task files are
supported. By default, each selected task uses its first three original rows;
``--full`` uses all rows. Select only gaps when reusing existing validation evidence.
This is a model-native chat transport plus pinned lm-eval scoring, not the publisher's
original model/scorer, a paper-equivalent result, or a performance test.
"""

import argparse
import hashlib
from importlib import metadata
import inspect
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workload-preparation" / "src"))
from vllm_hust_workload_data.registry import source_spec  # noqa: E402


RECIPE_SHA = "7fb84f2c423dc4486103fc8209095862aa75bb6848432f27cc8c705d440e62b9"  # pragma: allowlist secret - pinned lm-eval recipe checksum
METRIC_SHA = "bf217f7dae049098959886592a059b5132faf45fb456139eb76b9a16dd4ba0a1"  # pragma: allowlist secret - pinned lm-eval metric checksum
HARNESS_VERSION = "0.4.11"


def checked_endpoint(value):
    endpoint = value.rstrip("/")
    parsed = urlparse(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.path != "/v1"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("only local HTTP /v1 endpoints without credentials permitted")
    return endpoint


def checked_asset(path, asset):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != asset["sha256"]:
        raise ValueError(f"fixed BBH asset identity mismatch: {asset['path']}")
    if asset.get("bytes") is not None and len(raw) != asset["bytes"]:
        raise ValueError(f"fixed BBH asset size mismatch: {asset['path']}")
    return raw


def publisher_prefix(raw, expected_demos=3):
    text = raw.decode("utf-8")
    if text.count("\n-----\n") != 1:
        raise ValueError("publisher prompt header boundary changed")
    prefix = text.split("\n-----\n", 1)[1]
    demos = [
        part.split("\nA:", 1)[0].removeprefix("Q: ")
        for part in prefix.split("\n\n")
        if part.startswith("Q: ")
    ]
    if len(demos) != expected_demos or any(not question for question in demos):
        raise ValueError("publisher prompt demonstration count mismatch")
    return prefix, demos


def task_prompt(prefix, row):
    if not isinstance(row.get("input"), str) or not row["input"].strip():
        raise ValueError("original nonempty task input required")
    return prefix + "\n\nQ: " + row["input"] + "\nA: Let's think step by step.\n"


def post(url, payload):
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=300) as response:
        return json.load(response)


def selected_assets(task_ids):
    spec = source_spec("BBH")
    task_assets = {
        Path(asset["path"]).stem: asset
        for asset in spec["files"]
        if asset["role"] == "data"
    }
    prompt_assets = {
        Path(asset["path"]).stem: asset
        for asset in spec["files"]
        if asset["role"] == "publisher_prompt"
    }
    if len(task_assets) != 27 or task_assets.keys() != prompt_assets.keys():
        raise ValueError("pinned manifest must register all 27 BBH task/prompt pairs")
    if (
        not task_ids
        or len(set(task_ids)) != len(task_ids)
        or any(task_id not in task_assets for task_id in task_ids)
    ):
        raise ValueError(
            "task IDs must be distinct tasks registered in the pinned manifest"
        )
    return spec, task_assets, prompt_assets


def load_task(data_root, task_id, task_asset, prompt_asset):
    task_raw = checked_asset(Path(data_root) / task_asset["path"], task_asset)
    prompt_raw = checked_asset(Path(data_root) / prompt_asset["path"], prompt_asset)
    try:
        task = json.loads(task_raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError(f"invalid original task JSON: {task_id}") from None
    rows = task.get("examples") if isinstance(task, dict) else None
    if not isinstance(rows, list) or len(rows) != task_asset["rows"]:
        raise ValueError(f"original task row count mismatch: {task_id}")
    if any(
        not isinstance(row, dict)
        or not isinstance(row.get("input"), str)
        or not row["input"].strip()
        or not isinstance(row.get("target"), str)
        or not row["target"].strip()
        for row in rows
    ):
        raise ValueError(f"original task fields incomplete: {task_id}")
    prefix, _ = publisher_prefix(
        prompt_raw, expected_demos=prompt_asset.get("publisher_demos", 3)
    )
    return rows, prefix


def pinned_harness():
    if metadata.version("lm-eval") != HARNESS_VERSION:
        raise ValueError(f"requires reviewed lm-eval=={HARNESS_VERSION} environment")
    import lm_eval
    from lm_eval.api.registry import get_metric
    from lm_eval.filters.extraction import RegexFilter
    import yaml

    recipe_path = (
        Path(lm_eval.__file__).parent
        / "tasks/bbh/cot_fewshot/_cot_fewshot_template_yaml"
    )
    recipe_raw = recipe_path.read_bytes()
    if hashlib.sha256(recipe_raw).hexdigest() != RECIPE_SHA:
        raise ValueError("pinned lm-eval BBH recipe identity mismatch")
    config = yaml.safe_load(recipe_raw)
    metric = get_metric("exact_match")
    metric_path = Path(inspect.getfile(metric))
    if hashlib.sha256(metric_path.read_bytes()).hexdigest() != METRIC_SHA:
        raise ValueError("pinned lm-eval exact_match implementation identity mismatch")
    extractor = RegexFilter(
        regex_pattern=config["filter_list"][0]["filter"][0]["regex_pattern"]
    )
    return config, metric, extractor


def run(args):
    endpoint = checked_endpoint(args.endpoint)
    if args.max_model_len <= 0:
        raise ValueError("declared max model length must be a positive token count")
    spec, task_assets, prompt_assets = selected_assets(args.task_ids)
    config, metric, extractor = pinned_harness()
    if args.output.exists():
        raise FileExistsError("private output directory must not already exist")
    os.umask(0o077)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    receipt = {
        "source_id": "BBH",
        "revision": spec["revision"],
        "task_ids": list(args.task_ids),
        "sample_policy": "all_original_rows"
        if args.full
        else "first_three_original_rows",
        "selected_indices_by_task": {},
        "tasks": [],
        "model": args.model,
        "max_model_len": args.max_model_len,
        "started_at_unix": time.time(),
        "status": "STARTED",
        "harness_version": HARNESS_VERSION,
        "harness_recipe_sha256": RECIPE_SHA,
        "harness_metric_sha256": METRIC_SHA,
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scoring_scope": "pinned lm-eval BBH regex and exact_match; not publisher scoring code",
        "rendering": "original publisher three-shot CoT prefix and unchanged task input; model-native chat transport",
        "generation_kwargs": config["generation_kwargs"],
        "full_suite_claim": False,
        "paper_equivalence": False,
        "performance_claim": False,
    }
    try:
        for task_id in args.task_ids:
            rows, prefix = load_task(
                args.data_root,
                task_id,
                task_assets[task_id],
                prompt_assets[task_id],
            )
            selected = rows if args.full else rows[:3]
            receipt["selected_indices_by_task"][task_id] = list(range(len(selected)))
            task_result = {
                "task_id": task_id,
                "source_rows": len(rows),
                "task_sha256": task_assets[task_id]["sha256"],
                "publisher_prompt_sha256": prompt_assets[task_id]["sha256"],
                "samples": [],
            }
            for index, row in enumerate(selected):
                body = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": task_prompt(prefix, row)}],
                    "temperature": 0,
                    "max_tokens": config["generation_kwargs"]["max_gen_toks"],
                    "stop": config["generation_kwargs"]["until"],
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                capacity = post(
                    endpoint.removesuffix("/v1") + "/tokenize",
                    {
                        "model": args.model,
                        "messages": body["messages"],
                        "add_generation_prompt": True,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                if capacity["count"] + body["max_tokens"] > args.max_model_len:
                    raise ValueError(
                        f"complete original input exceeds declared capacity: {task_id}:{index}"
                    )
                response = post(endpoint + "/chat/completions", body)
                private_path = args.output / f"{task_id}-sample-{index}.private.json"
                private_path.write_text(
                    json.dumps(
                        {
                            "request": body,
                            "response": response,
                            "reference": row["target"],
                        }
                    ),
                    encoding="utf-8",
                )
                choices = response.get("choices", [])
                if len(choices) != 1 or not isinstance(
                    choices[0].get("message", {}).get("content"), str
                ):
                    raise ValueError(
                        f"one complete textual response required: {task_id}:{index}"
                    )
                content = choices[0]["message"]["content"]
                filtered = extractor.apply([[content]], [row])[0][0]
                task_result["samples"].append(
                    {
                        "original_index": index,
                        "input_sha256": hashlib.sha256(
                            row["input"].encode()
                        ).hexdigest(),
                        "finish_reason": choices[0].get("finish_reason"),
                        "stop_reason": choices[0].get("stop_reason"),
                        "answer_extracted": filtered != extractor.fallback,
                        "preflight_tokens": capacity["count"],
                        "prompt_tokens": response["usage"]["prompt_tokens"],
                        "completion_tokens": response["usage"]["completion_tokens"],
                        "metrics": metric(
                            predictions=[filtered], references=[row["target"]]
                        ),
                    }
                )
                receipt["tasks"] = [
                    *[task for task in receipt["tasks"] if task["task_id"] != task_id],
                    task_result,
                ]
                (args.output / "receipt.json").write_text(
                    json.dumps(receipt, indent=2), encoding="utf-8"
                )
        receipt["status"] = "ORIGINAL_TASK_GENERATIONS_HARNESS_SCORED"
        receipt["all_normal_stop"] = all(
            sample["finish_reason"] == "stop"
            for task in receipt["tasks"]
            for sample in task["samples"]
        )
        receipt["all_answers_extracted"] = all(
            sample["answer_extracted"]
            for task in receipt["tasks"]
            for sample in task["samples"]
        )
        receipt["all_full_prompt_counts_match"] = all(
            sample["prompt_tokens"] == sample["preflight_tokens"]
            for task in receipt["tasks"]
            for sample in task["samples"]
        )
    except BaseException as error:
        receipt.update(status="FAILED", error_type=type(error).__name__)
        raise
    finally:
        receipt["finished_at_unix"] = time.time()
        (args.output / "receipt.json").write_text(
            json.dumps(receipt, indent=2), encoding="utf-8"
        )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:19466/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--task-ids", nargs="+", required=True)
    parser.add_argument(
        "--full",
        action="store_true",
        help="run every original row of each explicitly selected task",
    )
    args = parser.parse_args()
    receipt = run(args)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
