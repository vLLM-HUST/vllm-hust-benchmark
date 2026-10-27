"""Run three original BBH Boolean tasks with publisher prompts and pinned harness scoring.

Use prepare-bbh-boolean first. Requires separate lm-eval==0.4.11 environment.
Not the publisher's original model/scorer, a full BBH score or performance test.
All task/response payloads remain in the explicitly selected private output directory.
"""

import argparse
import hashlib
from importlib import metadata
import inspect
import json
import os
from pathlib import Path
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen


REVISION = "9ee07bd481feebf959a6b59d61ea57bdcf30964d"  # pragma: allowlist secret - public publisher revision
TASK_SHA = "ea6c754ec005e2d3f2d085d349a740f593b5764f32ea4638ffed4cfc0061b12a"  # pragma: allowlist secret - original task file checksum
PROMPT_SHA = "685affba97425cdf3549d5d313ba876caed2fa7c8a4cf1ea7549faf22179609d"  # pragma: allowlist secret - original publisher prompt checksum
RECIPE_SHA = "7fb84f2c423dc4486103fc8209095862aa75bb6848432f27cc8c705d440e62b9"  # pragma: allowlist secret - pinned lm-eval recipe checksum
METRIC_SHA = "bf217f7dae049098959886592a059b5132faf45fb456139eb76b9a16dd4ba0a1"  # pragma: allowlist secret - pinned lm-eval metric module checksum


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


def checked_asset(path, digest):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("fixed original asset identity mismatch")
    return raw


def publisher_prefix(raw):
    text = raw.decode("utf-8")
    if text.count("\n-----\n") != 1:
        raise ValueError("publisher prompt header boundary changed")
    prefix = text.split("\n-----\n", 1)[1]
    demos = [
        part.split("\nA:", 1)[0].removeprefix("Q: ")
        for part in prefix.split("\n\n")
        if part.startswith("Q: ")
    ]
    if len(demos) != 3 or any(not question for question in demos):
        raise ValueError("three original publisher demonstrations required")
    return prefix, demos


def task_prompt(prefix, row):
    if not isinstance(row.get("input"), str) or not row["input"].strip():
        raise ValueError("original nonempty task input required")
    # Do not inspect or format target/reference here.
    return prefix + "\n\nQ: " + row["input"] + "\nA: Let's think step by step.\n"


def post(url, payload):
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=300) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:19466/v1")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    endpoint = checked_endpoint(args.endpoint)
    if args.output.exists():
        raise FileExistsError("private output directory must not already exist")
    rows = json.loads(
        checked_asset(args.data_root / "boolean_expressions.json", TASK_SHA)
    )["examples"]
    prefix, demos = publisher_prefix(
        checked_asset(
            args.data_root / "cot-prompts/boolean_expressions.txt", PROMPT_SHA
        )
    )
    if len(rows) != 250 or any(row["input"] in demos for row in rows[:3]):
        raise ValueError(
            "original task count or demonstration/evaluation boundary changed"
        )
    if metadata.version("lm-eval") != "0.4.11":
        raise ValueError("requires reviewed lm-eval==0.4.11 environment")
    import lm_eval
    from lm_eval.filters.extraction import RegexFilter
    from lm_eval.api.registry import get_metric
    import yaml

    recipe = (
        Path(lm_eval.__file__).parent
        / "tasks/bbh/cot_fewshot/_cot_fewshot_template_yaml"
    )
    config = yaml.safe_load(checked_asset(recipe, RECIPE_SHA))
    metric = get_metric("exact_match")
    checked_asset(Path(inspect.getfile(metric)), METRIC_SHA)
    extractor = RegexFilter(
        regex_pattern=config["filter_list"][0]["filter"][0]["regex_pattern"]
    )
    os.umask(0o077)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    receipt = {
        "source_id": "BBH",
        "subsuite": "boolean_expressions",
        "revision": REVISION,
        "source_rows": 250,
        "selected_indices": [0, 1, 2],
        "samples": [],
        "model": args.model,
        "started_at_unix": time.time(),
        "status": "STARTED",
        "task_sha256": TASK_SHA,
        "publisher_prompt_sha256": PROMPT_SHA,
        "harness_recipe_sha256": RECIPE_SHA,
        "harness_metric_sha256": METRIC_SHA,
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "harness_version": "0.4.11",
        "scoring_scope": "pinned lm-eval BBH regex and exact_match; not publisher scoring code",
        "rendering": "original publisher three-shot CoT prefix and unchanged task input; model-native chat transport",
        "generation_kwargs": config["generation_kwargs"],
        "full_suite_claim": False,
        "paper_equivalence": False,
        "performance_claim": False,
    }
    try:
        for index, row in enumerate(rows[:3]):
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
            if capacity["count"] + body["max_tokens"] > 131072:
                raise ValueError(
                    "complete original input exceeds capacity; no truncation"
                )
            response = post(endpoint + "/chat/completions", body)
            (args.output / f"sample-{index}.private.json").write_text(
                json.dumps(
                    {"request": body, "response": response, "reference": row["target"]}
                ),
                encoding="utf-8",
            )
            choices = response.get("choices", [])
            if len(choices) != 1 or not isinstance(
                choices[0].get("message", {}).get("content"), str
            ):
                raise ValueError("one complete textual response required")
            content = choices[0]["message"]["content"]
            filtered = extractor.apply([[content]], [row])[0][0]
            receipt["samples"].append(
                {
                    "original_index": index,
                    "input_sha256": hashlib.sha256(row["input"].encode()).hexdigest(),
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
            (args.output / "receipt.json").write_text(
                json.dumps(receipt, indent=2), encoding="utf-8"
            )
        receipt["status"] = "ORIGINAL_TASK_GENERATIONS_HARNESS_SCORED"
        receipt["all_normal_stop"] = all(
            s["finish_reason"] == "stop" for s in receipt["samples"]
        )
        receipt["all_answers_extracted"] = all(
            s["answer_extracted"] for s in receipt["samples"]
        )
        receipt["all_full_prompt_counts_match"] = all(
            s["prompt_tokens"] == s["preflight_tokens"] for s in receipt["samples"]
        )
    except BaseException as error:
        receipt.update(status="FAILED", error_type=type(error).__name__)
        raise
    finally:
        receipt["finished_at_unix"] = time.time()
        (args.output / "receipt.json").write_text(
            json.dumps(receipt, indent=2), encoding="utf-8"
        )
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
