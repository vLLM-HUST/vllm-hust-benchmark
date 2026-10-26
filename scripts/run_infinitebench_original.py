"""Original InfiniteBench tasks via pinned author prompts/scoring and local API.

Server only. Original rows, task budgets and author scorer remain intact. This
is an explicit local-model transport profile, not a GPT-4 paper replication.
No context truncation, task substitution, automatic retry or performance claim.
"""

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from verify_infinitebench_original_code import verify


def select(rows, *, start=0, count=3, full=False):
    if full:
        if start != 0 or count != 3:
            raise ValueError("full selection cannot combine with a range")
        return list(enumerate(rows))
    if start < 0 or count < 3 or start + count > len(rows):
        raise ValueError("preselect at least three original positions within range")
    return list(enumerate(rows))[start : start + count]


def execute(
    rows,
    *,
    task,
    utils,
    scorer,
    client,
    token_count,
    model,
    template,
    context_limit,
    output,
    data_dir,
    start=0,
    count=3,
    full=False,
    disable_thinking=False,
):
    selected = select(rows, start=start, count=count, full=full)
    if template not in utils.MODEL_TO_PROMPT_TEMPLATE:
        raise ValueError("explicit author template profile required")
    if task not in utils.DATA_NAME_TO_MAX_NEW_TOKENS:
        raise ValueError("unknown original task")
    budget = utils.DATA_NAME_TO_MAX_NEW_TOKENS[task]
    output = Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    receipt = {
        "status": "RUNNING_NOT_COMPLETE",
        "task": task,
        "model": model,
        "author_template_profile": template,
        "scorer_model_name": model,
        "selected_original_positions": [index for index, _ in selected],
        "full_selection": full,
        "max_new_tokens": budget,
        "completed_original_positions": [],
        "attempted_original_positions": [],
        "private_protocol_records": [],
        "native_execution_complete": False,
        "source_checkbox_eligible": False,
        "task_truncation": False,
        "paper_model_replication": False,
        "performance_claim": False,
        "model_chat_template_kwargs": {"enable_thinking": False}
        if disable_thinking
        else {},
    }

    def save():
        (output / "receipt-private.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )

    save()
    try:
        predictions, labels = [], []
        with (output / "predictions-private.jsonl").open(
            "x", encoding="utf-8"
        ) as stream:
            for index, row in selected:
                messages, _ = utils.create_msgs(
                    None, row, task, data_dir=str(data_dir), model_name=template
                )
                tokens = token_count(messages)
                if (
                    type(tokens) is not int
                    or tokens < 1
                    or tokens + budget > context_limit
                ):
                    raise ValueError(
                        "complete original context plus author budget does not fit; no trimming"
                    )
                label = utils.get_answer(row, task)
                body = dict(
                    model=model, messages=messages, max_tokens=budget, temperature=0
                )
                if disable_thinking:
                    body["extra_body"] = {
                        "chat_template_kwargs": {"enable_thinking": False}
                    }
                request_path = output / f"row-{index}.request-private.json"
                request_path.write_text(
                    json.dumps(body, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                protocol = {
                    "row_index": index,
                    "preflight_input_tokens": tokens,
                    "request_sha256": hashlib.sha256(
                        request_path.read_bytes()
                    ).hexdigest(),
                    "response_sha256": None,
                }
                receipt["private_protocol_records"].append(protocol)
                receipt["attempted_original_positions"].append(index)
                save()
                response = client.chat.completions.create(**body)
                response_path = output / f"row-{index}.response-private.json"
                response_path.write_text(
                    json.dumps(response.model_dump(mode="json"), ensure_ascii=False)
                    + "\n",
                    encoding="utf-8",
                )
                protocol["response_sha256"] = hashlib.sha256(
                    response_path.read_bytes()
                ).hexdigest()
                save()
                if len(response.choices) != 1:
                    raise ValueError("one original completion choice required")
                choice = response.choices[0]
                if choice.finish_reason not in {"stop", "length"} or not isinstance(
                    choice.message.content, str
                ):
                    raise ValueError("missing or unsuccessful model completion")
                prediction = choice.message.content
                predictions.append(prediction)
                labels.append(label)
                stream.write(
                    json.dumps(
                        {
                            "id": index,
                            "prediction": prediction,
                            "ground_truth": label,
                            "finish_reason": choice.finish_reason,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                stream.flush()
                receipt["completed_original_positions"].append(index)
                save()
        score = scorer.get_score(labels, predictions, task, model)
        if (
            type(score) not in {int, float}
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ValueError("official scorer returned invalid aggregate")
        receipt.update(
            status="SELECTED_ORIGINAL_TASKS_AND_AUTHOR_SCORE_COMPLETE",
            score=score,
            native_execution_complete=True,
        )
    except Exception as exc:
        receipt.update(status="FAILED_NOT_COMPLETE", error_type=type(exc).__name__)
        raise
    finally:
        save()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--context-limit", type=int, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:19466/v1")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Explicit compatible model transport option; original task prompt and budget unchanged",
    )
    args = parser.parse_args()
    route = urlparse(args.endpoint)
    if (
        route.scheme != "http"
        or route.hostname not in {"127.0.0.1", "localhost", "::1"}
        or route.path.rstrip("/") != "/v1"
        or route.username
        or route.password
        or route.query
        or route.fragment
    ):
        raise ValueError("explicit loopback OpenAI endpoint required")
    if args.context_limit <= 0:
        raise ValueError("positive verified service context limit required")
    data_root, output = args.data_root.resolve(), args.output.resolve()
    if output.is_relative_to(data_root) or data_root.is_relative_to(output):
        raise ValueError("private output must be separate from original assets")
    verify(args.official_checkout)
    from vllm_hust_workload_data.assets import verify_file
    from vllm_hust_workload_data.registry import source_spec

    source = source_spec("INFINITEBENCH")
    pins = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/workload-native/infinitebench-author-code-pins.json"
        ).read_text("utf-8")
    )
    if source["revision"] != pins["data_revision"]:
        raise ValueError("original data revision drift requires review")
    spec = next(
        (item for item in source["files"] if item["path"] == args.task + ".jsonl"), None
    )
    if spec is None:
        raise ValueError("registered original task asset required")
    data_identity = verify_file(args.data_root, spec)
    sys.path.insert(0, str(args.official_checkout.resolve() / "src"))
    utils = importlib.import_module("eval_utils")
    scorer = importlib.import_module("compute_scores")
    from openai import OpenAI

    client = OpenAI(
        base_url=args.endpoint,
        api_key="local-only",  # pragma: allowlist secret - dummy for loopback-only service, not a credential
        max_retries=0,
        timeout=600,
    )

    def token_count(messages):
        payload = {
            "model": args.model,
            "messages": messages,
            "add_generation_prompt": True,
        }
        if args.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        body = json.dumps(payload).encode()
        request = Request(
            args.endpoint.rstrip("/")[:-3] + "/tokenize",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=60) as response:
            return json.load(response)["count"]

    rows = utils.load_data(args.task, data_dir=str(args.data_root))
    receipt = execute(
        rows,
        task=args.task,
        utils=utils,
        scorer=scorer,
        client=client,
        token_count=token_count,
        model=args.model,
        template=args.template,
        context_limit=args.context_limit,
        output=args.output,
        data_dir=args.data_root,
        start=args.start,
        count=args.count,
        full=args.full,
        disable_thinking=args.disable_thinking,
    )
    try:
        verify_file(args.data_root, spec)
        verify(args.official_checkout)
    except Exception:
        receipt.update(
            status="FINAL_IDENTITY_RECHECK_FAILED", native_execution_complete=False
        )
        (args.output / "receipt-private.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        raise
    receipt.update(
        official_code_revision=pins["revision"],
        original_data_revision=source["revision"],
        original_data_identity=data_identity,
    )
    (args.output / "receipt-private.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "status",
                    "task",
                    "native_execution_complete",
                    "source_checkbox_eligible",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
