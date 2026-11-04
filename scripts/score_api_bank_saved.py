"""Export native inputs or score saved outputs for fixed API-Bank dialogues."""

import argparse
import ast
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit


FILES = (
    "AddAgenda-AddAlarm-GetUserToken-level-2-1.jsonl",
    "AddAgenda-AddMeeting-GetUserToken-level-2-2.jsonl",
    "AddAgenda-AddMeeting-GetUserToken-level-2-3.jsonl",
)
LOCAL_TOOLS = {"GetUserToken", "AddAgenda", "AddAlarm", "AddMeeting"}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def author_prompts(root):
    names = {"api_call_prompt", "response_prompt"}
    prompts = {}
    tree = ast.parse((root / "evaluator.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                if target.id in prompts:
                    raise ValueError(f"ambiguous author prompt: {target.id}")
                prompts[target.id] = ast.literal_eval(node.value)
    if prompts.keys() != names or not all(isinstance(v, str) for v in prompts.values()):
        raise ValueError("missing author prompts")
    return prompts


def author_messages(prompt, descriptions, history):
    messages = [{"role": "system", "content": prompt + descriptions}]
    for item in history:
        role = item["role"]
        if role == "API":
            params = ", ".join(f"{k}='{v}'" for k, v in item["param_dict"].items())
            content = (
                f"[{item['api_name']}({params})] Response: {item['result']['output']}"
            )
            messages.append({"role": "system", "content": content})
        else:
            messages.append(
                {
                    "role": {"User": "user", "AI": "assistant"}[role],
                    "content": item["text"],
                }
            )
    return messages


def generate(endpoint, model, requests, output):
    url = urlsplit(endpoint)
    if (
        url.scheme != "http"
        or url.hostname != "127.0.0.1"
        or not url.port
        or url.path != "/v1"
        or url.username is not None
        or url.query
        or url.fragment
        or not model.strip()
    ):
        raise ValueError("explicit loopback /v1 endpoint and model required")
    with output.open("x", encoding="utf-8") as stream:
        for row in requests:
            body = {"model": model, "messages": row["messages"]}
            client = HTTPConnection(url.hostname, url.port, timeout=600)
            try:
                client.request(
                    "POST",
                    "/v1/chat/completions",
                    json.dumps(body),
                    {"Content-Type": "application/json"},
                )
                reply = client.getresponse()
                if reply.status != 200:
                    raise RuntimeError(
                        f"HTTP {reply.status} for {row['file']}:{row['id']}"
                    )
                result = json.loads(reply.read())
            finally:
                client.close()
            choices = result["choices"]
            if len(choices) != 1 or not isinstance(
                choices[0]["message"]["content"], str
            ):
                raise ValueError("expected one text completion")
            record = {
                "file": row["file"],
                "id": row["id"],
                "output": choices[0]["message"]["content"],
                "finish_reason": choices[0].get("finish_reason"),
                "usage": result.get("usage"),
            }
            stream.write(json.dumps(record) + "\n")
            stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--predictions", type=Path)
    mode.add_argument("--export-inputs", action="store_true")
    mode.add_argument("--export-messages", action="store_true")
    mode.add_argument("--generate", type=Path, metavar="PRIVATE_OUTPUT")
    parser.add_argument("--endpoint")
    parser.add_argument("--model")
    args = parser.parse_args()
    if args.generate is not None and not (args.endpoint and args.model):
        parser.error("--generate requires --endpoint and --model")
    destination = args.generate.resolve() if args.generate else None
    root = args.author_root.resolve(strict=True)
    predictions = {}
    try:
        rows = [] if args.predictions is None else read_jsonl(args.predictions)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        parser.error(f"cannot read prediction JSONL: {exc}")
    for line_number, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            parser.error(f"line {line_number}: prediction must be an object")
        key = (row.get("file"), row.get("id"))
        if (
            key[0] not in FILES
            or type(key[1]) is not int
            or key[1] < 0
            or key in predictions
            or not isinstance(row.get("output"), str)
        ):
            parser.error(f"line {line_number}: invalid or duplicate prediction")
        predictions[key] = row["output"]

    os.chdir(root)
    sys.path.insert(0, str(root))
    from api_call_extraction import get_api_call, parse_api_call
    from evaluator_by_json import Evaluator, Sample

    cases = {}
    for name in FILES:
        rows = read_jsonl(root / "lv1-lv2-samples/level-1-given-desc" / name)
        if any(
            row.get("role") == "API" and row.get("api_name") not in LOCAL_TOOLS
            for row in rows
        ):
            parser.error("unexpected tool in fixed dialogue")
        samples = Sample.from_chat_history(rows)
        evaluator = Evaluator(samples)
        for i, sample in enumerate(samples):
            if sample.ground_truth["role"] in {"API", "AI"}:
                cases[name, i] = (evaluator, sample.ground_truth["role"])
    if args.predictions is None:
        prompts = author_prompts(root) if args.export_messages or destination else None
        requests = []
        for (name, i), (evaluator, role) in cases.items():
            descriptions, history = evaluator.get_model_input(i)
            if prompts is not None:
                prompt = prompts[
                    "api_call_prompt" if role == "API" else "response_prompt"
                ]
                requests.append(
                    {
                        "file": name,
                        "id": i,
                        "messages": author_messages(prompt, descriptions, history),
                    }
                )
                continue
            print(
                json.dumps(
                    {
                        "file": name,
                        "id": i,
                        "api_descriptions": descriptions,
                        "chat_history": history,
                    }
                )
            )
        if destination:
            generate(args.endpoint, args.model, requests, destination)
        else:
            for row in requests:
                print(json.dumps(row))
        return
    if predictions.keys() != cases.keys():
        parser.error("predictions must match every fixed scoring position exactly")
    results = []
    for (name, i), (evaluator, role) in cases.items():
        output = predictions[name, i]
        if role == "API":
            call = get_api_call(output)
            score = False
            if call and parse_api_call(call)[0] in LOCAL_TOOLS:
                score = bool(evaluator.evaluate(i, call)[0])
        else:
            # The publisher's inference loop bypasses ROUGE for empty replies.
            score = evaluator.evaluate(i, output) if output else 0
        results.append({"file": name, "id": i, "role": role, "score": score})
    print(json.dumps(results))


if __name__ == "__main__":
    main()
