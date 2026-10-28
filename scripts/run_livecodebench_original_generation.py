"""Send isolated-author-prepared LiveCodeBench requests to a local model.

Only the HTTP transport is implemented here. Prompts must come from the pinned
author formatter, not reconstructed questions. No code execution or scoring.
"""

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def local_endpoint(value):
    parsed = urlparse(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.path.rstrip("/") != "/v1"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("loopback HTTP /v1 endpoint required; no paid remote API")
    return value.rstrip("/")


def requests_from(path):
    rows = json.loads(Path(path).read_text("utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("nonempty isolated-author request list required")
    ids = []
    expected = {
        "n": 10,
        "max_tokens": 2000,
        "temperature": 0.2,
        "top_p": 0.95,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    for row in rows:
        body = row["request"]
        if not isinstance(row.get("question_id"), str) or not row["question_id"]:
            raise ValueError("original question identifier required")
        if any(body.get(k) != v for k, v in expected.items()):
            raise ValueError("original code-generation defaults must remain unchanged")
        if set(body) != set(expected) | {"model", "messages"}:
            raise ValueError("only original API request fields accepted")
        if not isinstance(body["model"], str) or not body["model"]:
            raise ValueError("actual model identifier required")
        if not isinstance(body["messages"], list) or not body["messages"]:
            raise ValueError("unchanged original messages required")
        if any(
            not isinstance(m, dict)
            or set(m) != {"role", "content"}
            or m["role"] not in {"system", "user", "assistant"}
            or not isinstance(m["content"], str)
            for m in body["messages"]
        ):
            raise ValueError("original text messages required")
        ids.append(row["question_id"])
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate original question identifiers")
    return rows


def request(endpoint, body, timeout):
    wire = Request(
        endpoint + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(wire, timeout=timeout) as response:
        return json.load(response)


def run(
    prompts, output, *, endpoint, disable_thinking=False, timeout=90, transport=request
):
    endpoint = local_endpoint(endpoint)
    prompts, output = Path(prompts).resolve(), Path(output).resolve()
    rows = requests_from(prompts)
    if output.exists() or output.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("fresh external output required; never overwrite evidence")
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("positive transport timeout required")
    output.mkdir(mode=0o700, parents=True)
    receipt = {
        "source_id": "LIVECODEBENCH",
        "status": "GENERATING_ORIGINAL_TASKS",
        "source_checkbox_eligible": False,
        "generated_code_executed": False,
        "native_scoring_complete": False,
        "attempted_requests": 0,
        "completed_question_ids": [],
        "model_samples_per_question": 10,
        "transport_timeout_seconds": timeout,
        "automatic_retries": 0,
        "disable_thinking": disable_thinking,
    }
    try:
        for position, row in enumerate(rows):
            body = dict(row["request"])
            if disable_thinking:
                body["chat_template_kwargs"] = {"enable_thinking": False}
            saved = {"question_id": row["question_id"], "request": body}
            path = output / f"{position}.request-response.private.json"
            path.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
            receipt["attempted_requests"] += 1
            reply = transport(endpoint, body, timeout)
            saved["response"] = reply
            path.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
            choices = reply.get("choices", [])
            if len(choices) != 10 or sorted(c.get("index") for c in choices) != list(
                range(10)
            ):
                raise ValueError(
                    "original n=10 candidates missing; do not replace task"
                )
            if any(
                not isinstance(c.get("message", {}).get("content"), str)
                or not c["message"]["content"].strip()
                for c in choices
            ):
                raise ValueError("empty original completion retained; no replacement")
            receipt["completed_question_ids"].append(row["question_id"])
        receipt["status"] = "ORIGINAL_GENERATION_RETURNED_NATIVE_SCORING_PENDING"
    except BaseException as error:
        receipt.update(
            status="STOP_RETAINED_NEEDS_TRIAGE", error_type=type(error).__name__
        )
        raise
    finally:
        (output / "receipt-private.json").write_text(
            json.dumps(receipt, indent=2), encoding="utf-8"
        )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:19466/v1")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "OFFLINE_PLAN_ONLY",
                    "model_calls": 0,
                    "source_checkbox_eligible": False,
                }
            )
        )
        return
    print(
        json.dumps(
            run(
                args.prompts,
                args.output,
                endpoint=args.endpoint,
                disable_thinking=args.disable_thinking,
                timeout=args.timeout,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
