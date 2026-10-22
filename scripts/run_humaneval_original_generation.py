"""Generate three original HumanEval completions; NEVER execute or score code.

Requires the pinned optional workload-data component and original publisher gzip.
Generation is not functional-correctness validation. Official scoring must run
separately in a verified execution sandbox, not the model-serving environment.
Output contains private model completions; do not publish the output directory.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen


REVISION = "b9453e8f1865df0a0057a1383767caa400acfbcf"  # pragma: allowlist secret - official source commit
DATA_SHA = "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"  # pragma: allowlist secret - original gzip checksum


def completion_payload(record, model):
    if record.runtime["kind"] != "task_dataset" or set(record.task["input"]) != {
        "prompt"
    }:
        raise ValueError("requires original prompt-only code task input")
    return {
        "model": model,
        "prompt": record.task["input"]["prompt"],
        "temperature": 0,
        "max_tokens": 2048,
        "n": 1,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:19466/v1")
    args = parser.parse_args()
    parsed = urlparse(args.endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") != "/v1"
    ):
        parser.error("only permits local HTTP /v1 endpoints without credentials")
    # Optional import after argument validation: help works in the root-only env.
    from vllm_hust_workload_data import iter_workload
    from vllm_hust_workload_data.registry import source_spec

    spec = source_spec("HUMANEVAL")
    if spec["revision"] != REVISION or spec["files"][0]["sha256"] != DATA_SHA:
        raise ValueError("install the exact reviewed workload-data source/wheel")
    records = list(iter_workload("HUMANEVAL", data_root=args.data_root, limit=3))
    if [r.task["task_id"] for r in records] != [
        "HumanEval/0",
        "HumanEval/1",
        "HumanEval/2",
    ]:
        raise ValueError("requires the first three original publisher tasks")
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    receipt = {
        "schema_version": "humaneval-generation/v1",
        "source_id": "HUMANEVAL",
        "revision": REVISION,
        "data_sha256": DATA_SHA,
        "status": "STARTED",
        "model": args.model,
        "selection": "FIRST_THREE_ORIGINAL_ROWS_BEFORE_EXECUTION",
        "transport": "RAW_COMPLETIONS_NO_CHAT_TEMPLATE_NO_TESTS_OR_SOLUTIONS",
        "generation_config": {"temperature": 0, "max_tokens": 2048, "n": 1},
        "scoring": "NOT_RUN_REQUIRES_VERIFIED_CODE_EXECUTION_SANDBOX",
        "complete_task_validation": False,
        "source_checkbox_eligible": False,
        "started_at_unix": time.time(),
        "samples": [],
    }

    def save():
        target = args.output / "receipt.json"
        target.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        target.chmod(0o600)

    save()
    for record in records:
        payload = completion_payload(record, args.model)
        item = {
            "task_id": record.task["task_id"],
            "status": "REQUESTING",
            "prompt_sha256": hashlib.sha256(payload["prompt"].encode()).hexdigest(),
            "request_attempted": True,
        }
        receipt["samples"].append(item)
        save()
        try:
            request = Request(
                args.endpoint.rstrip("/") + "/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=300) as response:
                result = json.load(response)
            choices = result.get("choices", [])
            if (
                len(choices) != 1
                or not isinstance(choices[0].get("text"), str)
                or not choices[0]["text"]
                or choices[0].get("finish_reason") not in {"stop", "length"}
            ):
                raise ValueError("missing or invalid terminal completion")
            private = args.output / (
                item["task_id"].replace("/", "-") + ".private.json"
            )
            private.write_text(
                json.dumps(
                    {"request": payload, "response": result}, ensure_ascii=False
                ),
                encoding="utf-8",
            )
            private.chmod(0o600)
            item.update(
                status="GENERATION_RECEIVED_NOT_SCORED",
                finish_reason=choices[0]["finish_reason"],
                usage=result.get("usage"),
                private_result_sha256=hashlib.sha256(private.read_bytes()).hexdigest(),
            )
        except Exception as error:
            item.update(status="REQUEST_FAILED", error_type=type(error).__name__)
            receipt["status"] = "GENERATION_FAILED_NOT_SCORED"
            save()
            raise
        save()
    receipt.update(
        status="THREE_ORIGINAL_PROMPTS_GENERATED_NOT_SCORED",
        finished_at_unix=time.time(),
    )
    save()


if __name__ == "__main__":
    main()
