"""Execute fixed official get_pred with full-input capacity checks and model routing."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

import argparse
import zipfile
from urllib.parse import urlparse


def validate_capacity_selection(capacity, model, model_max_length=131072):
    """Keep every explicitly selected original task; reject, never filter, overflow."""
    if (
        not isinstance(capacity, dict)
        or capacity.get("model") != model
        or type(capacity.get("model_max_length")) is not int
        or type(model_max_length) is not int
        or not 128 < model_max_length <= 2097152
        or capacity.get("model_max_length") != model_max_length
    ):
        raise ValueError("capacity profile differs from model/window binding")
    selected = capacity.get("selected")
    if not isinstance(selected, list) or not 1 <= len(selected) <= 503:
        raise ValueError("one to 503 explicitly preselected original tasks required")
    if any(
        not isinstance(row, dict)
        or type(row.get("row_index")) is not int
        or not 0 <= row["row_index"] < 503
        for row in selected
    ):
        raise ValueError("invalid original row position")
    indices = [row["row_index"] for row in selected]
    if len(set(indices)) != len(indices):
        raise ValueError("duplicate original task rows")
    if indices != sorted(indices):
        raise ValueError("selected tasks must retain original row order")
    for row in selected:
        digest = row.get("prompt_sha256")
        if (
            not isinstance(row.get("id"), str)
            or not row["id"]
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("original task ID and complete prompt digest required")
        if (
            row.get("status") != "FULL_INPUT_FITS"
            or type(row.get("full_prompt_tokens")) is not int
            or not 0 < row["full_prompt_tokens"] <= model_max_length - 128
        ):
            raise ValueError("complete original inputs do not fit deployment window")
    if len({row["id"] for row in selected}) != len(selected) or len(
        {row["prompt_sha256"] for row in selected}
    ) != len(selected):
        raise ValueError("duplicate task IDs or ambiguous prompt identities")
    return selected


def validate_deployment(models, model, model_max_length):
    matches = [item for item in models if item.get("id") == model]
    if len(matches) != 1 or matches[0].get("max_model_len") != model_max_length:
        raise ValueError("live model/window differs from explicit deployment binding")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--official-archive", type=Path, required=True)
    parser.add_argument("--capacity-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:19466/v1")
    parser.add_argument("--model", default="workload-qwen35-27b-128k-nonthinking")
    parser.add_argument("--model-max-length", type=int, default=131072)
    args = parser.parse_args()
    url = urlparse(args.endpoint)
    if (
        url.scheme != "http"
        or url.hostname not in ("127.0.0.1", "localhost")
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path.rstrip("/") != "/v1"
    ):
        parser.error(
            "only local HTTP NPU endpoints ending in /v1 without credentials/query are permitted"
        )
    if (
        hashlib.sha256(args.data_file.read_bytes()).hexdigest()
        != "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2"  # pragma: allowlist secret - original data checksum
    ):  # pragma: allowlist secret - original data checksum
        raise ValueError("original LongBench-v2 data identity mismatch")
    if (
        hashlib.sha256(args.official_archive.read_bytes()).hexdigest()
        != "2f92f9953e3dd2fd3915a132a0a4cb53acb879af6ef1e98962129a6f3fe262bb"  # pragma: allowlist secret - official archive checksum
    ):  # pragma: allowlist secret - official archive checksum
        raise ValueError("official LongBench archive identity mismatch")
    capacity = json.loads(args.capacity_selection.read_text(encoding="utf-8"))
    selected = validate_capacity_selection(capacity, args.model, args.model_max_length)
    rows = json.loads(args.data_file.read_text())
    items = [copy.deepcopy(rows[r["row_index"]]) for r in selected]
    if [r["_id"] for r in items] != [r["id"] for r in selected]:
        raise ValueError("original task identities differ from capacity selection")
    run = args.output.resolve()
    os.umask(0o077)
    run.mkdir(mode=0o700, parents=True, exist_ok=False)
    with zipfile.ZipFile(args.official_archive) as archive:
        archive.extractall(run / "official-source")
    source_dir_name = (
        "THUDM-LongBench-2e00731"  # pragma: allowlist secret - public archive directory
    )
    source = run / "official-source" / source_dir_name
    from openai import OpenAI

    prior = Path.cwd()
    os.chdir(source)
    try:
        spec = importlib.util.spec_from_file_location(
            "official_longbench_pred", source / "pred.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        os.chdir(prior)
    alias = args.model
    model_path = str(args.tokenizer.resolve())
    endpoint = args.endpoint
    module.model_map = {alias: model_path}
    module.maxlen_map = {alias: args.model_max_length}
    module.URL = endpoint
    expected = {r["prompt_sha256"]: r for r in selected}
    client = OpenAI(
        base_url=endpoint,
        api_key="local-unused",  # pragma: allowlist secret - dummy loopback key
        max_retries=0,
        timeout=300,  # pragma: allowlist secret - dummy loopback key
    )
    validate_deployment(
        [item.model_dump() for item in client.models.list().data],
        alias,
        args.model_max_length,
    )
    requests = []
    attempts = []

    def routed_create(**kwargs):
        if (
            kwargs["model"] != model_path
            or len(kwargs["messages"]) != 1
            or kwargs["messages"][0]["role"] != "user"
        ):
            raise ValueError("unexpected model routing or official message structure")
        prompt = kwargs["messages"][0]["content"]
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        if digest not in expected:
            raise RuntimeError(
                "Official prompt changed or was truncated; refused generation"
            )
        if kwargs["temperature"] != 0.1 or kwargs["max_tokens"] != 128:
            raise ValueError("original decoding budget/configuration changed")
        kwargs["model"] = alias
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        attempts.append({"id": expected[digest]["id"], "prompt_sha256": digest})
        request_number = len(attempts)
        (run / f"request-{request_number}.private.json").write_text(json.dumps(kwargs))
        response = client.chat.completions.create(**kwargs)
        (run / f"response-{request_number}.private.json").write_text(
            response.model_dump_json()
        )
        usage = response.usage
        if usage.prompt_tokens != expected[digest]["full_prompt_tokens"]:
            raise ValueError(
                "actual prompt token count differs from complete-input preflight"
            )
        requests.append(
            {
                "id": expected[digest]["id"],
                "prompt_sha256": digest,
                "full_prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "finish_reason": response.choices[0].finish_reason,
                "full_input_verified": True,
            }
        )
        return response

    def routed_client(*, base_url, api_key):
        if base_url != endpoint:
            raise ValueError("unapproved model endpoint refused")
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=routed_create))
        )

    module.OpenAI = routed_client
    receipt = {
        "source_id": "LONGBENCH_V2",
        "source_revision": "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9",  # pragma: allowlist secret - public dataset revision
        "runner_revision": "2e00731f8d0bff23dc4325161044d0ed8af94c1e",  # pragma: allowlist secret - public official source revision
        "selection": "explicitly preselected original-order tasks with complete prompt plus128 output bound; no automatic filtering, replacement or truncation",
        "selected_samples": [r["_id"] for r in items],
        "original_row_indices": [r["row_index"] for r in selected],
        "status": "STARTED",
        "started_at_unix": time.time(),
        "model": alias,
        "model_max_length": args.model_max_length,
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.1,
        "max_new_tokens": 128,
        "execution": "official get_pred/extract_answer; only deployment map and client model route bound",
        "full_suite_claim": False,
        "input_truncation_permitted": False,
    }
    target = run / "receipt.json"
    target.write_text(json.dumps(receipt, indent=2))
    try:
        with (run / "predictions.private.jsonl").open("x") as output:
            module.get_pred(
                items,
                SimpleNamespace(model=alias, rag=0, no_context=False, cot=False),
                output,
            )
        results = [
            json.loads(line)
            for line in (run / "predictions.private.jsonl").read_text().splitlines()
        ]
        if len(results) != len(items) or len(requests) != len(items):
            raise ValueError("original tasks did not all produce scored results")
        if [r["_id"] for r in results] != receipt["selected_samples"]:
            raise ValueError("scored task IDs differ from preselection")
        receipt.update(
            status="END_TO_END_EXECUTED_AND_SCORED",
            requests=requests,
            tasks=[
                {"id": r["_id"], "length_label": r["length"], "correct": r["judge"]}
                for r in results
            ],
            correct_tasks=sum(r["judge"] for r in results),
            unique_tasks=len(items),
            scorer="unchanged official extract_answer then pred==answer, no compensated score",
        )
    except BaseException as error:
        receipt.update(status="EXECUTION_ERROR", error_type=type(error).__name__)
        raise
    finally:
        receipt["requests"] = requests
        receipt["model_request_attempts"] = attempts
        receipt["finished_at_unix"] = time.time()
        target.write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
