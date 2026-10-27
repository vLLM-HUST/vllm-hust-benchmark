"""Original ChartQA images/questions through pinned, unmodified VLMEvalKit.

Plan by default. Explicit execution uses a loopback vision service and the
original dataset prompt, API encoder and relaxed-accuracy scorer. No external
judge, source rewriting, question substitution or automatic retry.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import ssl
import sys
from urllib.parse import urlparse
from urllib.request import urlopen

from prepare_chartqa_native_toolkit import verify
from prepare_chartqa_original_view import BYTES, NAME, ROWS, SHA256, digest, verify_view


def select_positions(size, indices=None, full=False):
    if full:
        if indices is not None:
            raise ValueError("Full suite cannot combine with selected positions")
        return list(range(size))
    positions = [0, 1, 2] if indices is None else indices
    if len(positions) < 3 or len(set(positions)) != len(positions):
        raise ValueError("Preselect at least three distinct original positions")
    if any(type(index) is not int or index < 0 or index >= size for index in positions):
        raise ValueError("Original positions outside source range")
    if positions != sorted(positions):
        raise ValueError("Selected cases must retain original source order")
    return list(positions)


def validate_endpoint(endpoint):
    route = urlparse(endpoint)
    if (
        route.scheme != "http"
        or route.hostname not in {"localhost", "127.0.0.1", "::1"}
        or route.path.rstrip("/") != "/v1/chat/completions"
        or route.username
        or route.password
        or route.query
        or route.fragment
    ):
        raise ValueError("Credential-free loopback chat endpoint required")
    return route


def validate_completion(body, response, *, model, messages, disable_thinking):
    if body.get("model") != model or body.get("messages") != messages:
        raise ValueError("Actual wire request differs from original toolkit encoder")
    if body.get("max_tokens") != 2048 or body.get("temperature") != 0:
        raise ValueError("Original API wrapper generation defaults changed")
    expected = {"enable_thinking": False} if disable_thinking else None
    if body.get("chat_template_kwargs") != expected:
        raise ValueError("Declared model thinking profile differs from actual request")
    choices = response.get("choices", [])
    if len(choices) != 1:
        raise ValueError("One original completion required")
    choice = choices[0]
    if choice.get("finish_reason") not in {"stop", "length"} or not isinstance(
        choice.get("message", {}).get("content"), str
    ):
        raise ValueError(
            "Unsuccessful completion; retain failure, never replace question"
        )
    usage = response.get("usage", {})
    prompt, completion, total = (
        usage.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    )
    if (
        any(type(value) is not int for value in (prompt, completion, total))
        or prompt < 1
        or not 0 <= completion <= 2048
        or total != prompt + completion
    ):
        raise ValueError("Invalid actual token usage")
    return choice["message"]["content"].strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--toolkit-archive", action="store_true")
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--prepared-view", type=Path, required=True)
    parser.add_argument("--ffmpeg-library-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:19466/v1/chat/completions"
    )
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument(
        "--indices",
        help="Comma-separated, preselected original positions; minimum three",
    )
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    route = validate_endpoint(args.endpoint)
    selected = select_positions(
        ROWS,
        None
        if args.indices is None
        else [int(value) for value in args.indices.split(",")],
        args.full,
    )
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "PLAN_ONLY_NO_MODEL_CALL",
                    "original_positions": selected,
                    "original_api_budget": 2048,
                    "paid_judge_calls": 0,
                    "model": args.model,
                }
            )
        )
        return
    source, data, prepared, output = (
        path.resolve()
        for path in (
            args.official_checkout,
            args.source_file,
            args.prepared_view,
            args.output,
        )
    )
    for asset in (source, prepared):
        if output.is_relative_to(asset) or asset.is_relative_to(output):
            raise ValueError(
                "Private output must not overlap original source or prepared data"
            )
    if data.is_relative_to(output) or data == output or output.exists():
        raise ValueError("Fresh private output separate from originals required")
    if not args.ffmpeg_library_dir.is_dir():
        raise ValueError("Qualified private FFmpeg library directory required")
    source_proof = verify(source, archive=args.toolkit_archive)
    if data.stat().st_size != BYTES or digest(data) != SHA256:
        raise ValueError("Original Parquet identity mismatch")
    sys.dont_write_bytecode = True
    os.environ.update(TORCH_DEVICE_BACKEND_AUTOLOAD="0", LMUData=str(prepared))
    os.environ["LD_LIBRARY_PATH"] = (
        str(args.ffmpeg_library_dir.resolve())
        + os.pathsep
        + os.environ.get("LD_LIBRARY_PATH", "")
    )
    os.environ.pop("PYTHONPATH", None)
    sys.path.insert(0, str(source))
    original_ssl = ssl._create_default_https_context
    try:
        from vlmeval.dataset import ImageVQADataset
        from vlmeval.api.gpt import OpenAIWrapper
        from vlmeval.smp import dump
    finally:
        ssl._create_default_https_context = original_ssl
    import pandas as pd
    import pyarrow.parquet as pq

    originals = pq.read_table(data).to_pylist()
    view = prepared / f"{NAME}.tsv"
    if len(originals) != ROWS or verify_view(originals, view) != ROWS:
        raise ValueError("Complete original data/view relationships required")
    dataset = ImageVQADataset(NAME)
    if len(dataset.data) != ROWS:
        raise ValueError("Toolkit silently removed original rows")
    with urlopen(f"{route.scheme}://{route.netloc}/metrics", timeout=10) as response:
        metrics = response.read().decode()
    active = [
        float(line.split()[-1])
        for line in metrics.splitlines()
        if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{"))
    ]
    if not active or any(value != 0 for value in active):
        raise ValueError("Wait for active users; no automatic competition")
    model = OpenAIWrapper(
        model=args.model, key="local-only", api_base=args.endpoint, retry=1, timeout=300
    )  # pragma: allowlist secret - loopback dummy
    if model.max_tokens != 2048 or model.temperature != 0:
        raise ValueError("Original API wrapper defaults drifted")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    receipt = {
        "source_id": "CHARTQA",
        "status": "RUNNING",
        "model": args.model,
        "model_chat_template_kwargs": {"enable_thinking": False}
        if args.disable_thinking
        else {},
        "selected_original_positions": selected,
        "attempted_original_positions": [],
        "completed_original_positions": [],
        "source_checkbox_eligible": False,
        "source_proof": source_proof,
        "view_sha256": digest(view),
        "official_api_budget": 2048,
        "private_records": [],
        "paid_judge_calls": 0,
        "public_runner_sha256": digest(Path(__file__)),
    }

    def save():
        (output / "receipt-private.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )

    save()
    try:
        outputs = []
        for index in selected:
            line = dataset.data.iloc[index]
            if (
                int(line["index"]) != index
                or line["question"] != originals[index]["query"]
                or json.loads(line["answer"]) != originals[index]["label"]
            ):
                raise ValueError("Toolkit original question/reference/order changed")
            prompt = dataset.build_prompt(index)
            if (
                prompt[-1]["value"]
                != originals[index]["query"]
                + "\nAnswer the question using a single word or phrase."
            ):
                raise ValueError("Original toolkit prompt changed")
            receipt["attempted_original_positions"].append(index)
            save()
            kwargs = (
                {"chat_template_kwargs": {"enable_thinking": False}}
                if args.disable_thinking
                else {}
            )
            code, answer, response = model.generate_inner(prompt, **kwargs)
            raw = response.request.body
            raw = raw.encode() if isinstance(raw, str) else raw
            (output / f"row-{index}.request-private.json").write_bytes(raw)
            (output / f"row-{index}.response-private.json").write_text(
                response.text, encoding="utf-8"
            )
            if code != 0 or response.status_code != 200:
                raise ValueError(
                    "Native API failure retained without retry or replacement"
                )
            actual = response.json()
            verified_answer = validate_completion(
                json.loads(raw),
                actual,
                model=args.model,
                messages=model.prepare_inputs(prompt, None),
                disable_thinking=args.disable_thinking,
            )
            if answer != verified_answer:
                raise ValueError("Native extracted answer differs from response")
            usage = actual["usage"]
            outputs.append(
                {
                    "index": index,
                    "question": line["question"],
                    "answer": line["answer"],
                    "category": line["category"],
                    "prediction": answer,
                }
            )
            receipt["private_records"].append(
                {
                    "original_position": index,
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                    "finish_reason": actual["choices"][0]["finish_reason"],
                    "request_sha256": hashlib.sha256(raw).hexdigest(),
                    "response_sha256": digest(
                        output / f"row-{index}.response-private.json"
                    ),
                }
            )
            receipt["completed_original_positions"].append(index)
            save()
        predictions = output / "predictions-private.xlsx"
        dump(pd.DataFrame(outputs), str(predictions))
        receipt["author_metrics"] = dataset.evaluate(str(predictions)).to_dict(
            orient="records"
        )
        if digest(data) != SHA256 or verify_view(originals, view) != ROWS:
            raise ValueError("Original data/view changed during execution")
        verify(source, archive=args.toolkit_archive)
        receipt["status"] = (
            "SELECTED_ORIGINAL_VISUAL_TASKS_AND_TOOLKIT_SCORE_COMPLETE_NOT_SOURCE_DELIVERY"
        )
    except BaseException as error:
        receipt.update(status="FAILED_RETAINED", error_type=type(error).__name__)
        raise
    finally:
        save()
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "source_id",
                    "status",
                    "selected_original_positions",
                    "completed_original_positions",
                    "author_metrics",
                    "source_checkbox_eligible",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
