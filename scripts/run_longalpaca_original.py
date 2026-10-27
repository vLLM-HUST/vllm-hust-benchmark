"""Native-model execution of original LongAlpaca instructions, no quality score.

This published constructed training corpus has no held-out native benchmark
scorer. Preserve original instructions and optional input, use the deployed
model's native chat formatting, keep output/file annotations outside requests.
Not LongLoRA SFT, original Llama2-template/model reproduction, or model ranking.
"""

import argparse
import hashlib
import json
from pathlib import Path

from run_oasst_original_continuation import post, validate_endpoint
from run_toolace_original_continuation import (
    local_tokenizer_files,
    verify_tokenizer_files,
)

REVISION = "46dce924ed8786979556018e191c0f557d8f4aa2"  # pragma: allowlist secret - public original dataset commit
DATA_SHA = "090b348755c728cfb4b13fa443b54a0cb4e5128a92f142c6a52c5b4ff0f55e1f"  # pragma: allowlist secret - original public dataset checksum


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def originals(path):
    path = Path(path)
    if path.stat().st_size != 498315126 or sha(path) != DATA_SHA:
        raise ValueError("complete unchanged original LongAlpaca required")
    rows = json.loads(path.read_text("utf-8"))
    if not isinstance(rows, list) or len(rows) != 12000:
        raise ValueError("complete original12000 records required")
    for row in rows:
        if (
            not isinstance(row, dict)
            or not {"instruction", "output"} <= set(row)
            or set(row) - {"instruction", "output", "input", "file"}
            or not all(isinstance(value, str) for value in row.values())
        ):
            raise ValueError("original field contract changed; re-review, no coercion")
    return rows


def select(rows, *, start=0, count=3, full=False):
    if full:
        if start != 0 or count != 3:
            raise ValueError("full cannot mix with range")
        return list(enumerate(rows))
    if (
        type(start) is not int
        or type(count) is not int
        or start < 0
        or count < 3
        or start + count > len(rows)
    ):
        raise ValueError("preselect at least three original positions")
    return list(enumerate(rows))[start : start + count]


def messages(row):
    content = row["instruction"]
    if row.get("input", "") != "":
        content += "\n\n" + row["input"]
    return [{"role": "user", "content": content}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:19466/v1")
    parser.add_argument("--model", default="workload-qwen35-27b-128k-nonthinking")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    endpoint = validate_endpoint(args.endpoint)
    data, tokenizer_root, output = (
        args.data_file.resolve(),
        args.tokenizer.resolve(),
        args.output.resolve(),
    )
    if output.exists() or any(
        output.is_relative_to(root) or root.is_relative_to(output)
        for root in (data.parent, tokenizer_root)
    ):
        raise ValueError("fresh separate output required")
    rows = originals(data)
    selected = select(rows, start=args.start, count=args.count, full=args.full)
    from transformers import AutoTokenizer

    identity = local_tokenizer_files(tokenizer_root)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_root, local_files_only=True, trust_remote_code=False
    )
    prepared = []
    for index, row in selected:
        prompt = tokenizer.apply_chat_template(
            messages(row),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        if tokens <= 0 or tokens + 8192 > 131072:
            raise ValueError(
                "complete input exceeds deployment, never truncate or swap"
            )
        prepared.append((index, row, prompt, tokens))
    output.mkdir(mode=0o700, parents=True)

    def save(name, obj):
        path = output / name
        path.write_text(
            json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        path.chmod(0o600)

    receipt = {
        "source_id": "LONGALPACA",
        "status": "PREPARED_NOT_EXECUTED",
        "revision": REVISION,
        "dataset_sha256": DATA_SHA,
        "script_sha256": sha(__file__),
        "model": args.model,
        "original_rows": len(rows),
        "selected_original_positions": [i for i, _ in selected],
        "tokenizer_files": identity,
        "max_tokens": 8192,
        "context_capacity": 131072,
        "budget_boundary": "Declared inference-workload continuation budget, not a publisher benchmark budget or original SFT protocol",
        "input_truncated": False,
        "reference_output_or_file_annotation_in_input": False,
        "model_calls": 0,
        "cases": [],
        "quality_scored": False,
        "publisher_model_or_training_replication": False,
        "source_checkbox_eligible": False,
    }
    save("receipt-private.json", receipt)
    save(
        "preflight-private.json",
        [{"original_index": i, "input_tokens": tokens} for i, _, _, tokens in prepared],
    )
    if args.prepare_only:
        verify_tokenizer_files(tokenizer_root, identity)
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "selected": len(selected),
                    "input_tokens": [p[3] for p in prepared],
                    "model_calls": 0,
                }
            )
        )
        return
    try:
        for index, row, prompt, tokens in prepared:
            payload = {
                "model": args.model,
                "prompt": prompt,
                "max_tokens": 8192,
                "temperature": 0,
                "seed": 20261026,
                "stream": False,
            }
            save(f"case-{index:05d}-original.private.json", row)
            save(f"case-{index:05d}-request.private.json", payload)
            receipt["model_calls"] += 1
            receipt["status"] = "RUNNING"
            save("receipt-private.json", receipt)
            response = post(endpoint + "/completions", payload)
            save(f"case-{index:05d}-response.private.json", response)
            choices, usage = response.get("choices"), response.get("usage", {})
            if (
                not isinstance(choices, list)
                or len(choices) != 1
                or not isinstance(choices[0].get("text"), str)
                or choices[0].get("finish_reason") not in {"stop", "length"}
                or usage.get("prompt_tokens") != tokens
                or type(usage.get("completion_tokens")) is not int
                or not 0 <= usage["completion_tokens"] <= 8192
            ):
                raise ValueError("native terminal response integrity failed")
            receipt["cases"].append(
                {
                    "original_index": index,
                    "finish_reason": choices[0]["finish_reason"],
                    "usage": usage,
                    "empty_output_retained": choices[0]["text"] == "",
                }
            )
            save("receipt-private.json", receipt)
        verify_tokenizer_files(tokenizer_root, identity)
        if sha(data) != DATA_SHA:
            raise ValueError("original asset changed")
        receipt["status"] = (
            "ORIGINAL_INSTRUCTIONS_NATIVE_MODEL_TERMINALS_VERIFIED_NO_QUALITY_SCORE"
        )
        save("receipt-private.json", receipt)
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "model_calls": receipt["model_calls"],
                    "cases": receipt["cases"],
                    "quality_scored": False,
                }
            )
        )
    except Exception as error:
        receipt.update(status="FAILED_NOT_REPLACED", error_type=type(error).__name__)
        save("receipt-private.json", receipt)
        raise


if __name__ == "__main__":
    main()
