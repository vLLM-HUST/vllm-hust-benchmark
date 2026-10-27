"""Original MMMU prompts and author scorer through a local multimodal service.

Default is a plan. No source rewriting, external judge, retry or replacement.
"""

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.request import Request, urlopen

from prepare_mmmu_author_source import verify as verify_source
from prepare_mmmu_original_assets import PINS
from prepare_xsum_original_assets import verify as verify_file
from run_chartqa_original import validate_endpoint


def image_messages(prompt, row):
    """Preserve original marker placement, repeated references and image bytes."""
    parts = re.split(r"(<image [1-7]>)", prompt)
    references = {int(value) for value in re.findall(r"<image ([1-7])>", prompt)}
    available = {i for i in range(1, 8) if row.get(f"image_{i}") is not None}
    if references != available:
        raise ValueError(
            "Original image references differ from populated image fields; no dropping images"
        )
    content = []
    for part in parts:
        match = re.fullmatch(r"<image ([1-7])>", part)
        if not match:
            if part:
                content.append({"type": "text", "text": part})
            continue
        image = row[f"image_{match[1]}"]
        raw = image.get("bytes")
        if not isinstance(raw, bytes) or not raw:
            raise ValueError(
                "Original embedded image bytes required; no external path resolution"
            )
        if raw.startswith(b"\x89PNG\r\n\x1a\n"):
            mime = "image/png"
        elif raw.startswith(b"\xff\xd8"):
            mime = "image/jpeg"
        elif raw[:6] in (b"GIF87a", b"GIF89a"):
            mime = "image/gif"
        elif raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
            mime = "image/webp"
        else:
            raise ValueError("Unsupported original image encoding; preserve failure")
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime};base64," + base64.b64encode(raw).decode()
                },
            }
        )
    return [{"role": "user", "content": content}]


def validate_response(value, budget):
    choices = value.get("choices", [])
    if len(choices) != 1 or choices[0].get("finish_reason") not in {"stop", "length"}:
        raise ValueError("Failed original completion; do not replace question")
    answer = choices[0].get("message", {}).get("content")
    usage = value.get("usage", {})
    values = [
        usage.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    ]
    if not isinstance(answer, str) or any(type(v) is not int for v in values):
        raise ValueError("Native answer and actual token usage required")
    if values[0] < 1 or not 0 <= values[1] <= budget or sum(values[:2]) != values[2]:
        raise ValueError("Invalid native token accounting")
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing original subject folders (preparer output/data)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split", choices=["dev", "validation", "test"], default="validation"
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:19466/v1/chat/completions"
    )
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    route = validate_endpoint(args.endpoint)
    if not args.execute:
        print(
            json.dumps(
                {
                    "source_id": "MMMU",
                    "status": "PLAN_ONLY",
                    "split": args.split,
                    "selection": "full original split"
                    if args.full
                    else "first three original rows per each of30subjects",
                    "budget": 128,
                    "model_calls": 0,
                    "source_checkbox_eligible": False,
                }
            )
        )
        return
    source = args.official_checkout.resolve(strict=True)
    proof = verify_source(source)
    if args.output.exists():
        raise ValueError(
            "Fresh private output required; never overwrite prior attempts"
        )
    pins = json.loads(PINS.read_text(encoding="utf-8"))["dataset"]
    files = [
        item
        for item in pins["files"]
        if item["path"].split("/")[1].startswith(args.split + "-")
    ]
    for item in files:
        verify_file(args.data_root / item["path"], item)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source / "mmmu"))
    from utils.data_utils import CAT_SHORT2LONG, construct_prompt, load_yaml
    import pyarrow.parquet as pq

    config = load_yaml(str(source / "mmmu/configs/llava1.5.yaml"))
    config = {
        key: value[0] if isinstance(value, list) and len(value) == 1 else value
        for key, value in config.items()
    }
    with urlopen(f"{route.scheme}://{route.netloc}/metrics", timeout=10) as response:
        metrics = response.read().decode()
    active = [
        float(line.split()[-1])
        for line in metrics.splitlines()
        if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{"))
    ]
    if not active or any(active):
        raise ValueError("Existing active service users have priority")
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    receipt = {
        "source_id": "MMMU",
        "status": "RUNNING",
        "source": proof,
        "dataset_revision": pins["revision"],
        "split": args.split,
        "model": args.model,
        "max_tokens": 128,
        "temperature": config["temperature"],
        "enable_thinking": not args.disable_thinking,
        "completed": [],
        "attempted": [],
        "model_calls": 0,
        "source_checkbox_eligible": False,
        "paid_judge_calls": 0,
        "llava_reproduction_claim": False,
    }

    def save():
        (args.output / "receipt-private.json").write_text(
            json.dumps(receipt, indent=2), encoding="utf-8"
        )

    save()
    try:
        for short, subject in CAT_SHORT2LONG.items():
            originals = []
            for item in sorted(
                (item for item in files if item["path"].startswith(subject + "/")),
                key=lambda item: item["path"],
            ):
                originals.extend(
                    pq.read_table(args.data_root / item["path"]).to_pylist()
                )
            if len(originals) < 3:
                raise ValueError("Missing original subject rows")
            selected = originals if args.full else originals[:3]
            folder = args.output / subject
            folder.mkdir()
            outputs = []
            for position, row in enumerate(selected):
                options = ast.literal_eval(row["options"])
                if not isinstance(options, list) or not all(
                    isinstance(v, str) for v in options
                ):
                    raise ValueError(
                        "Original literal option list required before author eval"
                    )
                sample = construct_prompt(dict(row), config)
                messages = image_messages(sample["final_input_prompt"], row)
                body = {
                    "model": args.model,
                    "messages": messages,
                    "max_tokens": 128,
                    "temperature": config["temperature"],
                }
                if args.disable_thinking:
                    body["chat_template_kwargs"] = {"enable_thinking": False}
                raw = json.dumps(body).encode()
                task_id = row["id"]
                receipt["attempted"].append(task_id)
                save()
                request_path = folder / f"row-{position}.request-private.json"
                request_path.write_bytes(raw)
                receipt["model_calls"] += 1
                save()
                with urlopen(
                    Request(
                        args.endpoint,
                        data=raw,
                        headers={"Content-Type": "application/json"},
                    ),
                    timeout=300,
                ) as response:
                    result_raw = response.read()
                (folder / f"row-{position}.response-private.json").write_bytes(
                    result_raw
                )
                value = json.loads(result_raw)
                answer = validate_response(value, 128)
                output = {
                    "id": task_id,
                    "question_type": row["question_type"],
                    "answer": row["answer"],
                    "response": answer,
                }
                if row["question_type"] == "multiple-choice":
                    output.update(
                        all_choices=sample["all_choices"], index2ans=sample["index2ans"]
                    )
                outputs.append(output)
                receipt["completed"].append(
                    {
                        "id": task_id,
                        "subject": subject,
                        "position": position,
                        "image_references": len(
                            [
                                p
                                for p in messages[0]["content"]
                                if p["type"] == "image_url"
                            ]
                        ),
                        "usage": value["usage"],
                        "finish_reason": value["choices"][0]["finish_reason"],
                        "request_sha256": hashlib.sha256(raw).hexdigest(),
                        "response_sha256": hashlib.sha256(result_raw).hexdigest(),
                    }
                )
                save()
            (folder / "output.json").write_text(
                json.dumps(outputs, ensure_ascii=False), encoding="utf-8"
            )
            # Author CLI unchanged; freeze its documented random fallback seed.
            code = "import random,runpy,sys;random.seed(42);sys.argv=['main_parse_and_eval.py','--path',sys.argv[1],'--subject',sys.argv[2]];runpy.run_path('main_parse_and_eval.py',run_name='__main__')"
            with (folder / "scoring-private.log").open("xb") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        "-c",
                        code,
                        str(args.output.resolve()),
                        short,
                    ],
                    cwd=source / "mmmu",
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=180,
                )
        verify_source(source)
        for item in files:
            verify_file(args.data_root / item["path"], item)
        receipt["status"] = (
            "ORIGINAL_SELECTED_TASKS_AND_AUTHOR_SCORING_COMPLETE_NOT_SOURCE_DELIVERY"
        )
    except BaseException as error:
        receipt.update(
            status="FAILED_RETAINED_NOT_COMPLETE",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    finally:
        save()
    print(
        json.dumps(
            {
                "source_id": "MMMU",
                "status": receipt["status"],
                "tasks": len(receipt["completed"]),
                "source_checkbox_eligible": False,
            }
        )
    )


if __name__ == "__main__":
    main()
