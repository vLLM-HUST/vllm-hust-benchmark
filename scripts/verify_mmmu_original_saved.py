"""Offline original MMMU request/response and author-score verification.

No model endpoint or paid judge is called. Original run files are never edited.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from prepare_mmmu_author_source import verify as verify_source
from prepare_mmmu_original_assets import PINS
from prepare_xsum_original_assets import verify as verify_file
from run_mmmu_original import image_messages, validate_response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {"source_id": "MMMU", "status": "PLAN_ONLY", "new_model_calls": 0}
            )
        )
        return
    source = args.official_checkout.resolve(strict=True)
    verify_source(source)
    receipt = json.loads((args.run / "receipt-private.json").read_text())
    if (
        receipt["status"]
        != "ORIGINAL_SELECTED_TASKS_AND_AUTHOR_SCORING_COMPLETE_NOT_SOURCE_DELIVERY"
    ):
        raise ValueError("Incomplete original run; retain failure")
    if args.output.exists():
        raise ValueError("Fresh audit output required")
    pins = json.loads(PINS.read_text(encoding="utf-8"))["dataset"]
    if receipt["dataset_revision"] != pins["revision"] or receipt["max_tokens"] != 128:
        raise ValueError("Original protocol identity mismatch")
    files = [
        item
        for item in pins["files"]
        if item["path"].split("/")[1].startswith(receipt["split"] + "-")
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
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    count, multiple_images = 0, 0
    ids = []
    for short, subject in CAT_SHORT2LONG.items():
        records = []
        for item in sorted(
            (item for item in files if item["path"].startswith(subject + "/")),
            key=lambda item: item["path"],
        ):
            records.extend(pq.read_table(args.data_root / item["path"]).to_pylist())
        completed = [
            entry for entry in receipt["completed"] if entry["subject"] == subject
        ]
        # A source run must cover exactly its predeclared first3 or full split.
        if len(completed) not in {3, len(records)} or len(completed) < 3:
            raise ValueError("Missing declared subject coverage")
        saved = json.loads((args.run / subject / "output.json").read_text())
        if len(saved) != len(completed):
            raise ValueError("Missing original scoring inputs")
        for position, (entry, output) in enumerate(zip(completed, saved)):
            original = records[position]
            options = ast.literal_eval(original["options"])
            if not isinstance(options, list) or not all(
                isinstance(v, str) for v in options
            ):
                raise ValueError("Original literal options required")
            prompt = construct_prompt(dict(original), config)
            raw = (
                args.run / subject / f"row-{position}.request-private.json"
            ).read_bytes()
            response_raw = (
                args.run / subject / f"row-{position}.response-private.json"
            ).read_bytes()
            if (
                hashlib.sha256(raw).hexdigest() != entry["request_sha256"]
                or hashlib.sha256(response_raw).hexdigest() != entry["response_sha256"]
            ):
                raise ValueError("Saved native payload changed")
            body, response = json.loads(raw), json.loads(response_raw)
            expected = {
                "model": receipt["model"],
                "messages": image_messages(prompt["final_input_prompt"], original),
                "max_tokens": 128,
                "temperature": config["temperature"],
            }
            if not receipt["enable_thinking"]:
                expected["chat_template_kwargs"] = {"enable_thinking": False}
            if (
                body != expected
                or entry["id"] != original["id"]
                or entry["position"] != position
            ):
                raise ValueError("Original native input or order mismatch")
            answer = validate_response(response, 128)
            expected_output = {
                "id": original["id"],
                "question_type": original["question_type"],
                "answer": original["answer"],
                "response": answer,
            }
            if original["question_type"] == "multiple-choice":
                expected_output.update(
                    all_choices=prompt["all_choices"], index2ans=prompt["index2ans"]
                )
            if (
                output != expected_output
                or response["usage"] != entry["usage"]
                or response["choices"][0]["finish_reason"] != entry["finish_reason"]
            ):
                raise ValueError("Original reference, output or usage mismatch")
            image_count = len(
                [
                    p
                    for p in expected["messages"][0]["content"]
                    if p["type"] == "image_url"
                ]
            )
            if image_count != entry["image_references"]:
                raise ValueError("Image count mismatch")
            multiple_images += image_count > 1
            count += 1
            ids.append(original["id"])
        folder = args.output / subject
        folder.mkdir()
        (folder / "output.json").write_text(json.dumps(saved), encoding="utf-8")
        code = "import random,runpy,sys;random.seed(42);sys.argv=['main_parse_and_eval.py','--path',sys.argv[1],'--subject',sys.argv[2]];runpy.run_path('main_parse_and_eval.py',run_name='__main__')"
        with (folder / "scoring-private.log").open("xb") as log:
            subprocess.run(
                [sys.executable, "-B", "-c", code, str(args.output.resolve()), short],
                cwd=source / "mmmu",
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=180,
            )
        for name in ("parsed_output.json", "result.json"):
            if json.loads((folder / name).read_text()) != json.loads(
                (args.run / subject / name).read_text()
            ):
                raise ValueError("Original score or parsing changed")
    if (
        count != receipt["model_calls"]
        or ids != receipt["attempted"]
        or len(ids) != len(set(ids))
    ):
        raise ValueError("Original selection has missing or duplicate calls")
    verify_source(source)
    result = {
        "source_id": "MMMU",
        "status": "SAVED_ORIGINAL_PROMPTS_IMAGES_OUTPUTS_AND_AUTHOR_SCORES_VERIFIED",
        "split": receipt["split"],
        "subjects": len(CAT_SHORT2LONG),
        "tasks": count,
        "multi_image_tasks": multiple_images,
        "new_model_calls": 0,
        "paid_judge_calls": 0,
        "original_run_modified": False,
        "source_checkbox_eligible": False,
    }
    (args.output / "receipt-public.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
