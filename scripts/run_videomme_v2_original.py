"""Run frozen original Video-MME-v2 64-frame tasks and publisher group scoring.

Plan by default. Reuses verified original prompts; never regenerates questions,
replaces failures, reduces frames, or calls a paid judge. Scores are for the
declared fixed sample, not an official full leaderboard reproduction.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import sys

from prepare_chartqa_native_toolkit import verify
from prepare_videomme_v2_original_prompts import (
    PROFILE,
    ROOT,
    check_prompt,
    fixed_positions,
    sha256,
    verify_original_tsv,
)
from run_chartqa_original import validate_completion, validate_endpoint


def validate_record(record, original, position, prepared):
    if (
        record["source_index"] != position
        or record["original_question_id"] != original["question_id"]
        or record["video_id"] != original["video_id"]
    ):
        raise ValueError("original question identity or ordering changed")
    images = check_prompt(record["original_prompt"], original)
    hashes = record["frame_sha256"]
    if len(hashes) != 64:
        raise ValueError("all64 original frame identities required")
    for name, expected in zip(images, hashes, strict=True):
        path = Path(name).resolve(strict=True)
        if not path.is_relative_to(prepared) or sha256(path) != expected:
            raise ValueError("original prepared frame escaped root or changed")


def load_prepared(prepared):
    from vllm_hust_workload_data.assets import iter_rows, verify_file
    from vllm_hust_workload_data.registry import source_spec

    receipt = json.loads((prepared / "qualification-public.json").read_bytes())
    if (
        receipt["source_id"] != "VIDEOMME_V2"
        or receipt["status"]
        != "ORIGINAL_FIXED24_64FRAME_PROMPTS_PREPARED_NOT_NATIVE_INFERENCE"
        or receipt["frames_each"] != 64
        or receipt["prepared_original_questions"] != 24
        or receipt["distinct_frame_files"] != 384
        or receipt["with_subtitle"]
        or receipt["reasoning"]
        or receipt["resize_target_area"]
    ):
        raise ValueError("verified original base64-frame preparation required")
    profile = json.loads(PROFILE.read_bytes())
    snapshot = (
        prepared
        / "hf/hub/datasets--MME-Benchmarks--Video-MME-v2/snapshots"
        / profile["dataset_revision"]
    )
    spec = source_spec("VIDEOMME_V2")
    if spec["revision"] != profile["dataset_revision"]:
        raise ValueError("original annotation revision changed")
    for item in spec["files"]:
        verify_file(snapshot, item)
    rows = list(iter_rows(snapshot / "test.parquet"))
    verify_original_tsv(snapshot / "Video-MME-v2.tsv", rows)
    positions = fixed_positions(rows, profile["selected_video_ids"])
    prompt_file = prepared / "original-prompts-private.json"
    if sha256(prompt_file) != receipt["private_prompts_sha256"]:
        raise ValueError("prepared original prompt bytes changed")
    records = json.loads(prompt_file.read_bytes())
    if len(records) != 24 or receipt["selected_original_positions"] != positions:
        raise ValueError("frozen complete question groups changed")
    for record, position in zip(records, positions, strict=True):
        validate_record(record, rows[position], position, prepared)
    return rows, positions, records


def execute(args):
    source = args.official_checkout.resolve(strict=True)
    prepared = args.prepared_root.resolve(strict=True)
    output = args.output.resolve()
    if (
        output.exists()
        or output.is_relative_to(ROOT)
        or any(
            output.is_relative_to(path) or path.is_relative_to(output)
            for path in (source, prepared)
        )
    ):
        raise ValueError("fresh external output separate from original inputs required")
    validate_endpoint(args.endpoint)
    proof = verify(source, archive=True)
    rows, positions, records = load_prepared(prepared)
    os.environ.update(
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TORCH_DEVICE_BACKEND_AUTOLOAD="0",
    )
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    import pandas as pd
    from vlmeval.api.gpt import OpenAIWrapper
    from vlmeval.dataset.videommev2 import VideoMMEv2
    from vlmeval.smp import dump, get_intermediate_file_path, load

    model = OpenAIWrapper(
        model=args.model, key="local-only", api_base=args.endpoint, retry=1, timeout=600
    )  # pragma: allowlist secret
    if model.max_tokens != 2048 or model.temperature != 0:
        raise ValueError("publisher OpenAI wrapper generation defaults changed")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    receipt = {
        "source_id": "VIDEOMME_V2",
        "status": "RUNNING_ORIGINAL_FIXED_SAMPLE",
        "profile": "Video-MME-v2_64frame",
        "model": args.model,
        "selected_original_positions": positions,
        "attempted_original_positions": [],
        "completed_original_positions": [],
        "records": [],
        "source_proof": proof,
        "frames_each": 64,
        "max_tokens": 2048,
        "temperature": 0,
        "thinking_enabled": False,
        "paid_judge_calls": 0,
        "source_checkbox_eligible": False,
    }
    predictions = []

    def save():
        (output / "receipt-private.json").write_text(
            json.dumps(receipt, indent=2), encoding="utf-8"
        )

    save()
    try:
        for local_index, (position, record) in enumerate(
            zip(positions, records, strict=True)
        ):
            validate_record(record, rows[position], position, prepared)
            message, system = model.preprocess_message_with_role(
                copy.deepcopy(record["original_prompt"])
            )
            message = model.preproc_content(message)
            kwargs = copy.deepcopy(model.default_kwargs)
            kwargs.update(
                system_prompt=system, chat_template_kwargs={"enable_thinking": False}
            )
            receipt["attempted_original_positions"].append(position)
            save()
            code, answer, response = model.generate_inner(message, **kwargs)
            if response is None:
                raise ValueError(
                    "native request failed without response; do not retry or replace"
                )
            raw = response.request.body
            raw = raw.encode() if isinstance(raw, str) else raw
            (output / f"row-{position}.request-private.json").write_bytes(raw)
            (output / f"row-{position}.response-private.json").write_text(
                response.text, encoding="utf-8"
            )
            if code != 0 or response.status_code != 200:
                raise ValueError("native request failed; original failure retained")
            verified = validate_completion(
                json.loads(raw),
                response.json(),
                model=args.model,
                messages=model.prepare_inputs(message, system),
                disable_thinking=True,
            )
            if not verified or verified != answer:
                raise ValueError("native original completion missing or inconsistent")
            row = dict(rows[position])
            row.update(index=local_index, source_index=position, prediction=answer)
            predictions.append(row)
            dump(pd.DataFrame(predictions), str(output / "predictions-private.xlsx"))
            receipt["completed_original_positions"].append(position)
            receipt["records"].append(
                {
                    "source_index": position,
                    "usage": response.json()["usage"],
                    "finish_reason": response.json()["choices"][0]["finish_reason"],
                }
            )
            save()
        prediction_file = str(output / "predictions-private.xlsx")
        receipt["publisher_grouped_rating"] = VideoMMEv2.evaluate(
            prediction_file, model="exact_matching"
        )
        scores = load(get_intermediate_file_path(prediction_file, "_score"))
        if (
            len(scores) != 24
            or list(scores["source_index"]) != positions
            or not scores["score"].isin([0, 1]).all()
        ):
            raise ValueError(
                "original complete-group score identities or outcomes unresolved"
            )
        verify(source, archive=True)
        receipt["status"] = (
            "ORIGINAL_FIXED24_64FRAME_INFERENCE_AND_GROUPED_SCORING_COMPLETE_NOT_SOURCE_DELIVERY"
        )
    except BaseException as error:
        receipt.update(
            status="FAILED_RETAINED",
            error_type=type(error).__name__,
            error=str(error)[:400],
        )
        raise
    finally:
        save()
    return {
        key: receipt[key]
        for key in (
            "source_id",
            "status",
            "completed_original_positions",
            "source_checkbox_eligible",
        )
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("official-checkout", "prepared-root", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--model")
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:19466/v1/chat/completions"
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "source_id": "VIDEOMME_V2",
                    "status": "PLAN_ONLY",
                    "original_questions": 24,
                    "frames_each": 64,
                    "model_calls": 0,
                }
            )
        )
        return
    if any(
        getattr(args, name) is None
        for name in ("official_checkout", "prepared_root", "output", "model")
    ):
        parser.error(
            "explicit original source, prepared prompts, model and fresh output required"
        )
    os.umask(0o077)
    print(json.dumps(execute(args)))


if __name__ == "__main__":
    main()
