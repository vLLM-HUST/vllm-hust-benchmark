"""Run original Video-MME questions with declared official eight-frame profiles.

Plan by default. Requires verified complete original media; preserves upstream
prompt, frame/subtitle selection, API encoding and exact-matching scoring.
"""

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from prepare_chartqa_native_toolkit import verify
from prepare_videomme_original_view import REVISION, ROOT, prepare_view
from run_chartqa_original import validate_completion, validate_endpoint

PROFILES = {"Video-MME_8frame": False, "Video-MME_8frame_subs": True}


def prepare_subtitle_dependency(overlay=None):
    if overlay is not None:
        overlay = Path(overlay).resolve(strict=True)
        if not overlay.is_dir() or overlay.is_relative_to(ROOT):
            raise ValueError("External isolated subtitle dependency directory required")
        sys.path.insert(0, str(overlay))
    import pysubs2

    if importlib.metadata.version("pysubs2") != "1.8.0":
        raise ValueError("Qualified pysubs2==1.8.0 required, never auto-upgrade")
    if overlay is not None and not Path(pysubs2.__file__).resolve().is_relative_to(
        overlay
    ):
        raise ValueError("Subtitle import did not use the declared isolated overlay")


def pin_native_cache(snapshot):
    """Supply upstream's required cache ref, never rewrite an existing ref."""
    if (
        snapshot.name != REVISION
        or snapshot.parent.name != "snapshots"
        or snapshot.parent.parent.name != "datasets--lmms-lab--Video-MME"
    ):
        raise ValueError("Pinned official repository cache layout required")
    refs = snapshot.parent.parent / "refs"
    refs.mkdir(exist_ok=True)
    target = refs / "main"
    if target.exists():
        if target.read_text("utf-8").strip() != REVISION:
            raise ValueError("Existing native cache main ref differs; not overwritten")
    else:
        with target.open("x", encoding="utf-8") as stream:
            stream.write(REVISION)
    return snapshot.parent.parent.parent


def fixed_positions(records):
    if len(records) != 2700:
        raise ValueError("Complete original2700 table required")
    selected = []
    for duration, start in (("short", 0), ("medium", 900), ("long", 1800)):
        group = records[start : start + 900]
        if any(row["duration"] != duration for row in group):
            raise ValueError("Original duration order changed")
        videos = list(dict.fromkeys(row["video"] for row in group))
        if len(videos) != 300:
            raise ValueError("Original300 distinct videos per duration required")
        first = videos[:3]
        positions = [
            start + index for index, row in enumerate(group) if row["video"] in first
        ]
        if positions != list(range(start, start + 9)) or any(
            sum(row["video"] == video for row in group) != 3 for video in first
        ):
            raise ValueError("Original three questions per selected video changed")
        selected.extend(positions)
    return selected


def validate_scores(records, selected):
    if len(records) != len(selected):
        raise ValueError("Original scoring did not retain every selected task")
    if [int(row["source_index"]) for row in records] != selected:
        raise ValueError("Original scoring task identities changed")
    if any(row["score"] not in (0, 1) for row in records):
        raise ValueError("Unresolved original scores retained, not completed")


def check_media_receipt(snapshot, receipt, required_paths=()):
    if (
        receipt.get("source_id") != "VIDEOMME"
        or receipt.get("status")
        != "ORIGINAL_FULL_MEDIA_LAYOUT_AND_BYTES_VERIFIED_NOT_NATIVE_INFERENCE"
        or receipt.get("required_original_videos") != 900
        or receipt.get("original_subtitles") != 744
    ):
        raise ValueError("Complete original-media verification receipt required")
    identities = receipt.get("extracted_files", [])
    paths = [item["path"] for item in identities]
    if len(set(paths)) != len(paths) or not identities:
        raise ValueError("Ambiguous or missing verified original-media identities")
    if (
        sum(path.startswith("video/") and path.endswith(".mp4") for path in paths)
        != 900
        or sum(path.startswith("subtitle/") and path.endswith(".srt") for path in paths)
        != 744
    ):
        raise ValueError(
            "Complete original900 video and744 subtitle identities required"
        )
    if not set(required_paths) <= set(paths):
        raise ValueError("Selected native media absent from verification receipt")
    for item in identities:
        target = (snapshot / item["path"]).resolve(strict=True)
        if not target.is_relative_to(snapshot) or not target.is_file():
            raise ValueError("Original media escapes verified snapshot")
        if target.stat().st_size != item["bytes"]:
            raise ValueError("Verified original media has changed")
        if item["path"] in required_paths:
            digest = hashlib.sha256()
            with target.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != item["sha256"]:
                raise ValueError("Selected original media bytes have changed")


def prepare_original_prompts(dataset, selected, output):
    """Freeze original decoded inputs without making any model requests."""
    records = []
    for index in selected:
        prompt = dataset.build_prompt(index, video_llm=False)
        images = [item["value"] for item in prompt if item["type"] == "image"]
        if len(images) != 8:
            raise ValueError("Original eight-frame input required")
        hashes = [
            hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in images
        ]
        records.append(
            {"source_index": index, "original_prompt": prompt, "frame_sha256": hashes}
        )
    (output / "original-prompts-private.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "status": "ORIGINAL_VIDEO_PROMPTS_PREPARED_NOT_NATIVE_INFERENCE",
        "selected_original_positions": selected,
        "prepared_original_prompts": len(records),
        "frames_each": 8,
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--media-receipt", type=Path, required=True)
    parser.add_argument("--frame-cache", type=Path, required=True)
    parser.add_argument(
        "--subtitle-overlay",
        type=Path,
        help="Optional external pysubs2==1.8.0 target installation",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:19466/v1/chat/completions"
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--prepare-prompts-only",
        action="store_true",
        help="With execute, decode/freeze original prompts only; no model/scoring calls",
    )
    parser.add_argument("--accept-academic-only", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    validate_endpoint(args.endpoint)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "PLAN_ONLY",
                    "profile": args.profile,
                    "selected_original_tasks": 2700 if args.full else 27,
                    "frames_each": 8,
                    "model_calls": 0,
                    "source_checkbox_eligible": False,
                }
            )
        )
        return
    if not args.accept_academic_only:
        raise ValueError("Explicit academic-only terms acknowledgement required")
    source = args.official_checkout.resolve(strict=True)
    snapshot = args.snapshot.resolve(strict=True)
    frames = args.frame_cache.resolve(strict=True)
    output = args.output.resolve()
    if (
        output.exists()
        or any(
            path.is_relative_to(ROOT)
            or output.is_relative_to(path)
            or path.is_relative_to(output)
            for path in (source, snapshot, frames)
        )
        or output.is_relative_to(ROOT)
    ):
        raise ValueError("Fresh private output and external assets/cache required")
    proof = {
        **verify(source, archive=True),
        "source_id": "VIDEOMME",
        "scope": "Shared original toolkit source tree, not native task completion",
    }
    prepare_view(source, snapshot)
    with (snapshot / "Video-MME.tsv").open(encoding="utf-8", newline="") as stream:
        original_records = list(csv.DictReader(stream, delimiter="\t"))
    fixed = fixed_positions(original_records)
    selected = list(range(2700)) if args.full else fixed
    required_paths = set()
    for index in selected:
        row = original_records[index]
        required_paths.add(f"video/{row['video']}.mp4")
        subtitle = f"subtitle/{row['video']}.srt"
        if PROFILES[args.profile] and (snapshot / subtitle).exists():
            required_paths.add(subtitle)
    check_media_receipt(
        snapshot, json.loads(args.media_receipt.read_text("utf-8")), required_paths
    )
    cache = pin_native_cache(snapshot)
    if PROFILES[args.profile]:
        prepare_subtitle_dependency(args.subtitle_overlay)
    sys.dont_write_bytecode = True
    os.environ.update(
        LMUData=str(frames),
        TORCH_DEVICE_BACKEND_AUTOLOAD="0",
        HF_HUB_OFFLINE="1",
        HF_HUB_CACHE=str(cache),
        HUGGINGFACE_HUB_CACHE=str(cache),
    )
    os.environ.pop("PYTHONPATH", None)
    sys.path.insert(0, str(source))
    import pandas as pd
    from vlmeval.api.gpt import OpenAIWrapper
    from vlmeval.dataset.videomme import VideoMME
    from vlmeval.dataset.utils.judge_cache import get_judge_detail_file
    from vlmeval.smp import dump, load
    from vlmeval.smp import get_cache_path

    resolved_cache = get_cache_path("lmms-lab/Video-MME")
    if resolved_cache is None or Path(resolved_cache).resolve() != snapshot:
        raise ValueError("Native cache lookup mismatch; no implicit download")
    dataset = VideoMME(
        dataset="Video-MME", nframe=8, fps=-1, use_subtitle=PROFILES[args.profile]
    )
    if Path(dataset.data_root).resolve() != snapshot:
        raise ValueError("Original toolkit selected a different dataset snapshot")
    if fixed_positions(dataset.data.to_dict(orient="records")) != fixed:
        raise ValueError("Toolkit table changed original fixed selection")
    if args.prepare_prompts_only:
        output.mkdir(mode=0o700, parents=True, exist_ok=False)
        result = prepare_original_prompts(dataset, selected, output)
        result.update(source_id="VIDEOMME", profile=args.profile, source_proof=proof)
        verify(source, archive=True)
        (output / "receipt-private.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {key: value for key, value in result.items() if key != "source_proof"},
                indent=2,
            )
        )
        return
    model = OpenAIWrapper(
        model=args.model, key="local-only", api_base=args.endpoint, retry=1, timeout=300
    )  # pragma: allowlist secret - local dummy key
    if model.max_tokens != 2048 or model.temperature != 0:
        raise ValueError("Original model generation defaults drifted")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    receipt = {
        "source_id": "VIDEOMME",
        "status": "RUNNING",
        "profile": args.profile,
        "model": args.model,
        "selected_original_positions": selected,
        "attempted_original_positions": [],
        "completed_original_positions": [],
        "source_checkbox_eligible": False,
        "paid_judge_calls": 0,
        "source_proof": proof,
        "records": [],
    }

    def save():
        (output / "receipt-private.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )

    save()
    try:
        predictions = []
        for local_index, index in enumerate(selected):
            prompt = dataset.build_prompt(index, video_llm=False)
            if sum(item["type"] == "image" for item in prompt) != 8:
                raise ValueError("Original eight-frame input required")
            message, system = model.preprocess_message_with_role(copy.deepcopy(prompt))
            message = model.preproc_content(message)
            kwargs = copy.deepcopy(model.default_kwargs)
            kwargs.update(
                system_prompt=system, chat_template_kwargs={"enable_thinking": False}
            )
            receipt["attempted_original_positions"].append(index)
            save()
            code, answer, response = model.generate_inner(message, **kwargs)
            raw = response.request.body
            raw = raw.encode() if isinstance(raw, str) else raw
            (output / f"row-{index}.request-private.json").write_bytes(raw)
            (output / f"row-{index}.response-private.json").write_text(
                response.text, encoding="utf-8"
            )
            if code != 0 or response.status_code != 200:
                raise ValueError("Original request failed; no task replacement")
            verified = validate_completion(
                json.loads(raw),
                response.json(),
                model=args.model,
                messages=model.prepare_inputs(message, system),
                disable_thinking=True,
            )
            if not verified or answer != verified:
                raise ValueError("Missing or inconsistent native answer")
            record = dataset.data.iloc[index].to_dict()
            record.update(index=local_index, source_index=index, prediction=answer)
            predictions.append(record)
            dump(pd.DataFrame(predictions), str(output / "predictions-private.xlsx"))
            receipt["records"].append(
                {
                    "source_index": index,
                    "evaluation_index": local_index,
                    "usage": response.json()["usage"],
                    "finish_reason": response.json()["choices"][0]["finish_reason"],
                    "request_sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
            receipt["completed_original_positions"].append(index)
            save()
        prediction_file = str(output / "predictions-private.xlsx")
        receipt["author_metrics"] = dataset.evaluate(
            prediction_file, model="exact_matching"
        )
        details = load(
            get_judge_detail_file(prediction_file, "extract", "exact_matching")
        )
        validate_scores(details.to_dict(orient="records"), selected)
        verify(source, archive=True)
        receipt["status"] = (
            "SELECTED_ORIGINAL_TASKS_AND_SCORING_COMPLETE_NOT_SOURCE_DELIVERY"
        )
    except BaseException as error:
        receipt.update(
            status="FAILED_RETAINED", error_type=type(error).__name__, error=str(error)
        )
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
                    "profile",
                    "completed_original_positions",
                    "source_checkbox_eligible",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
