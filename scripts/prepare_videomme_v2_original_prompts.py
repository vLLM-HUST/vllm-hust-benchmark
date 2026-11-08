"""Prepare frozen original64-frame Video-MME-v2 prompts, without model calls.

Uses the unchanged pinned VLMEvalKit constructor, frame sampler and prompt
builder. The owned offline Hub cache retains all3200 original annotation rows
but only the six declared validation videos. It is not a full media snapshot.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

from prepare_chartqa_native_toolkit import verify


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/workload-native/videomme-v2-original-fixed-media.json"


def fixed_positions(rows, expected_videos):
    if len(rows) != 3200:
        raise ValueError("all3200 original annotations required, not a shortened table")
    groups = []
    for start in range(0, len(rows), 4):
        group = rows[start : start + 4]
        first = group[0]
        if first["group_type"] not in {"logic", "relevance"} or any(
            any(
                row[key] != first[key]
                for key in ("video_id", "group_type", "group_structure")
            )
            for row in group
        ):
            raise ValueError("original complete group relation differs")
        if [row["question_id"] for row in group] != [
            first["video_id"] + "-" + str(i) for i in range(1, 5)
        ]:
            raise ValueError("original four-question order differs")
        groups.append((start, first["video_id"], first["group_type"]))
    if len({item[1] for item in groups}) != 800:
        raise ValueError("all800 original unique groups required")
    selected = [
        item
        for kind in ("logic", "relevance")
        for item in [group for group in groups if group[2] == kind][:3]
    ]
    if [item[1] for item in selected] != expected_videos:
        raise ValueError("frozen first3 groups per type differ; never replace a task")
    return [
        position for start, _, _ in selected for position in range(start, start + 4)
    ]


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def check_prompt(prompt, original, owned_root=None):
    images = [item["value"] for item in prompt if item["type"] == "image"]
    texts = [item["value"] for item in prompt if item["type"] == "text"]
    if (
        len(images) != 64
        or len(set(images)) != 64
        or not texts
        or original["question"] not in texts[-1]
        or original["options"] not in texts[-1]
    ):
        raise ValueError(
            "original64 frames and unchanged original question/options required"
        )
    if owned_root is not None and any(
        not Path(path).resolve(strict=True).is_relative_to(owned_root)
        for path in images
    ):
        raise ValueError("original frame cache escaped the owned preparation root")
    return images


def verify_original_tsv(path, rows):
    """Verify every generated field against pinned Parquet, not an unknown TSV."""
    columns = [
        "index",
        "video",
        "video_path",
        "subtitle_path",
        "url",
        "group_type",
        "group_structure",
        "question_id",
        "question",
        "options",
        "answer",
        "level",
        "second_head",
        "third_head",
    ]
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != columns:
            raise ValueError("original generated TSV columns changed")
        generated = list(reader)
    if len(rows) != 3200 or len(generated) != len(rows):
        raise ValueError("original generated TSV must retain all3200 rows")
    for index, (original, actual) in enumerate(zip(rows, generated, strict=True)):
        expected = {key: original[key] for key in columns if key in original}
        video = original["video_id"]
        expected.update(
            index=str(index),
            video=video,
            video_path=f"./video/{video}.mp4",
            subtitle_path=f"./subtitle/{video}.jsonl",
        )
        if actual != expected:
            raise ValueError(f"original generated TSV fields differ at row{index}")
    return len(generated)


def prepare(args):
    if "huggingface_hub" in sys.modules:
        raise ValueError("fresh standalone process required for owned offline cache")
    source = args.official_checkout.resolve(strict=True)
    data = args.data_root.resolve(strict=True)
    media = args.media_root.resolve(strict=True)
    output = args.output.resolve()
    if (
        output.exists()
        or output.is_relative_to(ROOT)
        or any(
            output.is_relative_to(path) or path.is_relative_to(output)
            for path in (source, data, media)
        )
    ):
        raise ValueError(
            "fresh external output separate from original inputs/source required"
        )
    proof = verify(source, archive=True)
    profile = json.loads(PROFILE.read_text("utf-8"))
    from vllm_hust_workload_data.assets import iter_rows, verify_file
    from vllm_hust_workload_data.registry import source_spec
    from vllm_hust_workload_data.videomme_v2 import group_rows

    manifest = source_spec("VIDEOMME_V2")
    if manifest["revision"] != profile["dataset_revision"]:
        raise ValueError("original data/profile revision differs")
    for item in manifest["files"]:
        verify_file(data, item)
    rows = list(iter_rows(data / "test.parquet"))
    if len(list(group_rows(rows))) != 800:
        raise ValueError("original complete annotation groups required")
    positions = fixed_positions(rows, profile["selected_video_ids"])
    saved = json.loads((media / "media-public.json").read_bytes())
    if (
        saved["archive_sha256"] != profile["archive_sha256"]
        or [item["video_id"] for item in saved["files"]]
        != profile["selected_video_ids"]
    ):
        raise ValueError("verified original selected media binding required")
    for item in saved["files"]:
        path = (media / item["path"]).resolve(strict=True)
        if (
            path.parent != media
            or path.name != item["video_id"] + ".mp4"
            or path.stat().st_size != item["bytes"]
            or sha256(path) != item["observed_sha256"]
        ):
            raise ValueError("original selected video identity differs")
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    hub = output / "hf/hub"
    repository = hub / "datasets--MME-Benchmarks--Video-MME-v2"
    snapshot = repository / "snapshots" / manifest["revision"]
    snapshot.mkdir(parents=True)
    for item in manifest["files"]:
        destination = snapshot / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(data / item["path"], destination)
    (snapshot / "video").mkdir()
    for item in saved["files"]:
        os.link(media / item["path"], snapshot / "video" / (item["video_id"] + ".mp4"))
    (repository / "refs").mkdir()
    (repository / "refs/main").write_text(manifest["revision"], encoding="utf-8")
    # LMUDataRoot ignores a supplied directory unless it already exists.
    (output / "native-cache").mkdir(mode=0o700)
    os.environ.update(
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        HF_DATASETS_OFFLINE="1",
        HF_HOME=str(output / "hf"),
        HF_HUB_CACHE=str(hub),
        HUGGINGFACE_HUB_CACHE=str(hub),
        LMUData=str(output / "native-cache"),
        TORCH_DEVICE_BACKEND_AUTOLOAD="0",
        CUDA_VISIBLE_DEVICES="",
    )
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    from vlmeval.dataset.videommev2 import VideoMMEv2
    import torch

    if hasattr(torch, "npu") and torch.npu.is_initialized():
        raise RuntimeError("CPU preparation must not initialize an NPU context")

    dataset = VideoMMEv2(
        dataset="Video-MME-v2",
        nframe=64,
        fps=-1,
        with_subtitle=False,
        subtitle_interleave=False,
        reasoning=False,
        resize_target_area=False,
    )
    if (
        Path(dataset.data_root).resolve() != snapshot
        or len(dataset.data) != 3200
        or dataset.nframe != 64
    ):
        raise ValueError("original toolkit selected different data or frame settings")
    native_tsv = Path(dataset.data_file)
    verified_tsv_rows = verify_original_tsv(native_tsv, rows)
    generated_md5 = hashlib.md5(
        native_tsv.read_bytes(), usedforsecurity=False
    ).hexdigest()
    records = []
    for position in positions:
        row = dataset.data.iloc[position]
        original = rows[position]
        if int(row["index"]) != position or any(
            row[key] != original[key]
            for key in (
                "question_id",
                "question",
                "options",
                "answer",
                "group_type",
                "group_structure",
            )
        ):
            raise ValueError("native original question/reference/order changed")
        prompt = dataset.build_prompt(position, video_llm=False)
        images = check_prompt(prompt, original, owned_root=output)
        records.append(
            {
                "source_index": position,
                "original_question_id": original["question_id"],
                "video_id": original["video_id"],
                "original_prompt": prompt,
                "frame_sha256": [sha256(Path(path)) for path in images],
            }
        )
        (output / "progress-public.json").write_text(
            json.dumps(
                {
                    "prepared_original_questions": len(records),
                    "total": 24,
                    "model_calls": 0,
                }
            ),
            encoding="utf-8",
        )
    target = output / "original-prompts-private.json"
    distinct_frames = {
        item["value"]
        for record in records
        for item in record["original_prompt"]
        if item["type"] == "image"
    }
    if len(distinct_frames) != 6 * 64:
        raise ValueError("six original videos must retain384 distinct cached frames")
    target.write_text(json.dumps(records, indent=2), encoding="utf-8")
    verify(source, archive=True)
    if hasattr(torch, "npu") and torch.npu.is_initialized():
        raise RuntimeError("CPU preparation initialized an NPU context")
    report = {
        "source_id": "VIDEOMME_V2",
        "status": "ORIGINAL_FIXED24_64FRAME_PROMPTS_PREPARED_NOT_NATIVE_INFERENCE",
        "profile": "Video-MME-v2_64frame",
        "selected_original_positions": positions,
        "selected_video_ids": profile["selected_video_ids"],
        "prepared_original_questions": len(records),
        "frames_each": 64,
        "distinct_frame_files": len(
            {
                item["value"]
                for record in records
                for item in record["original_prompt"]
                if item["type"] == "image"
            }
        ),
        "full_annotation_rows_retained": len(dataset.data),
        "native_tsv_matches_pinned_original_md5": generated_md5 == dataset.MD5,
        "native_tsv_generated_md5": generated_md5,
        "native_tsv_author_cache_md5": dataset.MD5,
        "native_tsv_original_fields_verified_rows": verified_tsv_rows,
        "native_tsv_identity_basis": "All14 columns and all3200 rows exactly match pinned Parquet plus unchanged original constructor's deterministic path/index mapping. Author cache MD5 is recorded separately; upstream constructor itself permits regenerating a different TSV. No upstream code or TSV rewritten.",
        "original_constructor_sampler_and_prompt_builder_used": True,
        "with_subtitle": False,
        "reasoning": False,
        "resize_target_area": False,
        "offline_cache_media_scope": "Only frozen6 original videos, not all800; all3200 original annotations retained",
        "source_proof": proof,
        "private_prompts_sha256": sha256(target),
        "model_calls": 0,
        "npu_context_initialized": False,
        "source_checkbox_eligible": False,
    }
    (output / "qualification-public.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("official-checkout", "data-root", "media-root", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "source_id": "VIDEOMME_V2",
                    "status": "PLAN_ONLY",
                    "frames_each": 64,
                    "selected_questions": 24,
                    "model_calls": 0,
                }
            )
        )
        return
    if any(
        getattr(args, name) is None
        for name in ("official_checkout", "data_root", "media_root", "output")
    ):
        parser.error(
            "explicit original source/data/media and fresh external output required"
        )
    os.umask(0o077)
    print(json.dumps(prepare(args)))


if __name__ == "__main__":
    main()
