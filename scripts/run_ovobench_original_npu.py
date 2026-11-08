"""Original OVO offline evaluator with an explicit Qwen2-VL NPU binding.

Original prompts, task loops, video limits, output budget and scorer are reused.
Device placement and eager attention replace the CUDA/FlashAttention2 binding;
this is not a reproduction of the paper's hardware or performance measurements.
"""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

from vllm_hust_workload_data.assets import asset_path

from prepare_jsonschemabench_author_source import verify_source
from prepare_ovobench_native_selection import ROOT, fixed_selection, prepare
from verify_ovobench_original_media import PROFILE, verify

MODEL_PINS = ROOT / "configs/workload-native/ovobench-qwen2-vl-7b-model-pins.json"


def scorer_text(value):
    """Losslessly unwrap one decoded string, never extract or rewrite an answer."""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
        return value[0]
    raise ValueError("one decoded text required; no choosing or joining candidates")


def verify_model(root, revision, pins=None):
    pins = pins or json.loads(MODEL_PINS.read_text("utf-8"))
    if revision != pins["revision"] or pins["model"] != "Qwen/Qwen2-VL-7B-Instruct":
        raise ValueError("fixed original model identity required")
    for item in pins["files"]:
        if Path(item["path"]).name != item["path"]:
            raise ValueError("contained model file required")
        path = Path(root) / item["path"]
        if path.stat().st_size != item["bytes"]:
            raise ValueError("original model file size mismatch")
        digest = (
            hashlib.sha256()
            if item.get("lfs_sha256")
            else hashlib.sha1(b"blob " + str(item["bytes"]).encode() + b"\0")
        )
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                digest.update(block)
        if digest.hexdigest() != item.get("lfs_sha256", item["git_blob"]):
            raise ValueError("original model Git/LFS identity mismatch")


def video_messages(path, prompt, frames):
    if type(frames) is not int or not 2 <= frames <= 64:
        raise ValueError("original bounded video frame count required")
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(path),
                    "max_pixels": 360 * 420,
                    "nframes": frames,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def group_annotations(rows):
    groups = {"backward": [], "realtime": [], "forward": []}
    for row in rows:
        if row["task"] in {"EPM", "ASI", "HLD"}:
            groups["backward"].append(row)
        elif row["task"] in {"STU", "OJR", "ATR", "ACR", "OCR", "FPD"}:
            groups["realtime"].append(row)
        elif row["task"] in {"REC", "SSR", "CRR"}:
            groups["forward"].append(row)
        else:
            raise ValueError("unknown original OVO task")
    return groups


def media_input_snapshot(selection, registered, root, *, full=False):
    """Verify registered originals; freeze extra full-run inputs without inventing publisher hashes."""
    if not full and not registered["all_required_clips_registered"]:
        raise ValueError(
            "all original fixed-selection media required before model load"
        )
    known = {item["path"]: item for item in registered["files"]}
    snapshots = []
    for clip in selection["required_native_clips"]:
        name = clip["path"]
        if name in known:
            snapshots.append({**known[name], "registered_original_identity": True})
            continue
        path = asset_path(root, name)
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"complete original full-run media missing: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                digest.update(block)
        snapshots.append(
            {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
                "registered_original_identity": False,
            }
        )
    return snapshots


def run(args):
    if not re.fullmatch(r"npu:\d+", args.device):
        raise ValueError("explicit NPU device required")
    author = verify_source(args.author_root, PROFILE)
    selection = prepare(args.data_root, full=args.full)
    pins = json.loads(PROFILE.read_text("utf-8"))
    media = verify(
        pins["media_archive_inventory"]["qualified_original_members"],
        selection,
        args.media_root,
    )
    media_snapshot = media_input_snapshot(
        selection, media, args.media_root, full=args.full
    )
    verify_model(args.model_root, args.model_revision)
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("native data and outputs must stay outside the repository")
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    (output / "media-input-identity-private.json").write_text(
        json.dumps(media_snapshot, indent=2), "utf-8"
    )
    receipt = {
        "status": "RUNNING_NOT_COMPLETE",
        "model": "Qwen/Qwen2-VL-7B-Instruct",
        "model_revision": args.model_revision,
        "attention_backend": "eager",
        "device": args.device,
        "original_task_cases": selection["selected_original_cases"],
        "required_original_response_slots": selection["required_native_clip_count"],
        "full_selection": args.full,
        "registered_original_media_count": len(media["files"]),
        "extra_media_hashes_are_observed_not_publisher_identities": args.full,
        "attempted_slots": 0,
        "successful_slots": 0,
        "inference_errors": [],
        "native_execution_complete": False,
        "paper_hardware_replication": False,
        "performance_claim": False,
        "source_checkbox_eligible": False,
    }

    def save():
        (output / "receipt-private.json").write_text(
            json.dumps(receipt, indent=2), "utf-8"
        )

    save()
    # The caller must acquire its device window. No automatic server/service stops.
    try:
        import torch
        import torch_npu  # noqa: F401
        from decord import VideoReader
        from qwen_vl_utils import process_vision_info
        from qwen_vl_utils.vision_process import get_video_reader_backend
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        if not args.device.startswith("npu:") or not torch.npu.is_available():
            raise RuntimeError("explicit available NPU required; no CPU/CUDA fallback")
        if get_video_reader_backend() != "decord":
            raise RuntimeError(
                "qualified Decord reader required; no silent reader fallback"
            )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_root,
            torch_dtype="auto",
            attn_implementation="eager",
            local_files_only=True,
        ).to(args.device)
        model.eval()
        if any(p.device.type != "npu" for p in model.parameters()):
            raise RuntimeError("all neural model parameters must reside on NPU")
        processor = AutoProcessor.from_pretrained(
            args.model_root, local_files_only=True
        )
        receipt["model_dtype"] = str(model.dtype)
        sys.path.insert(0, str(author))
        module = importlib.import_module("utils.OVOBench")
        if Path(module.__file__).resolve() != author / "utils/OVOBench.py":
            raise RuntimeError("original evaluator import resolved to another module")
        base = module.OVOBenchOffline

        class BoundOriginal(base):
            def inference(self, video_file_name, prompt):
                slot = receipt["attempted_slots"]
                receipt["attempted_slots"] += 1
                save()
                request = {"clip": Path(video_file_name).name, "prompt": prompt}
                path = output / f"slot-{slot}.request-private.json"
                path.write_text(json.dumps(request, ensure_ascii=False), "utf-8")
                try:
                    frames = min(64, len(VideoReader(video_file_name)) - 2)
                    messages = video_messages(video_file_name, prompt, frames)
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    images, videos = process_vision_info(messages)
                    inputs = processor(
                        text=[text],
                        images=images,
                        videos=videos,
                        padding=True,
                        return_tensors="pt",
                    ).to(args.device)
                    grid = inputs.video_grid_thw.tolist()
                    decoded_frames = int(videos[0].shape[0])
                    temporal = model.config.vision_config.temporal_patch_size
                    if len(grid) != 1 or grid[0][0] * temporal != decoded_frames:
                        raise RuntimeError(
                            "model processor changed the original sampled video frame count"
                        )
                    with torch.inference_mode():
                        generated = model.generate(**inputs, max_new_tokens=128)
                    trimmed = [
                        row[len(original) :]
                        for original, row in zip(inputs.input_ids, generated)
                    ]
                    result = processor.batch_decode(
                        trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    (output / f"slot-{slot}.response-private.json").write_text(
                        json.dumps(
                            {
                                "response": result,
                                "generated_tokens": [len(x) for x in trimmed],
                                "frames": frames,
                                "request_sha256": hashlib.sha256(
                                    path.read_bytes()
                                ).hexdigest(),
                            },
                            ensure_ascii=False,
                        ),
                        "utf-8",
                    )
                    receipt["successful_slots"] += 1
                    return scorer_text(result)
                except Exception as error:
                    receipt["inference_errors"].append(
                        {
                            "slot": slot,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
                    raise
                finally:
                    save()

        original_rows = json.loads(
            (args.data_root / "data/ovo_bench_new.json").read_text("utf-8")
        )
        selected, _ = fixed_selection(original_rows, full=args.full)
        tasks = [
            "EPM",
            "ASI",
            "HLD",
            "STU",
            "OJR",
            "ATR",
            "ACR",
            "OCR",
            "FPD",
            "REC",
            "SSR",
            "CRR",
        ]
        config = SimpleNamespace(
            chunked_dir=str(args.media_root),
            save_results=True,
            result_dir=str(output / "results"),
            model="QWen2VL_7B",
        )
        BoundOriginal(config).eval(group_annotations(selected), tasks, "offline")
        command = [
            sys.executable,
            str(author / "score.py"),
            "--result_dir",
            config.result_dir,
            "--model",
            config.model,
            "--mode",
            "offline",
        ]
        scored = subprocess.run(
            command, cwd=author, capture_output=True, text=True, timeout=120
        )
        (output / "official-score-private.log").write_text(
            scored.stdout + scored.stderr, "utf-8"
        )
        receipt["official_score_exit_code"] = scored.returncode
        complete = (
            receipt["successful_slots"]
            == receipt["attempted_slots"]
            == selection["required_native_clip_count"]
            and scored.returncode == 0
        )
        receipt.update(
            status="FIXED_ORIGINAL_OFFLINE_PROTOCOL_COMPLETE"
            if complete
            else "INCOMPLETE_RETAINED_ORIGINAL_FAILURES",
            native_execution_complete=complete,
        )
        save()
        if not complete:
            raise RuntimeError(
                "native slots or official scoring incomplete; retain all failures"
            )
    except Exception as error:
        receipt.update(
            status="FAILED_NOT_COMPLETE",
            terminal_error_type=type(error).__name__,
            terminal_error=str(error),
        )
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("author-root", "data-root", "media-root", "model-root", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all original cases/stages; requires complete original precut media",
    )
    run(parser.parse_args())
