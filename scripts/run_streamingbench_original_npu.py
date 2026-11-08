"""Original video-only offline loops with explicit published Qwen2-VL NPU binding.

Default is a plan. Original CUDA/FlashAttention2 placement is adapted, not the
questions, time windows, adaptive FPS, prompts, 128-token budget or scorer.
Audio/omni and streaming interfaces are not implemented by this driver.
"""

import argparse
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
from types import ModuleType

from prepare_jsonschemabench_author_source import verify_source
from run_ovobench_original_npu import MODEL_PINS, verify_model
from score_streamingbench_original import PROFILE, score
from vllm_hust_workload_data.assets import asset_path, verify_file
from vllm_hust_workload_data.registry import source_spec


RESULT_KEY = "Qwen2-VL-NPU"
CLASSES = {
    "real": "StreamingBench",
    "sqa": "StreamingBenchSQA",
    "proactive": "StreamingBenchProactive",
}


def local_video_messages(messages):
    """Normalize author local-file URI for Decord, without altering model text."""
    normalized = deepcopy(messages)
    for message in normalized:
        for part in message.get("content", []):
            if isinstance(part, dict) and part.get("type") == "video":
                path = part.get("video")
                if isinstance(path, str) and path.startswith("file://"):
                    local = path[len("file://") :]
                    if not local or not Path(local).is_file():
                        raise ValueError(
                            "author local video URI must resolve to a file"
                        )
                    part["video"] = local
    return normalized


def groups(rows):
    for row in rows:
        yield from row if isinstance(row, list) else [row]


def media_names(rows):
    names = set()
    for group in groups(rows):
        path = PurePosixPath(group["video_path"])
        if len(path.parts) != 2 or path.parts[0] != "videos" or "\\" in str(path):
            raise ValueError("original videos/basename reference required")
        if path.suffix != ".mp4":
            raise ValueError("original MP4 reference required")
        names.add(path.name)
        if any(RESULT_KEY in q for q in group["questions"]):
            raise ValueError(
                "local result key already present; do not silently skip tasks"
            )
    return sorted(names)


def npu_program(raw, model_root, device, attention_backend="sdpa"):
    """Bind only original checkpoint transport, device placement and attention."""
    if not re.fullmatch(r"npu:\d+", device):
        raise ValueError("explicit NPU device required")
    if attention_backend not in {"sdpa", "eager"}:
        raise ValueError("explicit supported NPU attention backend required")
    tree = ast.parse(raw)
    counts = {"checkpoint": 0, "device": 0, "attention": 0, "placement": 0}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr == "from_pretrained":
            if (
                not node.args
                or not isinstance(node.args[0], ast.Constant)
                or node.args[0].value != "Qwen/Qwen2-VL-7B-Instruct"
            ):
                raise ValueError("unexpected original checkpoint binding")
            node.args[0] = ast.Constant(str(Path(model_root).resolve()))
            node.keywords.append(
                ast.keyword(arg="local_files_only", value=ast.Constant(True))
            )
            counts["checkpoint"] += 1
            for keyword in list(node.keywords):
                if keyword.arg == "attn_implementation":
                    if ast.literal_eval(keyword.value) != "flash_attention_2":
                        raise ValueError("unexpected original attention binding")
                    keyword.value = ast.Constant(attention_backend)
                    counts["attention"] += 1
                if keyword.arg == "device_map":
                    if ast.literal_eval(keyword.value) != "auto":
                        raise ValueError("unexpected original placement binding")
                    # Match the already qualified OVO loader: load local weights,
                    # then explicitly move to NPU before any neural forward.
                    # Do not introduce accelerate or permit automatic CPU offload.
                    node.keywords.remove(keyword)
                    counts["placement"] += 1
        elif (
            node.func.attr == "to"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "cuda"
        ):
            node.args[0] = ast.Constant(device)
            counts["device"] += 1
    if counts != {"checkpoint": 2, "device": 1, "attention": 1, "placement": 1}:
        raise ValueError(
            "original Qwen2VL binding structure differs; inspect instead of guessing"
        )
    return compile(
        ast.fix_missing_locations(tree), "<pinned-streamingbench-qwen2vl-npu>", "exec"
    )


def verify_prefix_cache(path, selected, question_sha, registered, author):
    """Verify locally prepared clips for the author's existing-file fast path.

    This is integrity/provenance checking, not authentication of an untrusted
    cache producer. Use only a trusted cache made by the original split_video.
    """
    path = Path(path).resolve(strict=True)
    saved = json.loads(path.read_text("utf-8"))
    utility = author / "src/utils/video_execution.py"
    if (
        saved.get("schema") != "streamingbench-original-prefix-cache/v1"
        or saved.get("source_id") != "STREAMINGBENCH"
        or saved.get("task") != "real"
        or saved.get("selection") != "first3_original_complete_outer_groups"
        or saved.get("context_time") != -1
        or saved.get("source_question_sha256") != question_sha
        or saved.get("split_video_code_sha256")
        != hashlib.sha256(utility.read_bytes()).hexdigest()
        or saved.get("directory") != "videos/tmp_60"
    ):
        raise ValueError("original fixed3 prefix-cache protocol identity differs")
    names = media_names(selected)
    if saved.get("source_media_sha256") != {
        name: registered[name]["sha256"] for name in names
    }:
        raise ValueError("prefix cache source video identity differs")
    expected = []
    for group in groups(selected):
        stem = PurePosixPath(group["video_path"]).stem
        for question in group["questions"]:
            end = sum(
                int(value) * 60**index
                for index, value in enumerate(
                    reversed(question["time_stamp"].split(":"))
                )
            )
            expected.append((f"{stem}_0_{end}.mp4", end))
    clips = saved.get("clips")
    if (
        not isinstance(clips, list)
        or len(clips) != len(expected)
        or any(not isinstance(clip, dict) for clip in clips)
    ):
        raise ValueError("complete fixed-selection prefix cache required")
    directory = path.parent / "videos/tmp_60"
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or not directory.resolve().is_relative_to(path.parent)
    ):
        raise ValueError("owned regular prefix directory required")
    for clip, (name, end) in zip(clips, expected, strict=True):
        if (
            clip.get("path") != name
            or clip.get("start_seconds") != 0
            or clip.get("end_seconds") != end
            or type(clip.get("bytes")) is not int
            or clip["bytes"] <= 0
        ):
            raise ValueError("original prefix name/time selection differs")
        target = directory / name
        if (
            target.is_symlink()
            or not target.is_file()
            or target.stat().st_size != clip.get("bytes")
        ):
            raise ValueError("original prepared prefix size/type differs")
        with target.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != clip.get("sha256"):
            raise ValueError("original prepared prefix digest differs")
    return directory, len(clips)


def run(args):
    if not re.fullmatch(r"npu:\d+", args.device):
        raise ValueError("explicit NPU device required")
    author = verify_source(args.author_root, PROFILE)
    item = next(
        x
        for x in source_spec("STREAMINGBENCH")["files"]
        if x["path"] == f"src/data/questions_{args.task}.json"
    )
    verify_file(args.data_root, item)
    original = json.loads(asset_path(args.data_root, item["path"]).read_text("utf-8"))
    selected = original if args.full else original[:3]
    if not selected or (not args.full and len(selected) != 3):
        raise ValueError("complete original task groups required")
    names = media_names(selected)
    registered = {
        x["path"]: x
        for x in json.loads(PROFILE.read_text("utf-8"))["media_archive_inventory"][
            "qualified_original_members"
        ]
    }
    missing, unregistered, snapshots = [], [], []
    for name in names:
        try:
            path = asset_path(args.media_root, name)
        except FileNotFoundError:
            missing.append(name)
            continue
        if not path.is_file():
            missing.append(name)
        elif name in registered:
            verify_file(args.media_root, registered[name])
            snapshots.append({"path": name, "registered_original_identity": True})
        else:
            unregistered.append(name)
            if args.execute and args.full:
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                snapshots.append(
                    {
                        "path": name,
                        "bytes": path.stat().st_size,
                        "observed_sha256": digest,
                        "registered_original_identity": False,
                    }
                )
    raw = (author / "src/model/Qwen2VL.py").read_bytes()
    attention = getattr(args, "attention_backend", "sdpa")
    program = npu_program(raw, args.model_root, args.device, attention)
    cache_path = getattr(args, "prefix_cache", None)
    prefix_directory, prefix_count = None, 0
    if cache_path is not None:
        if args.task != "real" or args.full:
            raise ValueError(
                "prefix reuse currently requires complete fixed3 real mode"
            )
        prefix_directory, prefix_count = verify_prefix_cache(
            cache_path, selected, item["sha256"], registered, author
        )
    pins = json.loads(MODEL_PINS.read_text("utf-8"))
    if args.model_revision != pins["revision"]:
        raise ValueError("fixed original Qwen2-VL model revision required")
    receipt = {
        "source_id": "STREAMINGBENCH",
        "task": args.task,
        "source_question_sha256": item["sha256"],
        "original_qwen_wrapper_sha256": hashlib.sha256(raw).hexdigest(),
        "model": "Qwen/Qwen2-VL-7B-Instruct",
        "model_revision": args.model_revision,
        "status": "PLAN_ONLY",
        "full_selection": args.full,
        "selected_complete_outer_groups": len(selected),
        "original_question_occurrences": sum(
            len(x["questions"]) for x in groups(selected)
        ),
        "context_time": -1,
        "original_max_new_tokens": 128,
        "original_adaptive_fps": [1.0, 0.5, 0.2],
        "fps_values_are_publisher_branches_not_observed_sampling": True,
        "publisher_fps_suffix_parsing_limitation_preserved": True,
        "original_max_pixels": 151200,
        "missing_original_media": missing,
        "unregistered_media": unregistered,
        "media_snapshot": snapshots,
        "device": args.device,
        "attention_binding": attention + " replaces CUDA FlashAttention2",
        "verified_original_cached_prefixes": prefix_count,
        "author_existing_prefix_branch_reused": prefix_directory is not None,
        "explicit_published_qwen_class_not_author_cli_fallback": True,
        "model_result_key": RESULT_KEY,
        "native_model_calls": 0,
        "native_execution_complete": False,
        "model_artifacts_verified": False,
        "original_scoring_complete": False,
        "source_checkbox_eligible": False,
        "paper_hardware_or_quality_reproduction": False,
    }
    if not args.execute:
        return receipt
    if missing or (unregistered and not args.full):
        raise ValueError(
            "complete registered fixed-selection media required before NPU load"
        )
    verify_model(args.model_root, args.model_revision, pins)
    receipt["model_artifacts_verified"] = True
    output = args.output.resolve()
    for protected in (
        author.resolve(),
        args.data_root.resolve(),
        args.media_root.resolve(),
        args.model_root.resolve(),
        PROFILE.parents[2].resolve(),
    ):
        if (
            output == protected
            or output.is_relative_to(protected)
            or protected.is_relative_to(output)
        ):
            raise ValueError(
                "fresh external output separate from sources/media/model required"
            )
    if prefix_directory is not None and (
        output.is_relative_to(prefix_directory.parent.parent)
        or prefix_directory.parent.parent.is_relative_to(output)
    ):
        raise ValueError("fresh native output must be separate from prefix cache")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    subset = output / "original-subset.private.json"
    subset.write_text(json.dumps(selected), "utf-8")
    answers = output / "answers.private.json"
    videos = output / "videos"
    videos.mkdir()
    for name in names:
        (videos / name).symlink_to(asset_path(args.media_root, name))
    if prefix_directory is not None:
        (videos / "tmp_60").symlink_to(prefix_directory, target_is_directory=True)
    previous_cwd, previous_path = Path.cwd(), list(sys.path)
    try:
        import torch
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            raise RuntimeError("available NPU required; no CPU/CUDA fallback")
        torch.npu.set_device(args.device)
        if any(name in sys.modules for name in ("model", "benchmark", "utils")):
            raise RuntimeError(
                "fresh isolated native CLI required; source modules already loaded"
            )
        sys.path.insert(0, str(author / "src"))
        os.chdir(output)
        with (output / "native-stdout.private.log").open("x", encoding="utf-8") as log:
            with redirect_stdout(log):
                module = ModuleType("explicit_streamingbench_qwen2vl")
                exec(program, module.__dict__)
                original_vision_info = module.process_vision_info
                module.process_vision_info = lambda messages: original_vision_info(
                    local_video_messages(messages)
                )
                model = module.Qwen2VL()
                module.model = module.model.to(args.device)
                if any(
                    parameter.device.type != "npu"
                    for parameter in module.model.parameters()
                ):
                    raise RuntimeError("all neural model weights must reside on NPU")
                model.name = lambda: RESULT_KEY
                original_run = model.Run

                def tracked_run(file, prompt):
                    receipt["native_model_calls"] += 1
                    return original_run(file, prompt)

                model.Run = tracked_run
                native = importlib.import_module("benchmark." + CLASSES[args.task])
                getattr(native, CLASSES[args.task])(selected).eval(
                    deepcopy(selected), model, str(answers), -1
                )
        scored = score(
            author_root=author,
            data_root=args.data_root,
            original_subset=subset,
            answers_path=answers,
            model=RESULT_KEY,
            task=args.task,
            output=output / "original-score",
        )
        if prefix_directory is not None:
            verify_prefix_cache(
                cache_path, selected, item["sha256"], registered, author
            )
        receipt.update(
            status="ORIGINAL_VIDEO_PROTOCOL_EXECUTED_AND_SCORED",
            native_execution_complete=True,
            original_scoring_complete=scored["exit_code"] == 0,
        )
    except BaseException as error:
        receipt.update(
            status="FAILED_RETAINED_NO_RETRY", error_type=type(error).__name__
        )
        raise
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_path
        (output / "receipt-public.json").write_text(
            json.dumps(receipt, indent=2), "utf-8"
        )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--task", choices=CLASSES, default="real")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--attention-backend", choices=["sdpa", "eager"], default="sdpa"
    )
    parser.add_argument(
        "--prefix-cache",
        type=Path,
        help="Trusted local original split_video cache manifest; fixed3 real mode only",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--execute", action="store_true")
    print(json.dumps(run(parser.parse_args()), indent=2))
