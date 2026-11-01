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


def npu_program(raw, model_root, device):
    """Bind only original checkpoint transport, device placement and attention."""
    if not re.fullmatch(r"npu:\d+", device):
        raise ValueError("explicit NPU device required")
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
                    keyword.value = ast.Constant("eager")
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
    program = npu_program(raw, args.model_root, args.device)
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
        "attention_binding": "eager replaces CUDA FlashAttention2",
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
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    subset = output / "original-subset.private.json"
    subset.write_text(json.dumps(selected), "utf-8")
    answers = output / "answers.private.json"
    videos = output / "videos"
    videos.mkdir()
    for name in names:
        (videos / name).symlink_to(asset_path(args.media_root, name))
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--execute", action="store_true")
    print(json.dumps(run(parser.parse_args()), indent=2))
