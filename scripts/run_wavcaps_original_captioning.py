"""Prepare or run the pinned author's single-audio example; plan is offline.

No prompts, references, model weights or inference libraries are fetched.
Execution requires the separately prepared, declared NPU environment.
"""

import argparse
import ast
import contextlib
import errno
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys

from prepare_jsonschemabench_author_source import verify_source
from verify_dailytalk_original_models import verify_file
from verify_wavcaps_original_checkpoint import PROFILE as ASSETS

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/workload-native/wavcaps-original-captioning-protocol.json"
SOURCES = ROOT / "configs/workload-native/wavcaps-author-source.json"
COMPONENTS = {
    "SoundBible": "soundbible",
    "BBC_Sound_Effects": "bbc",
    "AudioSet_SL": "audioset",
    "FreeSound": "freesound",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def link_asset(source, target):
    try:
        os.link(source, target)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        target.symlink_to(source)


def bind_paths(raw, checkpoint, audio):
    tree = ast.parse(raw)
    found = []
    for node in tree.body:
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            continue
        name = node.targets[0].id
        if name not in {"checkpoint_path", "audio_path"}:
            continue
        if not isinstance(node.value, ast.Constant) or node.value.value != "":
            raise ValueError("original deployment placeholder differs")
        node.value = ast.Constant(
            str(checkpoint if name == "checkpoint_path" else audio)
        )
        found.append(name)
    if sorted(found) != ["audio_path", "checkpoint_path"]:
        raise ValueError("both original deployment placeholders required")
    return ast.fix_missing_locations(tree)


def validate_inputs(spec):
    author = verify_source(Path(spec["author_root"]), SOURCES)
    assets = json.loads(ASSETS.read_text("utf-8"))
    verify_file(Path(spec["checkpoint"]), assets["checkpoint"])
    verify_file(Path(spec["encoder_checkpoint"]), assets["audio_encoder"])
    for item in assets["text_decoder"]["files"]:
        verify_file(Path(spec["bart_root"]) / item["path"], item)
    component = spec["component"]
    media = json.loads(
        (
            ROOT
            / f"configs/workload-native/wavcaps-{COMPONENTS[component]}-original-fixed3-media.json"
        ).read_text("utf-8")
    )
    metadata = Path(spec["metadata_root"]) / media["metadata"]["path"]
    verify_file(metadata, media["metadata"])
    rows = [
        row
        for row in json.loads(metadata.read_text("utf-8"))["data"]
        if str(row["id"]) == spec["case_id"]
    ]
    if len(rows) != 1:
        raise ValueError("one original metadata row must match case_id")
    audio = Path(spec["audio_file"])
    if not audio.is_file():
        raise ValueError("original audio file required")
    frozen = next(
        (item for item in media["files"] if item["id"] == spec["case_id"]), None
    )
    if frozen:
        verify_file(audio, frozen)
    return author, rows[0], digest(audio), frozen is not None


def prepare(spec, output):
    output = Path(output).resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("write artifacts to external server storage")
    author, row, audio_sha, frozen = validate_inputs(spec)
    output.mkdir(parents=True, exist_ok=False)
    assets = json.loads(ASSETS.read_text("utf-8"))
    workspace = output / "workspace"
    encoder = workspace / "pretrained_models/audio_encoder/HTSAT.ckpt"
    encoder.parent.mkdir(parents=True)
    link_asset(Path(spec["encoder_checkpoint"]).resolve(), encoder)
    model = output / "hf-hub/models--facebook--bart-base"
    snapshot = model / "snapshots" / assets["text_decoder"]["revision"]
    snapshot.mkdir(parents=True)
    (model / "refs").mkdir()
    (model / "refs/main").write_text(
        assets["text_decoder"]["revision"], encoding="utf-8"
    )
    for item in assets["text_decoder"]["files"]:
        link_asset(
            (Path(spec["bart_root"]) / item["path"]).resolve(), snapshot / item["path"]
        )
    case = {
        "schema": "wavcaps-original-captioning-case/v1",
        **spec,
        "author_root": str(author),
        "audio_sha256": audio_sha,
        "frozen_media_identity_verified": frozen,
        "original_row": row,
        "model_calls": 0,
    }
    (output / "case-private.json").write_text(
        json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return case


def execute(output):
    output = Path(output).resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("write artifacts to external server storage")
    if (output / "result-private.json").exists():
        raise ValueError("saved execution exists; do not rerun or overwrite it")
    case = json.loads((output / "case-private.json").read_text("utf-8"))
    if case["schema"] != "wavcaps-original-captioning-case/v1":
        raise ValueError("unsupported case manifest")
    if case["device"] not in {"npu:0", "npu:1"} or not 0 < case["memory_fraction"] <= 1:
        raise ValueError("declared NPU device and memory fraction required")
    author, row, audio_sha, _ = validate_inputs(case)
    if audio_sha != case["audio_sha256"] or row != case["original_row"]:
        raise ValueError("prepared original input changed")
    encoder_link = output / "workspace/pretrained_models/audio_encoder/HTSAT.ckpt"
    if not encoder_link.samefile(case["encoder_checkpoint"]):
        raise ValueError("prepared original encoder link changed")
    assets = json.loads(ASSETS.read_text("utf-8"))
    snapshot = (
        output
        / "hf-hub/models--facebook--bart-base/snapshots"
        / assets["text_decoder"]["revision"]
    )
    for item in assets["text_decoder"]["files"]:
        if not (snapshot / item["path"]).samefile(
            Path(case["bart_root"]) / item["path"]
        ):
            raise ValueError("prepared original BART link changed")
    protocol = json.loads(PROTOCOL.read_text("utf-8"))
    versions = protocol["verified_frozen_cohort_environment"]
    for name, version in versions.items():
        if name in {
            "precision",
            "neural_execution_device",
            "candidate_compatibility_profile_not_author_lock",
        }:
            continue
        if importlib.metadata.version(name) != version:
            raise ValueError(f"declared environment differs: {name}")
    os.environ.update(
        HF_HUB_CACHE=str(output / "hf-hub"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TORCH_FORCE_WEIGHTS_ONLY_LOAD="1",
    )
    sys.dont_write_bytecode = True
    import numpy as np
    import torch
    import torch_npu  # noqa: F401 - register the NPU backend

    sys.path.insert(0, str(author / "captioning"))
    from models import bart_captioning

    torch.npu.set_device(case["device"])
    torch.npu.set_per_process_memory_fraction(case["memory_fraction"], case["device"])
    original_load = torch.load
    allowed = {
        Path(case[key]).resolve() for key in ["checkpoint", "encoder_checkpoint"]
    }
    allowed.add(encoder_link.resolve())
    result = {
        "schema": "wavcaps-original-captioning-result/v1",
        "component": case["component"],
        "original_id": case["case_id"],
        "audio_sha256": audio_sha,
        "status": "RUNNING",
        "native_forward_started": False,
        "source_checkbox_eligible": False,
    }

    def safe_load(path, *args, **kwargs):
        if Path(path).resolve() not in allowed:
            raise ValueError("undeclared checkpoint load")
        names = torch.serialization.get_unsafe_globals_in_checkpoint(path)
        if not set(names) <= {"numpy.core.multiarray.scalar", "numpy.dtype"}:
            raise ValueError("undeclared checkpoint globals")
        kwargs.update(weights_only=True, map_location="cpu")
        value = original_load(path, *args, **kwargs)
        if Path(path).resolve() == Path(case["checkpoint"]).resolve():
            value["config"] = dict(value["config"], device=case["device"])
        return value

    def check(value):
        if isinstance(value, torch.Tensor) and value.device.type != "npu":
            raise ValueError("CPU neural input forbidden")
        if isinstance(value, (list, tuple, dict)):
            for item in value.values() if isinstance(value, dict) else value:
                check(item)

    def before(module, args, kwargs):
        if any(
            x.device.type != "npu"
            for x in [*module.parameters(recurse=False), *module.buffers(recurse=False)]
        ):
            raise ValueError("CPU neural parameter/buffer forbidden")
        check(args)
        check(kwargs)
        result["native_forward_started"] = True

    original_init = bart_captioning.BartCaptionModel.__init__

    def guarded_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        for module in self.modules():
            module.register_forward_pre_hook(before, with_kwargs=True)

    raw = (author / "captioning/test.py").read_bytes()
    program = bind_paths(raw, case["checkpoint"], case["audio_file"])
    scope = {"__name__": "__main__", "__file__": str(author / "captioning/test.py")}
    stdout = io.StringIO()
    previous_cwd = Path.cwd()
    try:
        torch.load = safe_load
        bart_captioning.BartCaptionModel.__init__ = guarded_init
        os.chdir(output / "workspace")
        with (
            torch.serialization.safe_globals(
                [np.core.multiarray.scalar, np.dtype, np.dtypes.Float64DType]
            ),
            contextlib.redirect_stdout(stdout),
        ):
            exec(compile(program, scope["__file__"], "exec"), scope)
        torch.npu.synchronize()
        verify_source(author, SOURCES)
        if digest(case["audio_file"]) != audio_sha:
            raise ValueError("original input changed during execution")
        if any(
            parameter.dtype != torch.float32
            for parameter in scope["model"].parameters()
        ):
            raise ValueError("original float32 model precision changed")
        caption = scope["caption"]
        if (
            not isinstance(caption, list)
            or len(caption) != 1
            or not isinstance(caption[0], str)
        ):
            raise ValueError("original single-audio caption structure differs")
        result.update(
            status="ORIGINAL_CAPTION_GENERATED",
            caption=caption,
            precision="original_float32",
            device=case["device"],
            original_generation_defaults=protocol["publisher_generation_defaults"],
            original_source_unchanged=True,
            official_quality_score=None,
        )
    except BaseException as error:
        result.update(
            status="FAILED_RETAINED",
            error_type=type(error).__name__,
            error=str(error)[:500],
        )
        raise
    finally:
        result["model_calls"] = int(result["native_forward_started"])
        torch.load = original_load
        bart_captioning.BartCaptionModel.__init__ = original_init
        os.chdir(previous_cwd)
        (output / "original-stdout-private.txt").write_text(
            stdout.getvalue(), encoding="utf-8"
        )
        (output / "result-private.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action", choices=["plan", "prepare", "execute"], default="plan"
    )
    parser.add_argument("--output-root", type=Path)
    for name in [
        "author-root",
        "metadata-root",
        "checkpoint",
        "encoder-checkpoint",
        "bart-root",
        "audio-file",
    ]:
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--component", choices=COMPONENTS)
    parser.add_argument("--case-id")
    parser.add_argument("--device", choices=["npu:0", "npu:1"], default="npu:0")
    parser.add_argument("--memory-fraction", type=float, default=0.10)
    args = parser.parse_args()
    if args.action == "plan":
        print(
            json.dumps(
                {
                    "source_id": "WAVCAPS",
                    "status": "PLAN_ONLY_NO_MODEL_LOAD",
                    "protocol": PROTOCOL.relative_to(ROOT).as_posix(),
                    "model_calls": 0,
                }
            )
        )
        return
    if args.output_root is None:
        parser.error("--output-root is required")
    if args.action == "execute":
        result = execute(args.output_root)
    else:
        names = [
            "author_root",
            "metadata_root",
            "checkpoint",
            "encoder_checkpoint",
            "bart_root",
            "audio_file",
            "component",
            "case_id",
        ]
        if (
            any(getattr(args, name) is None for name in names)
            or not 0 < args.memory_fraction <= 1
        ):
            parser.error(
                "all input paths, component, case-id and a valid memory fraction are required"
            )
        spec = {
            name: str(getattr(args, name).resolve())
            if isinstance(getattr(args, name), Path)
            else getattr(args, name)
            for name in names
        }
        spec.update(device=args.device, memory_fraction=args.memory_fraction)
        prepare(spec, args.output_root)
        result = {
            "source_id": "WAVCAPS",
            "status": "PREPARED_NOT_EXECUTED",
            "model_calls": 0,
        }
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "caption"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
