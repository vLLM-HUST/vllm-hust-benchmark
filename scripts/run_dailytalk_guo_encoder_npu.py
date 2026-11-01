"""Original Guo encode([raw_text])[0] on NPU; default offline plan only."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_dailytalk_original_guo_inputs import history_plan, select_rows
from prepare_dailytalk_original_metadata import prepare as verify_metadata
from verify_dailytalk_original_models import PROFILE, verify_file


ROOT = Path(__file__).resolve().parents[1]


def feature_path(prepared, relative):
    name = PurePosixPath(relative)
    if name.is_absolute() or ".." in name.parts or "\\" in relative:
        raise ValueError("contained original feature path required")
    expected = prepared / "preprocessed_data/DailyTalk/text_emb"
    target = prepared / "preprocessed_data/DailyTalk" / relative
    if target.parent.resolve() != expected.resolve() or target.is_symlink():
        raise ValueError(
            "original feature must stay in the prepared text_emb directory"
        )
    if not target.resolve().is_relative_to(prepared.resolve()):
        raise ValueError("feature directory escapes prepared inputs")
    return target


def load_inputs(args):
    from types import SimpleNamespace

    prepared = args.prepared.resolve(strict=True)
    verify_metadata(SimpleNamespace(data_root=args.metadata_root, action="verify"))
    contract = json.loads(
        (prepared / "guo-input-contract.private.json").read_text("utf-8")
    )
    if contract["history_window_turns"] != 10 or contract["feature_dimension"] != 512:
        raise ValueError("retain original Guo history and feature dimensions")
    selected = contract["utterances"]
    identities = {item["utterance_id"] for item in selected}
    if not selected or len(identities) != len(selected):
        raise ValueError("nonempty distinct original utterance identities required")
    original = (
        args.metadata_root / f"preprocessed_data/DailyTalk/{args.split}_phone.txt"
    )
    rows = select_rows(original.read_bytes(), identities)
    source = args.source_file or prepared / "val_phone.fixed3.private.txt"
    if source.read_bytes() != b"".join(raw for raw, _ in rows):
        raise ValueError("source rows must retain original bytes and order")
    if selected != history_plan(rows):
        raise ValueError(
            "original complete-dialogue history or feature relation differs"
        )
    for item in selected:
        feature_path(prepared, item["feature_path"])
    return rows, contract


def verify_saved(args, rows, contract):
    import numpy as np

    receipt = json.loads((args.output / "execution-private.json").read_text("utf-8"))
    profile = json.loads(PROFILE.read_text("utf-8"))
    if (
        receipt["status"] != "ORIGINAL_GUO_TEXT_FEATURES_GENERATED_ON_NPU_NOT_TTS"
        or receipt["encoder_revision"] != profile["encoder"]["revision"]
        or receipt["model_calls"] != len(rows)
        or receipt["feature_dimension"] != 512
        or receipt["max_seq_length"] != 128
        or receipt["model_dtype"] != "float32"
    ):
        raise ValueError("complete matching original NPU encoder receipt required")
    features = receipt["features"]
    if len(features) != len(rows):
        raise ValueError("every original utterance requires one saved feature")
    for saved, item in zip(features, contract["utterances"]):
        if (
            saved["utterance_id"] != item["utterance_id"]
            or saved["path"] != item["feature_path"]
        ):
            raise ValueError("saved feature identity or path differs")
        target = feature_path(args.prepared, saved["path"])
        if target.stat().st_size > 65536:
            raise ValueError(
                "original 512-dimensional feature file exceeds bounded size"
            )
        verify_file(target, saved)
        with target.open("rb") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError("unsupported original numeric feature encoding")
            if shape != (512,) or dtype != np.dtype("float32"):
                raise ValueError(
                    "original FP32 512-vector header required before mapping"
                )
            if target.stat().st_size != stream.tell() + 512 * 4:
                raise ValueError("feature payload length differs from original vector")
        vector = np.load(target, allow_pickle=False, mmap_mode="r")
        if (
            not isinstance(vector, np.ndarray)
            or vector.shape != (512,)
            or vector.dtype != np.float32
            or not np.isfinite(vector).all()
        ):
            raise ValueError("original finite FP32 512-dimensional feature required")
    return {
        "source_id": "DAILYTALK",
        "status": "SAVED_ORIGINAL_GUO_FEATURES_VERIFIED_NOT_REEXECUTED",
        "feature_files": len(features),
        "new_model_calls": 0,
        "source_checkbox_eligible": False,
    }


def run(args):
    if args.action == "plan":
        return {
            "source_id": "DAILYTALK",
            "status": "PLAN_ONLY_NO_MODEL_LOAD",
            "original_encode_rule": "encode([raw_text])[0]",
            "feature_dimension": 512,
            "model_calls": 0,
            "source_checkbox_eligible": False,
        }
    if not re.fullmatch(r"npu:\d+", args.device):
        raise ValueError("explicit NPU device required; no CPU/CUDA fallback")
    if args.action == "execute" and not args.resource_window_confirmed:
        raise ValueError("confirm an idle resource window before execute")
    rows, contract = load_inputs(args)
    profile = json.loads(PROFILE.read_text("utf-8"))
    for item in profile["encoder"]["files"]:
        verify_file(args.encoder_root / item["path"], item)
    if args.action == "verify":
        return verify_saved(args, rows, contract)
    if args.output.resolve().is_relative_to(
        ROOT
    ) or args.prepared.resolve().is_relative_to(ROOT):
        raise ValueError(
            "keep prepared data and execution output outside the repository"
        )
    if args.output.exists() or any(
        feature_path(args.prepared, item["feature_path"]).exists()
        for item in contract["utterances"]
    ):
        raise ValueError(
            "existing outputs must be verified, never blindly overwritten or regenerated"
        )
    import os

    os.umask(0o077)
    args.output.mkdir(parents=True, mode=0o700)
    destination = args.prepared / "preprocessed_data/DailyTalk/text_emb"
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    receipt = args.output / "execution-private.json"
    report = {
        "source_id": "DAILYTALK",
        "status": "RUNNING_ORIGINAL_GUO_TEXT_ENCODER_ON_NPU",
        "device": args.device,
        "original_encode_rule": "encode([raw_text])[0]",
        "encoder_revision": profile["encoder"]["revision"],
        "max_seq_length": 128,
        "feature_dimension": 512,
        "model_calls": 0,
        "model_dtype": "float32",
        "native_tts_or_scoring_executed": False,
        "source_checkbox_eligible": False,
        "features": [],
    }
    receipt.write_text(json.dumps(report, indent=2), encoding="utf-8")
    hooks = []
    try:
        import numpy as np
        import torch
        import torch_npu  # noqa: F401 - register the explicit NPU backend
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            str(args.encoder_root),
            device=args.device,
            local_files_only=True,
            model_kwargs={"attn_implementation": "eager"},
        )
        if (
            model.get_sentence_embedding_dimension() != 512
            or model.max_seq_length != 128
        ):
            raise ValueError("original encoder dimensions and sequence policy required")
        if not all(
            p.device.type == "npu" and p.dtype == torch.float32
            for p in model.parameters()
        ):
            raise ValueError("all neural parameters must retain original FP32 on NPU")
        calls = {"parameterized_module_forwards_on_npu": 0}

        def check_tensor(value):
            if isinstance(value, torch.Tensor) and value.device.type != "npu":
                raise ValueError("CPU/CUDA neural input forbidden")
            if isinstance(value, dict):
                for child in value.values():
                    check_tensor(child)
            elif isinstance(value, (tuple, list)):
                for child in value:
                    check_tensor(child)

        def check(module, arguments, keywords):
            if not all(
                p.device.type == "npu" for p in module.parameters(recurse=False)
            ):
                raise ValueError("CPU/CUDA parameterized module forbidden")
            check_tensor(arguments)
            check_tensor(keywords)
            calls["parameterized_module_forwards_on_npu"] += 1

        hooks = [
            m.register_forward_pre_hook(check, with_kwargs=True)
            for m in model.modules()
            if list(m.parameters(recurse=False))
        ]
        for (_, fields), item in zip(rows, contract["utterances"]):
            vector = model.encode([fields[3]])[0]
            if (
                vector.shape != (512,)
                or vector.dtype != np.float32
                or not np.isfinite(vector).all()
            ):
                raise ValueError("original encoder produced an invalid feature")
            target = feature_path(args.prepared, item["feature_path"])
            with target.open("xb") as stream:
                np.save(stream, vector, allow_pickle=False)
            report["model_calls"] += 1
            report["features"].append(
                {
                    "utterance_id": item["utterance_id"],
                    "path": item["feature_path"],
                    "bytes": target.stat().st_size,
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                }
            )
            receipt.write_text(json.dumps(report, indent=2), encoding="utf-8")
        torch.npu.synchronize()
        report.update(
            status="ORIGINAL_GUO_TEXT_FEATURES_GENERATED_ON_NPU_NOT_TTS", **calls
        )
        if not calls["parameterized_module_forwards_on_npu"]:
            raise ValueError("no verified parameterized NPU forward occurred")
    except BaseException as exc:
        report.update(
            status="FAILED_RETAINED_NO_FEATURE_SUBSTITUTION",
            error_type=type(exc).__name__,
        )
        raise
    finally:
        for hook in hooks:
            hook.remove()
        receipt.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {key: value for key, value in report.items() if key != "features"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action", choices=["plan", "execute", "verify"], default="plan"
    )
    for name in ["prepared", "metadata-root", "encoder-root", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-file", type=Path)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--resource-window-confirmed", action="store_true")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
