"""Run original Guo batch synthesis in its prepared workspace, NPU only."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import sys
from types import SimpleNamespace
import wave

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_dailytalk_synthesis_workspace import prepare
from run_dailytalk_guo_encoder_npu import load_inputs
from verify_dailytalk_synthesis_outputs import audit


def check_tensor_devices(value):
    import torch

    if isinstance(value, torch.Tensor):
        if value.device.type != "npu":
            raise ValueError("CPU/CUDA neural input forbidden")
    elif isinstance(value, dict):
        for item in value.values():
            check_tensor_devices(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            check_tensor_devices(item)


def run(args):
    if args.action == "plan":
        return {
            "source_id": "DAILYTALK",
            "status": "PLAN_ONLY_NO_MODEL_LOAD",
            "model_calls": 0,
            "source_checkbox_eligible": False,
        }
    if not re.fullmatch(r"npu:\d+", args.device):
        raise ValueError("explicit NPU device required")
    if args.action == "execute" and (
        not args.resource_window_confirmed
        or args.lease_file is None
        or not 0 < args.allocator_gib <= 6
    ):
        raise ValueError(
            "confirm idle resource window, shared lease file and at most 6GiB allocator budget"
        )
    prepare(SimpleNamespace(**{**vars(args), "action": "verify"}))
    if args.action == "verify":
        return audit(
            SimpleNamespace(**{**vars(args), "stage": args.result_stage or args.stage})
        )
    if args.result_stage is not None:
        raise ValueError("result-stage is saved-result verification only")
    stage = args.stage.resolve(strict=True)
    if (stage / "native-execution").exists() or (stage / "results").exists():
        raise FileExistsError(
            "retain previous native results; use a fresh prepared stage"
        )
    _, contract = load_inputs(args)
    owner = {"source_id": "DAILYTALK_ORIGINAL_GUO_SYNTHESIS", "pid": os.getpid()}
    with args.lease_file.open("x", encoding="utf-8") as stream:
        json.dump(owner, stream)
    receipt = stage / "native-execution/state-private.json"
    report = {
        "source_id": "DAILYTALK",
        "status": "RUNNING_ORIGINAL_GUO_BATCH_SYNTHESIS",
        "original_seed": 1234,
        "original_batch_size": 8,
        "original_history_window_turns": 10,
        "original_restore_step": 900000,
        "original_pitch_energy_duration_controls": [1.0, 1.0, 1.0],
        "weights_only_required": True,
        "unsafe_pickle_fallback_enabled": False,
        "native_model_forward_entries": {"acoustic": 0, "vocoder": 0},
        "human_quality_evaluation_performed": False,
        "quality_score_claimed": False,
        "explicit_fp32_gru_operator_decomposition": True,
        "source_checkbox_eligible": False,
    }
    previous_cwd, previous_argv = Path.cwd(), sys.argv[:]
    old_env = {
        key: os.environ.get(key)
        for key in ("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "DAILYTALK_NPU_DEVICE")
    }
    try:
        receipt.parent.mkdir(mode=0o700)
        receipt.write_text(json.dumps(report, indent=2), encoding="utf-8")
        import numpy as np
        import torch
        import torch_npu  # noqa: F401 - registers the NPU backend
        import matplotlib
        from dailytalk_gru_npu import install_npu_gru_forward

        if matplotlib.__version__ != "3.9.4":
            raise ValueError("use the separately pinned Matplotlib 3.9.4 overlay")
        if not torch.npu.is_available():
            raise RuntimeError("NPU unavailable; no neural fallback")
        torch.npu.set_device(args.device)
        total = torch.npu.get_device_properties(args.device).total_memory
        torch.npu.set_per_process_memory_fraction(
            args.allocator_gib * 1024**3 / total, args.device
        )
        torch.npu.matmul.allow_hf32 = False
        os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
        os.environ["DAILYTALK_NPU_DEVICE"] = args.device
        unsafe = set(
            torch.serialization.get_unsafe_globals_in_checkpoint(
                args.models_root / "900000.pth.tar"
            )
        )
        if not unsafe <= {"numpy.dtype", "numpy.core.multiarray.scalar"}:
            raise ValueError(
                "unexpected checkpoint globals; unsafe pickle fallback forbidden"
            )
        workspace = stage / "workspace"
        sys.path.insert(0, str(workspace))
        os.chdir(workspace)
        from utils import model as publisher_models

        def checked(factory, role):
            def build(*values, **kwargs):
                neural = factory(*values, **kwargs)
                if not isinstance(neural, torch.nn.Module) or any(
                    p.device.type != "npu" or p.dtype != torch.float32
                    for p in neural.parameters()
                ):
                    raise ValueError("original FP32 NPU neural parameters required")
                if role == "acoustic":
                    modules = [
                        m for m in neural.modules() if isinstance(m, torch.nn.GRU)
                    ]
                    if len(modules) != 1 or modules[0].training:
                        raise ValueError("one original inference Guo GRU required")
                    install_npu_gru_forward(modules[0])

                def before(module, arguments, named):
                    if any(p.device.type != "npu" for p in module.parameters()):
                        raise ValueError("neural parameters left NPU")
                    check_tensor_devices(arguments)
                    check_tensor_devices(named)
                    report["native_model_forward_entries"][role] += 1
                    receipt.write_text(json.dumps(report, indent=2), encoding="utf-8")

                neural.register_forward_pre_hook(before, with_kwargs=True)
                return neural

            return build

        publisher_models.get_model = checked(publisher_models.get_model, "acoustic")
        publisher_models.get_vocoder = checked(publisher_models.get_vocoder, "vocoder")
        source = args.source_file or args.prepared / "val_phone.fixed3.private.txt"
        sys.argv = [
            str(workspace / "synthesize.py"),
            "--restore_step",
            "900000",
            "--mode",
            "batch",
            "--source",
            str(source.resolve()),
            "--dataset",
            "DailyTalk",
        ]
        with torch.serialization.safe_globals(
            [np.core.multiarray.scalar, np.dtype, np.dtypes.Float64DType]
        ):
            runpy.run_path(sys.argv[0], run_name="__main__")
        torch.npu.synchronize()
        outputs = []
        for item in contract["utterances"]:
            identity = item["utterance_id"]
            relative = f"results/900000/{identity.rsplit('_d', 1)[1]}/{identity}.wav"
            wav = stage / relative
            with wave.open(str(wav), "rb") as stream:
                frames = stream.getnframes()
            outputs.append(
                {
                    "utterance_id": identity,
                    "path": relative,
                    "frames": frames,
                    "sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
                    "original_plot_path": relative[:-4] + ".png",
                }
            )
        report.update(
            status="ORIGINAL_GUO_BATCH_SYNTHESIS_RETURNED_STRUCTURAL_OUTPUT_CHECKS_PASSED",
            original_utterances=len(outputs),
            outputs=outputs,
            isolated_matplotlib_version=matplotlib.__version__,
        )
        receipt.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return audit(SimpleNamespace(**{**vars(args), "action": "verify"}))
    except BaseException as error:
        report.update(
            status="ORIGINAL_GUO_SYNTHESIS_FAILED_RETAINED",
            error_type=type(error).__name__,
        )
        if receipt.parent.exists():
            receipt.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise
    finally:
        os.chdir(previous_cwd)
        sys.argv = previous_argv
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if json.loads(args.lease_file.read_text("utf-8")) != owner:
            raise RuntimeError("shared resource lease owner changed; not removing it")
        args.lease_file.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "author-root",
        "prepared",
        "metadata-root",
        "models-root",
        "audio-root",
        "encoder-output",
        "stage",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-file", type=Path)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--result-stage", type=Path)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--allocator-gib", type=float, default=6)
    parser.add_argument("--lease-file", type=Path)
    parser.add_argument("--resource-window-confirmed", action="store_true")
    parser.add_argument(
        "--action", choices=["plan", "execute", "verify"], default="plan"
    )
    args = parser.parse_args()
    for name in (
        "author_root",
        "prepared",
        "metadata_root",
        "models_root",
        "audio_root",
        "encoder_output",
        "stage",
        "source_file",
        "result_stage",
        "lease_file",
    ):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
