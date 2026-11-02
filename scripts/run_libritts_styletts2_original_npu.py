"""Original LibriTTS text to StyleTTS2 waveform; plan by default, explicit NPU.

Uses the publisher's unchanged notebook inference definitions. Reference voice
comes from a different original utterance of the same speaker. This is a declared
TTS protocol, not an official corpus quality metric or a performance result.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile

from prepare_libritts_styletts2_model import verify

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "configs/workload-native/libritts-styletts2-model.json"
AUTHOR_TREE = "ac9c64d908d0bdca127de56bec33c6f78d519c52"  # pragma: allowlist secret - verified public author tree
DATA_SHA = "33a5342094f3bba7ccc2e0500b9e72d558f72eb99328ac8debe1d9080402f10d"  # pragma: allowlist secret - official TorchAudio corpus checksum


def reference_for(identity, members):
    speaker = identity.split("_")[0]
    candidates = sorted(
        name
        for name in members
        if PurePosixPath(name).parts[-3] == speaker
        and PurePosixPath(name).stem != identity
    )
    if not candidates:
        raise ValueError("distinct original same-speaker recording required")
    return candidates[-1]


def notebook_definitions(notebook):
    cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    preprocess = next(
        cell for cell in cells if cell.startswith("to_mel = torchaudio.transforms.")
    )
    inference_cell = next(cell for cell in cells if cell.startswith("def inference("))
    tree = ast.parse(inference_cell)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("single unchanged author inference definition required")
    return preprocess, inference_cell


def execute(args):
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(ROOT):
        raise ValueError("fresh external output directory required")
    author = args.author_root.resolve()
    pins = json.loads(PINS.read_text("utf-8"))

    def git(*parts):
        return subprocess.check_output(
            ["git", "-C", str(author), *parts], text=True
        ).strip()

    if (
        git("rev-parse", "HEAD") != pins["author_revision"]
        or git("write-tree") != AUTHOR_TREE
        or git("diff", "--name-only")
        or git("ls-files", "--others")
    ):
        raise ValueError("exact unchanged pinned publisher source required")
    for item in pins["files"]:
        verify(args.model_root / item["path"], item)
    with args.archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != DATA_SHA:
            raise ValueError("original full test-other corpus identity differs")
    from vllm_hust_workload_data.libritts import original_rows

    all_rows = list(original_rows(args.archive))
    if len(all_rows) != 5120:
        raise ValueError("all original 5120 records required before selection")
    rows = all_rows if args.full else all_rows[:3]
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "source_id": "LIBRITTS",
        "status": "PUBLISHER_STYLE_TTS2_NPU_RUNNING",
        "protocol": "original-test-other-multispeaker-reference-conditioned-tts",
        "selection": "all5120" if args.full else "lexical-first3",
        "reference_rule": "lexically last distinct utterance of the same original speaker",
        "alpha": 0.3,
        "beta": 0.7,
        "diffusion_steps": 5,
        "embedding_scale": 1,
        "seed": 0,
        "dtype": "float32",
        "device": "npu:0",
        "author_tail_postprocessing_samples": 50,
        "outputs": [],
        "attempted_records": [],
        "source_checkbox_eligible": False,
        "human_quality_evaluation_performed": False,
    }
    receipt = output / "qualification-public.json"

    def save():
        receipt.write_text(json.dumps(report, indent=2), encoding="utf-8")

    save()
    try:
        import random

        import librosa
        import nltk
        import numpy as np
        import phonemizer
        import soundfile as sf
        import torch
        import torch_npu  # noqa: F401 - explicit NPU backend
        import torchaudio
        import yaml

        if not torch.npu.is_available():
            raise RuntimeError("NPU required; no neural CPU/CUDA fallback")
        torch.npu.set_device(0)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.npu.manual_seed_all(0)
        nltk.data.path[:] = [str(args.nltk_root.resolve())]
        sys.path.insert(0, str(author))
        os.chdir(author)
        from models import ASRCNN, JDCNet, build_model
        from Modules.diffusion.sampler import (
            ADPM2Sampler,
            DiffusionSampler,
            KarrasSchedule,
        )
        from text_utils import TextCleaner
        from transformers import AlbertConfig
        from Utils.PLBERT.util import CustomAlbert
        from utils import recursive_munch

        config = yaml.safe_load(
            (args.model_root / "Models/LibriTTS/config.yml").read_bytes()
        )
        asr = yaml.safe_load((author / config["ASR_config"]).read_bytes())
        bert = yaml.safe_load(
            (author / config["PLBERT_dir"] / "config.yml").read_bytes()
        )
        params = recursive_munch(config["model_params"])
        model = build_model(
            params,
            ASRCNN(**asr["model_params"]),
            JDCNet(num_class=1, seq_len=192),
            CustomAlbert(AlbertConfig(**bert["model_params"])),
        )
        checkpoint = args.model_root / "Models/LibriTTS/epochs_2nd_00020.pth"
        states = torch.load(checkpoint, map_location="cpu", weights_only=True)["net"]
        for name, network in model.items():
            state = states[name]
            if all(key.startswith("module.") for key in state):
                state = {key[7:]: value for key, value in state.items()}
            network.load_state_dict(state, strict=True)
            network.to("npu:0").eval()
            if any(
                p.device.type != "npu" or p.dtype != torch.float32
                for p in network.parameters()
            ):
                raise RuntimeError("all neural parameters must remain float32 on NPU")
        del states
        sampler = DiffusionSampler(
            model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False,
        )
        namespace = {
            "torch": torch,
            "torchaudio": torchaudio,
            "librosa": librosa,
            "model": model,
            "model_params": params,
            "device": "npu:0",
            "word_tokenize": nltk.word_tokenize,
            "textclenaer": TextCleaner(),
            "global_phonemizer": phonemizer.backend.EspeakBackend(
                language="en-us", preserve_punctuation=True, with_stress=True
            ),
            "sampler": sampler,
        }
        definitions = notebook_definitions(
            json.loads((author / "Demo/Inference_LibriTTS.ipynb").read_text())
        )
        for definition in definitions:
            exec(compile(definition, "unchanged-publisher-notebook", "exec"), namespace)
        with tarfile.open(args.archive, "r:gz") as archive:
            members = {
                member.name: member
                for member in archive.getmembers()
                if member.isfile() and member.name.endswith(".wav")
            }
            for row in rows:
                ref = reference_for(row["id"], members)
                raw = archive.extractfile(members[ref]).read()
                reference_path = output / (PurePosixPath(ref).stem + "-reference.wav")
                if not reference_path.exists():
                    reference_path.write_bytes(raw)
                report["attempted_records"].append(row["id"])
                save()
                style = namespace["compute_style"](str(reference_path))
                if style.device.type != "npu":
                    raise RuntimeError(
                        "reference neural encoding must originate on NPU"
                    )
                waveform = namespace["inference"](row["normalized_text"], style)
                if (
                    waveform.ndim != 1
                    or not waveform.size
                    or not np.isfinite(waveform).all()
                    or not np.any(waveform)
                ):
                    raise ValueError(
                        "finite nonempty nonzero original synthesis required"
                    )
                path = output / (row["id"] + ".wav")
                sf.write(path, waveform, 24000, subtype="FLOAT")
                report["outputs"].append(
                    {
                        "id": row["id"],
                        "normalized_text_sha256": hashlib.sha256(
                            row["normalized_text"].encode()
                        ).hexdigest(),
                        "conditioning_reference_id": PurePosixPath(ref).stem,
                        "conditioning_reference_sha256": hashlib.sha256(
                            raw
                        ).hexdigest(),
                        "samples": int(waveform.size),
                        "sample_rate": 24000,
                        "waveform_sha256": hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest(),
                    }
                )
                save()
        report["status"] = "FIXED_OR_FULL_ORIGINAL_TTS_SYNTHESIS_PASSED_HANDOFF_PENDING"
    except BaseException as error:
        report.update(status="NATIVE_FAILURE_RETAINED", error=str(error)[:2000])
        raise
    finally:
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--nltk-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    for name in ("author_root", "model_root", "archive", "nltk_root", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.execute:
        execute(args)
    else:
        print(
            json.dumps(
                {
                    "mode": "PLAN_ONLY",
                    "protocol": "published-styletts2-libritts",
                    "neural_calls": 0,
                    "full": args.full,
                }
            )
        )


if __name__ == "__main__":
    main()
