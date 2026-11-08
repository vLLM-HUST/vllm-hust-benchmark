"""Original LJ Speech -> VITS text-to-wave protocol, explicit Ascend binding.

Defaults to a plan. Fixed smoke uses the first three original metadata rows;
--full uses all rows. No truncation, ASR replacement, MOS or performance claim.
"""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile

from prepare_jsonschemabench_author_source import verify_source
from prepare_ljspeech_original_assets import verify

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PINS = ROOT / "configs/workload-native/ljspeech-vits-author-source.json"
DATA_PINS = ROOT / "configs/workload-native/ljspeech-original-assets.json"
CHECKPOINT_SHA256 = "c94fb49d08ba90c598de16e7d5dec8d26bf225c1cf193a4fba05eb2dbda5a561"  # pragma: allowlist secret - observed official-linked checkpoint
METADATA_SHA256 = "852bce5c79e5184ba5bf2ea98aad4bcf49a5d787def5fd1f04fa2ab8aa6dc535"  # pragma: allowlist secret - original metadata identity


def metadata_rows(archive_path, *, expected_digest=METADATA_SHA256):
    with tarfile.open(archive_path, "r|bz2") as archive:
        for member in archive:
            if member.name != "LJSpeech-1.1/metadata.csv":
                continue
            if not member.isfile() or not 0 < member.size < 16 * 1024**2:
                raise ValueError("original regular metadata member required")
            with archive.extractfile(member) as stream:
                raw = stream.read()
            break
        else:
            raise ValueError("original metadata missing")
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        raise ValueError("original metadata digest differs")
    rows = list(
        csv.reader(
            io.StringIO(raw.decode("utf-8")), delimiter="|", quoting=csv.QUOTE_NONE
        )
    )
    if len(rows) != 13100 or any(len(row) != 3 or not all(row) for row in rows):
        raise ValueError("all 13100 original metadata triples required")
    if [row[0] for row in rows[:3]] != ["LJ001-0001", "LJ001-0002", "LJ001-0003"]:
        raise ValueError("original metadata order differs")
    return rows


def verify_checkpoint(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != 145599717:
        raise ValueError("original regular pretrained_ljs.pth required")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise ValueError("publisher-linked checkpoint identity differs")


def execute(args):
    archive = args.archive.resolve()
    checkpoint = args.checkpoint.resolve()
    author = args.author_root.resolve()
    output = args.output.resolve()
    if output.is_relative_to(ROOT) or output.exists():
        raise ValueError("fresh output directory outside checkout required")
    if not os.environ.get("PHONEMIZER_ESPEAK_PATH"):
        raise ValueError("explicit qualified eSpeak executable required")
    verify_source(author, SOURCE_PINS)
    data = json.loads(DATA_PINS.read_text("utf-8"))["dataset"]
    verify(archive, data)
    verify_checkpoint(checkpoint)
    all_rows = metadata_rows(archive)
    rows = all_rows if args.full else all_rows[:3]
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "source_id": "LJSPEECH",
        "status": "ORIGINAL_VITS_SYNTHESIS_RUNNING",
        "protocol": "original-vits-single-speaker-text-to-wave",
        "selected_rows": len(rows),
        "selection": "all_original_rows" if args.full else "first3_original_rows",
        "input_field": "publisher_normalized_transcription",
        "text_rewritten": False,
        "original_reference_audio_used_as_model_input": False,
        "noise_scale": 0.667,
        "noise_scale_w": 0.8,
        "length_scale": 1,
        "sampling_rate": 22050,
        "output_truncated": False,
        "device": "npu:0",
        "dtype": "float32",
        "seed": args.seed,
        "seed_is_disclosed_reproducibility_setting": True,
        "model_calls": 0,
        "outputs": [],
        "source_checkbox_eligible": False,
        "human_quality_evaluation_performed": False,
    }
    receipt = output / "qualification-public.json"

    def save():
        receipt.write_text(json.dumps(report, indent=2), encoding="utf-8")

    save()
    try:
        import numpy as np
        from scipy.io import wavfile
        import torch
        import torch_npu  # noqa: F401 - explicit backend, no CPU/CUDA fallback

        if not torch.npu.is_available():
            raise RuntimeError("Ascend NPU required; no neural CPU/CUDA fallback")
        torch.manual_seed(args.seed)
        torch.npu.manual_seed_all(args.seed)
        torch.npu.set_device(0)
        sys.path.insert(0, str(author))
        import commons
        from models import SynthesizerTrn
        from text import text_to_sequence
        from text.symbols import symbols
        import utils

        hps = utils.get_hparams_from_file(str(author / "configs/ljs_base.json"))
        model = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
        )
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"], strict=True)
        del saved
        model = model.to("npu:0").eval()
        if any(
            p.device.type != "npu" or p.dtype != torch.float32
            for p in model.parameters()
        ):
            raise RuntimeError("all model parameters must remain float32 on NPU")
        for identity, _text, normalized in rows:
            sequence = text_to_sequence(normalized, hps.data.text_cleaners)
            if hps.data.add_blank:
                sequence = commons.intersperse(sequence, 0)
            tokens = torch.LongTensor(sequence).unsqueeze(0).to("npu:0")
            lengths = torch.LongTensor([len(sequence)]).to("npu:0")
            report["model_calls"] += 1
            save()
            with torch.no_grad():
                generated = model.infer(
                    tokens,
                    lengths,
                    noise_scale=0.667,
                    noise_scale_w=0.8,
                    length_scale=1,
                )[0][0, 0]
            if generated.device.type != "npu":
                raise RuntimeError("neural output must originate on NPU")
            waveform = generated.detach().cpu().float().numpy()
            if (
                waveform.ndim != 1
                or waveform.size == 0
                or not np.isfinite(waveform).all()
            ):
                raise ValueError("empty or nonfinite original synthesis output")
            path = output / f"{identity}.wav"
            wavfile.write(path, hps.data.sampling_rate, waveform)
            rate, restored = wavfile.read(path)
            if rate != 22050 or not np.array_equal(restored, waveform):
                raise ValueError("saved float waveform differs from model output")
            report["outputs"].append(
                {
                    "id": identity,
                    "input_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
                    "wav": path.name,
                    "samples": int(waveform.size),
                    "finite": True,
                    "peak_absolute": float(np.abs(waveform).max()),
                    "rms": float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2))),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "reference_audio_member": f"LJSpeech-1.1/wavs/{identity}.wav",
                }
            )
            save()
        report["status"] = "ORIGINAL_VITS_NPU_SYNTHESIS_AND_WAVE_ROUNDTRIP_PASSED"
        save()
    except Exception as exc:
        report.update(
            status="ORIGINAL_VITS_SYNTHESIS_FAILED_RETAINED",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        save()
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "source_id": "LJSPEECH",
                    "status": "PLAN_ONLY_NO_MODEL_CALLS",
                    "rows": 13100 if args.full else 3,
                    "input_field": "publisher_normalized_transcription",
                    "output_truncated": False,
                    "model_calls": 0,
                    "source_checkbox_eligible": False,
                }
            )
        )
        return
    print(json.dumps(execute(args)))


if __name__ == "__main__":
    main()
