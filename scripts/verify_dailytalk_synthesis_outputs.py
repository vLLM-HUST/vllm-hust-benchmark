"""Audit saved original Guo synthesis without loading or rerunning neural models."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import wave

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_dailytalk_guo_encoder_npu import load_inputs
from verify_dailytalk_original_models import PROFILE, verify_file


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def contained_file(root, relative):
    path = root / relative
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(
        root.resolve()
    ):
        raise ValueError("saved output must remain in its declared stage")
    if not path.is_file():
        raise ValueError("saved output must be a regular file")
    return path


def audit(args):
    report = {
        "source_id": "DAILYTALK",
        "status": "PLAN_ONLY_NO_MODEL_LOAD",
        "audit_model_calls": 0,
        "quality_score_claimed": False,
        "human_quality_evaluation_performed": False,
        "source_checkbox_eligible": False,
    }
    if args.action == "plan":
        return report
    rows, contract = load_inputs(args)
    stage = args.stage.resolve(strict=True)
    native = json.loads(
        contained_file(stage, "native-execution/state-private.json").read_text("utf-8")
    )
    if native["status"] != (
        "ORIGINAL_GUO_BATCH_SYNTHESIS_RETURNED_STRUCTURAL_OUTPUT_CHECKS_PASSED"
    ):
        raise ValueError("original synthesis did not return successfully")
    for key, expected in {
        "original_seed": 1234,
        "original_batch_size": 8,
        "original_history_window_turns": 10,
        "original_restore_step": 900000,
        "original_pitch_energy_duration_controls": [1.0, 1.0, 1.0],
        "weights_only_required": True,
        "unsafe_pickle_fallback_enabled": False,
    }.items():
        if native[key] != expected:
            raise ValueError(f"saved original synthesis configuration differs: {key}")
    batches = (len(rows) + 7) // 8
    if native["native_model_forward_entries"] != {
        "acoustic": batches,
        "vocoder": batches,
    }:
        raise ValueError("original acoustic/vocoder batch coverage differs")
    profile = json.loads(PROFILE.read_text("utf-8"))
    for role, name in (
        ("acoustic", "900000.pth.tar"),
        ("vocoder", "generator_universal.pth.tar"),
    ):
        verify_file(contained_file(args.models_root, name), profile[role])
    identities = [item["utterance_id"] for item in contract["utterances"]]
    outputs = native["outputs"]
    if len(outputs) != len(identities) or {
        item["utterance_id"] for item in outputs
    } != set(identities):
        raise ValueError(
            "saved outputs must cover every original utterance exactly once"
        )
    saved = {item["utterance_id"]: item for item in outputs}
    frames, total_bytes, nonzero = [], 0, 0
    expected_paths = set()
    for identity in identities:
        if not re.fullmatch(r"\d+_\d+_d\d+", identity):
            raise ValueError("original utterance identity required")
        relative = f"results/900000/{identity.rsplit('_d', 1)[1]}/{identity}.wav"
        record = saved[identity]
        if (
            record["path"] != relative
            or record["original_plot_path"] != relative[:-4] + ".png"
        ):
            raise ValueError("saved paths differ from original publisher naming")
        wav = contained_file(stage, relative)
        png = contained_file(stage, relative[:-4] + ".png")
        expected_paths.add(wav.resolve())
        if digest(wav) != record["sha256"]:
            raise ValueError("saved WAV digest differs")
        with wave.open(str(wav), "rb") as stream:
            count = stream.getnframes()
            if (
                stream.getframerate() != 22050
                or stream.getsampwidth() != 2
                or stream.getnchannels() != 1
                or stream.getcomptype() != "NONE"
                or count <= 0
                or count != record["frames"]
            ):
                raise ValueError(
                    "original mono PCM format or saved frame count differs"
                )
            remaining, has_signal = count, False
            while remaining:
                requested = min(remaining, 32768)
                raw = stream.readframes(requested)
                if len(raw) != requested * 2:
                    raise ValueError("truncated PCM payload")
                has_signal |= any(raw)
                remaining -= requested
        with png.open("rb") as stream:
            header = stream.read(24)
        if (
            len(header) != 24
            or header[:8] != b"\x89PNG\r\n\x1a\n"
            or header[12:16] != b"IHDR"
            or min(struct.unpack(">II", header[16:24])) <= 0
        ):
            raise ValueError("original plot PNG header or dimensions invalid")
        frames.append(count)
        total_bytes += wav.stat().st_size
        nonzero += int(has_signal)
    if {
        path.resolve() for path in (stage / "results").rglob("*.wav")
    } != expected_paths:
        raise ValueError("unexpected or missing synthesis WAV files")
    report.update(
        status="SAVED_ORIGINAL_GUO_SYNTHESIS_OUTPUTS_VERIFIED_NOT_SOURCE_DELIVERY",
        original_utterances=len(identities),
        acoustic_forward_batches=batches,
        vocoder_forward_batches=batches,
        wav_files_full_pcm_decoded=len(frames),
        wav_files_nonzero=nonzero,
        png_files_signature_and_dimensions_verified=len(frames),
        total_wav_bytes=total_bytes,
        minimum_frames=min(frames),
        maximum_frames=max(frames),
        official_model_files_rehashed=2,
        original_requests_and_history_ids_preserved=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--source-file", type=Path)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--action", choices=["plan", "verify"], default="plan")
    print(json.dumps(audit(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
