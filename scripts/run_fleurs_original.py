"""Original FLEURS waveforms through pinned multilingual Whisper on Ascend NPU.

Private output contains original annotations and model text. No prompt conversion,
language filtering, waveform trimming, model fallback, or failed-case replacement.
"""

import argparse
import hashlib
from importlib import metadata
from io import BytesIO
import json
from pathlib import Path

from prepare_fleurs_original_assets import select_files
from prepare_xsum_original_assets import bound_path, sha, verify

PROFILE = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/fleurs-whisper-profile.json"
)


def verify_framework(profile):
    pins = json.loads(
        (PROFILE.parent / profile["framework_pins_resource"]).read_text(
            encoding="utf-8"
        )
    )
    if metadata.version("openai-whisper") != profile["framework_version"]:
        raise ValueError("pinned publisher Whisper version required")
    distribution = metadata.distribution("openai-whisper")
    for item in pins["framework_files"]:
        raw = Path(distribution.locate_file(item["path"])).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if len(raw) != item["bytes"] or blob != item["git_blob"]:
            raise ValueError("installed Whisper publisher source identity mismatch")
    return len(pins["framework_files"])


def original_indices(rows, full=False):
    if rows <= 0:
        raise ValueError("positive original row count required")
    return range(rows if full else min(3, rows))


def decode_original_audio(row, soundfile):
    blob = row["audio"]["bytes"]
    stream = BytesIO(blob)
    info = soundfile.info(stream)
    stream.seek(0)
    audio, rate = soundfile.read(stream, dtype="float32")
    if (
        rate != 16000
        or info.channels != 1
        or audio.ndim != 1
        or len(audio) != row["num_samples"]
        or len(audio) != info.frames
    ):
        raise ValueError("original mono 16k waveform/sample count mismatch; no repair")
    return audio, hashlib.sha256(blob).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--locales", nargs="+")
    parser.add_argument("--device", default="npu:1")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if not args.device.startswith("npu:") or args.output.exists():
        raise ValueError("explicit NPU and fresh private output required")
    from vllm_hust_workload_data.registry import source_spec
    import pyarrow.parquet as pq
    import soundfile as sf
    import torch
    import torch_npu  # noqa: F401
    import whisper

    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    verified_framework_files = verify_framework(profile)
    checkpoint = args.model_root / profile["checkpoint"]
    verify(
        checkpoint,
        {"bytes": profile["checkpoint_bytes"], "sha256": profile["checkpoint_sha256"]},
    )
    if whisper._MODELS[profile["model"]] != profile["checkpoint_url"]:
        raise ValueError("publisher model URL differs from fixed checkpoint binding")
    source = source_spec("FLEURS")
    selected = select_files(source, "test", args.locales)
    # Preflight every selected original file before allocating NPU memory.
    for item in selected:
        verify(bound_path(args.data_root, "", item), item)
    args.output.mkdir(parents=True, mode=0o700)
    receipt = {
        "source_id": "FLEURS",
        "revision": source["revision"],
        "profile_sha256": sha(PROFILE),
        "runner_sha256": sha(Path(__file__)),
        "framework_files_verified": verified_framework_files,
        "model": profile["model"],
        "checkpoint_sha256": profile["checkpoint_sha256"],
        "device": args.device,
        "task": "transcribe",
        "language": None,
        "full": args.full,
        "selected_locales": [item["locale"] for item in selected],
        "expected_cases_per_locale": {
            item["locale"]: len(original_indices(item["rows"], args.full))
            for item in selected
        },
        "status": "LOADING_MODEL",
        "model_calls": 0,
        "cases": [],
        "failures": [],
        "source_checkbox_eligible": False,
    }

    def save():
        (args.output / "receipt-private.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )

    save()
    torch.npu.set_device(args.device)
    try:
        model = whisper.load_model(
            profile["model"], device=args.device, download_root=str(args.model_root)
        )
        if not all(str(p.device) == args.device for p in model.parameters()):
            raise ValueError("model parameters not exclusively on selected NPU")
        receipt["status"] = "RUNNING_ORIGINAL_TASKS"
        save()
        for item in selected:
            locale = item["locale"]
            parquet = pq.ParquetFile(bound_path(args.data_root, "", item))
            if parquet.metadata.num_rows != item["rows"]:
                raise ValueError("original row count differs from publisher metadata")
            indices = original_indices(item["rows"], args.full)
            for index, row in enumerate(parquet.iter_batches(batch_size=1)):
                if index not in indices:
                    break
                try:
                    original = row.to_pylist()[0]
                    audio, audio_sha = decode_original_audio(original, sf)
                    receipt["model_calls"] += 1
                    save()
                    result = model.transcribe(audio, task="transcribe", verbose=None)
                    filename = f"{locale}-{index:06d}.private.json"
                    payload = {
                        "locale": locale,
                        "position": index,
                        "original": {
                            key: value
                            for key, value in original.items()
                            if key != "audio"
                        },
                        "original_audio_path": original["audio"]["path"],
                        "audio_sha256": audio_sha,
                        "result": result,
                    }
                    destination = args.output / filename
                    destination.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    receipt["cases"].append(
                        {
                            "locale": locale,
                            "position": index,
                            "file": filename,
                            "sha256": sha(destination),
                            "empty_output_retained": not result["text"],
                        }
                    )
                except Exception as error:
                    receipt["failures"].append(
                        {
                            "locale": locale,
                            "position": index,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
                save()
        receipt["status"] = (
            "NATIVE_EXECUTION_COMPLETE_SCORE_PENDING"
            if not receipt["failures"]
            else "FAILED_CASES_RETAINED_NOT_COMPLETE"
        )
    except Exception as error:
        receipt.update(
            status="FAILED_NOT_COMPLETE",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    finally:
        save()
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "source_id",
                    "status",
                    "model_calls",
                    "source_checkbox_eligible",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
