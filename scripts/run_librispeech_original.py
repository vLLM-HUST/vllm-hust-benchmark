"""Original LibriSpeech FLAC, Whisper transcribe and publisher-normalized WER.

No 30-second trimming, invented prompt, CPU model fallback or research analysis.
Outputs contain original references and model text: keep the output directory private.
"""

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import time


MODEL_SHA256 = "25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead"  # pragma: allowlist secret - publisher checkpoint hash


def verify_publisher_installation():
    pins = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/workload-native/librispeech-whisper-author-pins.json"
        ).read_text()
    )
    if metadata.version("openai-whisper") != pins["framework_version"]:
        raise ValueError("publisher openai-whisper==20250625 required")
    distribution = metadata.distribution("openai-whisper")
    for spec in pins["framework_files"]:
        raw = Path(distribution.locate_file(spec["path"])).read_bytes()
        actual = hashlib.sha1(
            b"blob " + str(len(raw)).encode() + b"\0" + raw
        ).hexdigest()
        if len(raw) != spec["bytes"] or actual != spec["git_blob"]:
            raise ValueError("installed Whisper differs from pinned publisher source")
    if pins["model"]["publisher_sha256"] != MODEL_SHA256:
        raise ValueError("checkpoint pin differs from original source binding")
    return pins


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def select_indices(durations, *, full=False, indices=None):
    if not durations or any(value <= 0 for value in durations):
        raise ValueError("positive original audio durations required")
    if full and indices is not None:
        raise ValueError("full and explicit indices are mutually exclusive")
    if full:
        return list(range(len(durations)))
    if indices is not None:
        if len(set(indices)) != len(indices) or any(
            index < 0 or index >= len(durations) for index in indices
        ):
            raise ValueError("unique in-range original indices required")
        return sorted(indices)
    return sorted(
        set(range(min(3, len(durations))))
        | {max(range(len(durations)), key=durations.__getitem__)}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:1")
    parser.add_argument(
        "--split", choices=["test-clean", "test-other", "both"], default="both"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--full", action="store_true")
    group.add_argument(
        "--indices",
        help="fixed comma-separated original sorted-ID indices per selected split",
    )
    args = parser.parse_args()
    if not args.device.startswith("npu:") or args.output.exists():
        raise ValueError("NPU device and fresh private output required")
    from vllm_hust_workload_data.librispeech import (
        ARCHIVES,
        original_archive_rows,
        prepare_original_archives,
        prepared_audio_path,
    )
    from vllm_hust_workload_data.adapters import adapt_row
    from vllm_hust_workload_data.registry import adapter_spec, source_spec
    import soundfile as sf
    import torch
    import torch_npu  # noqa: F401
    import whisper
    import jiwer
    from whisper.normalizers import EnglishTextNormalizer

    pins = verify_publisher_installation()
    checkpoint = args.model_root / "base.en.pt"
    if digest(checkpoint) != MODEL_SHA256:
        raise ValueError("publisher base.en checkpoint SHA256 mismatch")
    if not args.prepared_root.exists():
        prepare_original_archives(data_root=args.data_root, output=args.prepared_root)
    preparation = json.loads(
        (args.prepared_root / "preparation.private.json").read_text()
    )
    args.output.mkdir(parents=True, mode=0o700)
    receipt = {
        "source_id": "LIBRISPEECH",
        "status": "PREPARING",
        "model_calls": 0,
        "framework_revision": pins["framework_revision"],
        "original_installed_source_files_verified": len(pins["framework_files"]),
        "protocol": "original Whisper transcribe complete waveform; language=en, task=transcribe, other native defaults",
        "device": args.device,
        "model": "base.en",
        "splits": [],
        "full_suite_claim": bool(args.full),
        "waveform_clipping": False,
    }

    def save():
        (args.output / "receipt-public.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )

    save()
    try:
        selected = []
        for asset in source_spec("LIBRISPEECH")["files"]:
            split = ARCHIVES[asset["path"]]
            if args.split not in ("both", split):
                continue
            archive = args.data_root / asset["path"]
            if digest(archive) != asset["sha256"]:
                raise ValueError("original archive SHA256 mismatch")
            rows = list(original_archive_rows(archive))
            if len(rows) != asset["rows"]:
                raise ValueError("original archive row count mismatch")
            records, paths, durations = [], [], []
            for index, row in enumerate(rows):
                record = adapt_row(
                    "LIBRISPEECH",
                    row,
                    source={
                        "source_id": "LIBRISPEECH",
                        "file": asset["path"],
                        "file_sha256": asset["sha256"],
                    },
                    spec=adapter_spec("LIBRISPEECH"),
                    index=index,
                )
                path = prepared_audio_path(
                    record, prepared_root=args.prepared_root, preparation=preparation
                )
                info = sf.info(path)
                if info.samplerate != 16000 or info.channels != 1:
                    raise ValueError(
                        "original mono 16kHz waveform required; no resampling"
                    )
                records.append(record)
                paths.append(path)
                durations.append(info.frames / info.samplerate)
            indices = select_indices(
                durations,
                full=args.full,
                indices=None
                if args.indices is None
                else [int(value) for value in args.indices.split(",")],
            )
            scope = {
                "split": split,
                "original_rows": len(rows),
                "selected_indices_before_outputs": indices,
                "completed": 0,
                "scoring_complete": False,
            }
            receipt["splits"].append(scope)
            selected.append((scope, records, paths))
        save()  # Freeze both scopes before loading the model or inspecting results.
        torch.npu.set_device(args.device)
        model = whisper.load_model(
            "base.en", device=args.device, download_root=str(args.model_root)
        )
        if not all(
            str(parameter.device) == args.device for parameter in model.parameters()
        ):
            raise ValueError(
                "all original model parameters must reside on selected NPU"
            )
        normalizer = EnglishTextNormalizer()
        for scope, records, paths in selected:
            references, hypotheses = [], []
            directory = args.output / scope["split"]
            directory.mkdir(mode=0o700)
            for index in scope["selected_indices_before_outputs"]:
                path, record = paths[index], records[index]
                audio, rate = sf.read(path, dtype="float32")
                if (
                    rate != 16000
                    or audio.ndim != 1
                    or len(audio) != sf.info(path).frames
                ):
                    raise ValueError("complete original waveform required")
                start = time.monotonic()
                result = model.transcribe(
                    audio, language="en", task="transcribe", verbose=None
                )
                reference = record.outcome_record()["text"]
                case = {
                    "original_index": index,
                    "utterance_id": record.task["task_id"],
                    "original_flac_sha256": digest(path),
                    "actual_input_samples": len(audio),
                    "original_audio_samples": sf.info(path).frames,
                    "sample_rate": rate,
                    "original_reference": reference,
                    "native_result": result,
                    "model_seconds": time.monotonic() - start,
                }
                (directory / f"{index}.private.json").write_text(
                    json.dumps(case) + "\n"
                )
                references.append(normalizer(reference))
                hypotheses.append(normalizer(result["text"]))
                receipt["model_calls"] += 1
                scope["completed"] += 1
                save()
            scope.update(wer=jiwer.wer(references, hypotheses), scoring_complete=True)
            save()
        receipt["status"] = "SELECTED_ORIGINAL_TASKS_EXECUTED_AND_SCORED"
    except Exception as exc:
        receipt.update(
            status="FAILED_NOT_COMPLETE", error_type=type(exc).__name__, error=str(exc)
        )
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
