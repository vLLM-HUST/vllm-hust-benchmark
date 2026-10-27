"""Original Multi-News line pairs, official Pegasus pipeline and HF ROUGE.

Reuse the data-only paired reader and shared, pinned HF scoring utilities.
Original documents/references/results remain private; NPU model, CPU scoring.
"""

import argparse
import hashlib
from importlib import metadata
import importlib.util
import json
from pathlib import Path
import time


PINS = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/multinews-author-pins.json"
)
INPUT_POLICY = "Original publisher loader fields and Google checkpoint summarization pipeline; only native encoder window truncation, original generation config unmodified"


def shared_utilities():
    spec = importlib.util.spec_from_file_location(
        "original_summary_utilities", Path(__file__).with_name("run_xsum_original.py")
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def canonical_row(row):
    return {key: row[key] for key in ["document", "summary"]}


def validate_saved_run(receipt, pins):
    statuses = {
        "ORIGINAL_MULTINEWS_GENERATION_COMPLETED",
        "ORIGINAL_FIRST3_NATIVE_PEGASUS_GENERATION_COMPLETED_SCORING_AND_PUBLIC_INTEGRATION_PENDING",
    }
    if (
        receipt.get("source_id") != "MULTINEWS"
        or receipt.get("status") not in statuses
        or receipt.get("model_revision") != pins["model"]["revision"]
        or receipt.get("dataset_revision") != pins["dataset"]["revision"]
        or receipt.get("input_policy") != INPUT_POLICY
        or receipt.get("native_generation_config") != pins["generation_config"]
        or receipt.get("versions", {}).get("transformers") != pins["framework_version"]
        or not receipt.get("model_parameters_verified_on_npu")
        or receipt.get("native_encoder_maximum_tokens")
        != pins["protocol"]["encoder_maximum_tokens"]
        or not receipt.get("device", "").startswith("npu:")
    ):
        raise ValueError(
            "completed original Multi-News NPU model/protocol binding required"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["generate", "score"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--reuse", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:1")
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--full", action="store_true")
    selection.add_argument(
        "--indices", help="fixed original physical line indices, comma-separated"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh private output required")
    if args.mode == "generate" and (
        args.model_root is None
        or args.reuse is not None
        or not args.device.startswith("npu:")
    ):
        raise ValueError("original model root and NPU generation required")
    if args.mode == "score" and (
        args.source_root is None
        or args.reuse is None
        or args.full
        or args.indices is not None
    ):
        raise ValueError(
            "score requires original sources and saved responses; selection is inherited"
        )
    common = shared_utilities()
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    common.verify_files(args.data_root, pins["dataset"]["files"])
    source = next(
        file
        for file in pins["dataset"]["files"]
        if file["role"] == "data" and file["split"] == args.split
    )
    from vllm_hust_workload_data.multinews import original_rows

    rows = list(original_rows(common.contained(args.data_root, source["path"])))
    if len(rows) != source["rows"]:
        raise ValueError("original split row count differs")
    args.output.mkdir(parents=True, mode=0o700)
    receipt = {
        "source_id": "MULTINEWS",
        "status": "PREPARING",
        "model_calls": 0,
        "split": args.split,
        "model": pins["model"]["repo_id"],
        "model_revision": pins["model"]["revision"],
        "dataset_revision": pins["dataset"]["revision"],
        "source_checkbox_eligible": False,
        "raw_payloads_published": False,
        "original_loader_NEWLINE_CHAR_restoration": True,
        "original_bundle_separators_preserved": True,
        "input_policy": INPUT_POLICY,
    }

    def save():
        (args.output / "receipt-public.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )

    save()
    try:
        if args.mode == "generate":
            common.verify_files(args.model_root, pins["model"]["files"])
            if metadata.version("transformers") != pins["framework_version"]:
                raise ValueError("pinned native Transformers client required")
            indices = common.selected_indices(
                len(rows),
                full=args.full,
                indices=None
                if args.indices is None
                else [int(value) for value in args.indices.split(",")],
            )
            receipt.update(
                selected_original_indices_before_outputs=indices,
                device=args.device,
                versions={
                    name: metadata.version(name)
                    for name in [
                        "torch",
                        "torch-npu",
                        "transformers",
                        "sentencepiece",
                        "protobuf",
                    ]
                },
            )
            save()
            import torch
            import torch_npu  # noqa: F401
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

            torch.npu.set_device(args.device)
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_root, local_files_only=True
            )
            model = (
                AutoModelForSeq2SeqLM.from_pretrained(
                    args.model_root, local_files_only=True
                )
                .eval()
                .to(args.device)
            )
            if (
                model.generation_config.to_dict() != pins["generation_config"]
                or tokenizer.model_max_length
                != pins["protocol"]["encoder_maximum_tokens"]
            ):
                raise ValueError("original model generation/window policy changed")
            if not all(
                str(parameter.device) == args.device for parameter in model.parameters()
            ):
                raise ValueError(
                    "all original model parameters must reside on selected NPU"
                )
            summarize = pipeline(
                "summarization", model=model, tokenizer=tokenizer, device=args.device
            )
            receipt.update(
                model_parameters_verified_on_npu=True,
                native_generation_config=model.generation_config.to_dict(),
                native_encoder_maximum_tokens=tokenizer.model_max_length,
            )
            for index in indices:
                row = canonical_row(rows[index])
                start = time.monotonic()
                result = summarize(
                    row["document"],
                    truncation=pins["protocol"]["truncation"],
                    generation_config=model.generation_config,
                )
                case = {
                    "original_index": index,
                    "original_row": row,
                    "native_result": result,
                    "original_document_utf8_sha256": hashlib.sha256(
                        row["document"].encode()
                    ).hexdigest(),
                    "model_seconds": time.monotonic() - start,
                }
                (args.output / f"{index}.private.json").write_text(
                    json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                receipt["model_calls"] += 1
                save()
            receipt["status"] = "ORIGINAL_MULTINEWS_GENERATION_COMPLETED"
        else:
            previous = json.loads(
                (args.reuse / "receipt-public.json").read_text(encoding="utf-8")
            )
            validate_saved_run(previous, pins)
            if previous["split"] != args.split:
                raise ValueError("saved original split differs")
            indices = previous["selected_original_indices_before_outputs"]
            if indices != common.selected_indices(
                len(rows), indices=indices
            ) or previous["model_calls"] != len(indices):
                raise ValueError(
                    "original selected positions are incomplete or unordered"
                )
            predictions, references = [], []
            for index in indices:
                case = json.loads(
                    common.contained(args.reuse, f"{index}.private.json").read_text(
                        encoding="utf-8"
                    )
                )
                row = canonical_row(rows[index])
                predictions.append(common.validate_saved_case(case, row, index))
                references.append(row["summary"])
            scores = common.original_rouge(
                predictions,
                references,
                source_root=args.source_root,
                output=args.output,
                pins=pins,
            )
            receipt.update(
                status="ORIGINAL_MULTINEWS_SAVED_RESPONSES_SCORED",
                original_indices=indices,
                native_responses_reused=len(indices),
                scoring_complete=True,
                rouge=scores,
            )
    except Exception as exc:
        receipt.update(
            status="FAILED_NOT_COMPLETE", error_type=type(exc).__name__, error=str(exc)
        )
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
