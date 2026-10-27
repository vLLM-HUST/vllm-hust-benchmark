"""Original STSBenchmark pairs and SBERT evaluator; NPU model or cached rescoring.

Scores and split metadata never enter model inputs. No prompts, binary-label
conversion, sentence rewriting, independent training or research observation.
"""

import argparse
from collections import Counter
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import time


PINS = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/stsb-author-pins.json"
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_identity(sentences):
    return hashlib.sha256(
        json.dumps(list(sentences), ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def verify_files(root, files):
    root = Path(root).resolve()
    for spec in files:
        path = (root / spec["path"]).resolve()
        if not path.is_relative_to(root) or path.stat().st_size != spec["bytes"]:
            raise ValueError("original pinned file path or size mismatch")
        if spec.get("sha256") and sha(path) != spec["sha256"]:
            raise ValueError("original pinned file SHA256 mismatch")
        if spec.get("git_blob"):
            payload = path.read_bytes()
            if (
                hashlib.sha1(
                    b"blob " + str(len(payload)).encode() + b"\0" + payload
                ).hexdigest()
                != spec["git_blob"]
            ):
                raise ValueError("original publisher Git object mismatch")
        if not spec.get("sha256") and not spec.get("git_blob"):
            raise ValueError("unfrozen original file identity")


def validate_cached_call(call, *, sentences, kwargs, directory):
    if (
        call["input_sha256"] != input_identity(sentences)
        or call["encode_options"] != kwargs
    ):
        raise ValueError("cached encoding differs from original input or protocol")
    path = (Path(directory) / call["file"]).resolve()
    if (
        not path.is_relative_to(Path(directory).resolve())
        or sha(path) != call["sha256"]
    ):
        raise ValueError("cached embedding file identity mismatch")
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "test"], action="append")
    parser.add_argument("--mode", choices=["run", "score"], default="run")
    parser.add_argument("--reuse", type=Path)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args(argv)
    if args.mode == "run" and (not args.model_root or args.reuse):
        parser.error(
            "run needs pinned model-root and does not silently reuse generation"
        )
    if args.mode == "run" and (
        not args.device.startswith("npu:") or not args.device[4:].isdigit()
    ):
        parser.error("model execution requires explicit npu:<index>")
    if args.mode == "score" and (not args.reuse or args.model_root):
        parser.error("score needs only the original saved run, never a CPU model")
    if args.split and len(args.split) != len(set(args.split)):
        parser.error("duplicate original split")
    return args


def main(argv=None):
    args = parse_args(argv)
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    if args.output.exists():
        raise ValueError("fresh private output directory required")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from vllm_hust_workload_data.stsb import original_rows

    verify_files(args.data_root, [pins["dataset"]])
    rows = list(original_rows(args.data_root / pins["dataset"]["path"]))
    if dict(Counter(row["split"] for row in rows)) != pins["dataset"]["split_rows"]:
        raise ValueError("original full split counts differ")
    if metadata.version("sentence-transformers") != pins["framework_version"]:
        raise ValueError("pinned original evaluator version required")
    distribution = metadata.distribution("sentence-transformers")
    verify_files(
        distribution.locate_file(""),
        pins["installed_source_files"],
    )
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
    import numpy as np

    selected = args.split or pins["protocol"]["splits"]
    args.output.mkdir(parents=True, mode=0o700)
    receipt = {
        "source_id": "STSB",
        "status": "RUNNING",
        "mode": args.mode,
        "generation_performed_this_run": args.mode == "run",
        "dataset_sha256": pins["dataset"]["sha256"],
        "model_repo": pins["model"]["repo_id"],
        "model_revision": pins["model"]["revision"],
        "framework_revision": pins["framework_revision"],
        "protocol": pins["protocol"],
        "splits": [],
        "encoding_calls": [],
        "device": args.device if args.mode == "run" else None,
        "source_checkbox_eligible": False,
        "raw_payloads_published": False,
        "started_at_unix": time.time(),
    }

    def save():
        (args.output / "receipt-public.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )

    save()
    try:
        if args.mode == "run":
            import torch
            import torch_npu  # noqa: F401 - registers authorized NPU backend
            from sentence_transformers import SentenceTransformer

            if metadata.version("transformers") != "4.57.3":
                raise ValueError("validated Transformers client version required")
            verify_files(args.model_root, pins["model"]["files"])
            target = torch.device(args.device)
            torch.npu.set_device(target)
            if torch.npu.mem_get_info()[0] < 6 * 1024**3:
                raise ValueError("insufficient NPU memory; do not stop other services")

            class NativeEncoder:
                # The original loader checks the concrete class name against
                # the saved model type, so never subclass it for validation.
                def __init__(self, original_model):
                    self.original_model = original_model
                    self.similarity_fn_name = original_model.similarity_fn_name
                    self.model_card_data = original_model.model_card_data

                def encode(self, sentences, **kwargs):
                    output = self.original_model.encode(sentences, **kwargs)
                    if (
                        not isinstance(output, np.ndarray)
                        or output.ndim != 2
                        or not np.isfinite(output).all()
                    ):
                        raise ValueError(
                            "original evaluator's finite numpy embeddings required"
                        )
                    filename = f"encode-{len(receipt['encoding_calls'])}.private.npy"
                    np.save(args.output / filename, output, allow_pickle=False)
                    receipt["encoding_calls"].append(
                        {
                            "input_sha256": input_identity(sentences),
                            "encode_options": kwargs,
                            "file": filename,
                            "sha256": sha(args.output / filename),
                            "rows": len(sentences),
                        }
                    )
                    save()
                    return output

            native_model = SentenceTransformer(
                str(args.model_root),
                device=args.device,
                local_files_only=True,
                trust_remote_code=False,
                model_kwargs={"dtype": torch.float32, "attn_implementation": "eager"},
            )
            if not all(
                parameter.device == target for parameter in native_model.parameters()
            ):
                raise ValueError("model parameters must all be on selected NPU")
            native_config = json.loads(
                (args.model_root / "sentence_bert_config.json").read_text()
            )
            if native_model.max_seq_length != native_config["max_seq_length"]:
                raise ValueError("original model-native context policy changed")

            def check_native_output(module, inputs, output):
                embedding = output["sentence_embedding"]
                if (
                    embedding.device != target
                    or not torch.isfinite(embedding).all().item()
                ):
                    raise ValueError("finite NPU model embeddings required")
                receipt["model_forward_embeddings_verified_on_npu"] = True

            # Non-mutating assertion, not a replacement of the original graph.
            native_model[-1].register_forward_hook(check_native_output)
            model = NativeEncoder(native_model)
            receipt.update(
                model_parameters_verified_on_npu=True,
                encoder_native_max_length=native_model.max_seq_length,
                similarity_fn_name=model.similarity_fn_name,
            )
        else:
            from sentence_transformers.model_card import (
                SentenceTransformerModelCardData,
            )

            previous = json.loads(
                (args.reuse / "receipt-public.json").read_text(encoding="utf-8")
            )
            for key in [
                "source_id",
                "dataset_sha256",
                "model_repo",
                "model_revision",
                "framework_revision",
                "protocol",
            ]:
                if previous[key] != receipt[key]:
                    raise ValueError("cached native run identity or protocol differs")
            if (
                previous["status"] != "ORIGINAL_FULL_SPLITS_EXECUTED_AND_SCORED"
                or [r["split"] for r in previous["splits"]] != selected
            ):
                raise ValueError("complete matching original split sequence required")

            class CachedEncoder:
                similarity_fn_name = previous["similarity_fn_name"]
                model_card_data = SentenceTransformerModelCardData()

                def encode(self, sentences, **kwargs):
                    index = len(receipt["encoding_calls"])
                    call = previous["encoding_calls"][index]
                    path = validate_cached_call(
                        call, sentences=sentences, kwargs=kwargs, directory=args.reuse
                    )
                    values = np.load(path, allow_pickle=False)
                    if (
                        values.ndim != 2
                        or len(values) != len(sentences)
                        or not np.isfinite(values).all()
                    ):
                        raise ValueError(
                            "cached original embedding shape or values differ"
                        )
                    receipt["encoding_calls"].append(call)
                    return values.copy()

            model = CachedEncoder()
            receipt["saved_native_receipt_sha256"] = sha(
                args.reuse / "receipt-public.json"
            )
        for split in selected:
            original = [row for row in rows if row["split"] == split]
            if len(original) != pins["protocol"]["expected_rows"][split]:
                raise ValueError("full original evaluation split required")
            evaluator = EmbeddingSimilarityEvaluator(
                sentences1=[row["sentence1"] for row in original],
                sentences2=[row["sentence2"] for row in original],
                scores=[float(row["score"]) / 5.0 for row in original],
                name="sts_" + split,
                batch_size=pins["protocol"]["batch_size"],
                show_progress_bar=False,
                write_csv=False,
            )
            values = evaluator(model)
            if not values or not all(
                math.isfinite(float(value)) for value in values.values()
            ):
                raise ValueError(
                    "original evaluator returned undefined metrics; do not replace tasks"
                )
            receipt["splits"].append(
                {
                    "split": split,
                    "original_pairs": len(original),
                    "metrics": {key: float(value) for key, value in values.items()},
                    "primary_metric": evaluator.primary_metric,
                }
            )
            save()
        if len(receipt["encoding_calls"]) != 2 * len(selected):
            raise ValueError(
                "original evaluator must encode exactly both sentence columns"
            )
        if args.mode == "run" and not receipt.get(
            "model_forward_embeddings_verified_on_npu"
        ):
            raise ValueError("original NPU model forward was not verified")
        receipt["status"] = (
            "ORIGINAL_FULL_SPLITS_EXECUTED_AND_SCORED"
            if args.mode == "run"
            else "ORIGINAL_EMBEDDINGS_RESCORED_NOT_NEW_MODEL_EXECUTION"
        )
    except Exception as exc:
        receipt.update(
            status="FAILED_NO_TASK_OR_MODEL_REPLACEMENT", error_type=type(exc).__name__
        )
        raise
    finally:
        receipt["finished_at_unix"] = time.time()
        save()


if __name__ == "__main__":
    main()
