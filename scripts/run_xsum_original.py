"""Publisher XSum articles, official BART generation and original HF ROUGE recipe.

Original data, references, responses and complete indices remain external/private.
Generation uses NPU only. CPU scoring reuses saved responses without model calls.
"""

import argparse
import ast
import hashlib
from importlib import metadata
import json
from pathlib import Path
import time
from zipfile import ZipFile


PINS = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/xsum-author-pins.json"
)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError("contained original file path required")
    return path


def verify_files(root, files):
    for spec in files:
        path = contained(root, spec["path"])
        if path.stat().st_size != spec["bytes"] or sha(path) != spec["sha256"]:
            raise ValueError("original pinned asset bytes changed")


def selected_indices(size, *, full=False, indices=None):
    if size <= 0 or (full and indices is not None):
        raise ValueError("nonempty original split and exclusive selection required")
    selected = (
        list(range(size))
        if full
        else list(range(min(3, size)))
        if indices is None
        else indices
    )
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(type(i) is not int or i < 0 or i >= size for i in selected)
    ):
        raise ValueError("unique original in-range indices required")
    return sorted(selected)


def validate_saved_run(receipt, pins):
    valid_status = {
        "ORIGINAL_XSUM_GENERATION_COMPLETED",
        "ORIGINAL_FIRST3_NATIVE_BART_GENERATION_COMPLETED_SCORING_AND_PUBLIC_INTEGRATION_PENDING",
    }
    if (
        receipt.get("status") not in valid_status
        or receipt.get("source_id") != "XSUM"
        or receipt.get("model_revision") != pins["model"]["revision"]
        or receipt.get("dataset_revision") != pins["dataset"]["revision"]
        or receipt.get("versions", {}).get("transformers") != pins["framework_version"]
        or receipt.get("native_generation_config") != pins["generation_config"]
        or not receipt.get("all_model_parameters_verified_on_npu")
        or not receipt.get("device", "").startswith("npu:")
    ):
        raise ValueError("completed original NPU model/protocol binding required")


def validate_saved_case(case, row, index):
    if (
        case.get("original_index") != index
        or case.get("original_row") != row
        or case.get("original_document_utf8_sha256")
        != hashlib.sha256(row["document"].encode()).hexdigest()
        or not isinstance(case.get("native_result"), list)
        or len(case["native_result"]) != 1
        or not isinstance(case["native_result"][0].get("summary_text"), str)
    ):
        raise ValueError(
            "saved response differs from original article/reference/identity"
        )
    return case["native_result"][0]["summary_text"]


def original_postprocess(source):
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "postprocess_text"
    ]
    if len(functions) != 1:
        raise ValueError("exact original HF postprocess_text required")
    import nltk

    namespace = {"nltk": nltk}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace["postprocess_text"]


def original_rouge(predictions, references, *, source_root, output, pins):
    verify_files(source_root, pins["scoring_files"])
    import evaluate
    import nltk
    import numpy as np

    nltk_root = Path(output) / "nltk_data"
    with ZipFile(contained(source_root, "source/punkt_tab.zip")) as zipped:
        names = [
            name
            for name in zipped.namelist()
            if name.startswith("punkt_tab/english/") and not name.endswith("/")
        ]
        if len(names) != 4:
            raise ValueError("original English Punkt resource required")
        for name in names:
            path = contained(nltk_root / "tokenizers", name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(zipped.read(name))
    nltk.data.path[:] = [str(nltk_root)]
    postprocess = original_postprocess(
        contained(source_root, "source/run_summarization.py")
    )
    predictions, references = postprocess(predictions, references)
    np.random.seed(pins["protocol"]["bootstrap_seed"])
    rouge = evaluate.load(
        str(contained(source_root, "metric/rouge.py")),
        cache_dir=str(Path(output) / "metric-cache"),
    )
    scores = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=pins["protocol"]["use_stemmer"],
    )
    return {
        name: round(value * 100, pins["protocol"]["score_percent_round_digits"])
        for name, value in scores.items()
    }


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
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--full", action="store_true")
    group.add_argument(
        "--indices",
        help="comma-separated original physical row indices, fixed before outputs",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh private output required")
    if args.mode == "generate" and (
        args.model_root is None
        or args.reuse is not None
        or not args.device.startswith("npu:")
    ):
        raise ValueError(
            "generation requires original model root and NPU; no CPU fallback"
        )
    if args.mode == "score" and (
        args.source_root is None
        or args.reuse is None
        or args.full
        or args.indices is not None
    ):
        raise ValueError(
            "score requires original sources and saved run; selection is inherited, never replaced"
        )
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    verify_files(args.data_root, pins["dataset"]["files"])
    import pyarrow.parquet as pq

    spec = next(
        item for item in pins["dataset"]["files"] if item["split"] == args.split
    )
    table = pq.read_table(contained(args.data_root, spec["path"]))
    if table.num_rows != spec["rows"] or table.column_names != [
        "document",
        "summary",
        "id",
    ]:
        raise ValueError("original split schema/count mismatch")
    args.output.mkdir(parents=True, mode=0o700)
    receipt = {
        "source_id": "XSUM",
        "status": "PREPARING",
        "model_calls": 0,
        "split": args.split,
        "model": pins["model"]["repo_id"],
        "model_revision": pins["model"]["revision"],
        "dataset_revision": pins["dataset"]["revision"],
        "source_checkbox_eligible": False,
        "raw_payloads_published": False,
    }

    def save():
        (args.output / "receipt-public.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )

    save()
    try:
        if args.mode == "generate":
            verify_files(args.model_root, pins["model"]["files"])
            if metadata.version("transformers") != pins["framework_version"]:
                raise ValueError("pinned original generation framework required")
            indices = selected_indices(
                table.num_rows,
                full=args.full,
                indices=None
                if args.indices is None
                else [int(value) for value in args.indices.split(",")],
            )
            receipt.update(
                selected_original_indices_before_outputs=indices,
                device=args.device,
                versions={"transformers": metadata.version("transformers")},
            )
            save()
            import torch
            import torch_npu  # noqa: F401
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

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
                tokenizer.model_max_length != pins["protocol"]["encoder_maximum_tokens"]
                or model.generation_config.to_dict() != pins["generation_config"]
            ):
                raise ValueError("original encoder/generation policy changed")
            if not all(
                str(parameter.device) == args.device for parameter in model.parameters()
            ):
                raise ValueError("original model must reside entirely on selected NPU")
            summarize = pipeline(
                "summarization", model=model, tokenizer=tokenizer, device=args.device
            )
            receipt.update(
                native_generation_config=model.generation_config.to_dict(),
                all_model_parameters_verified_on_npu=True,
            )
            for index in indices:
                row = table.slice(index, 1).to_pylist()[0]
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
            receipt["status"] = "ORIGINAL_XSUM_GENERATION_COMPLETED"
        else:
            previous = json.loads(
                (args.reuse / "receipt-public.json").read_text(encoding="utf-8")
            )
            validate_saved_run(previous, pins)
            if previous["split"] != args.split:
                raise ValueError("saved original split differs")
            indices = previous["selected_original_indices_before_outputs"]
            if indices != selected_indices(table.num_rows, indices=indices) or previous[
                "model_calls"
            ] != len(indices):
                raise ValueError(
                    "original selected positions are incomplete or unordered"
                )
            predictions, references = [], []
            for index in indices:
                case = json.loads(
                    contained(args.reuse, f"{index}.private.json").read_text(
                        encoding="utf-8"
                    )
                )
                row = table.slice(index, 1).to_pylist()[0]
                predictions.append(validate_saved_case(case, row, index))
                references.append(row["summary"])
            scores = original_rouge(
                predictions,
                references,
                source_root=args.source_root,
                output=args.output,
                pins=pins,
            )
            receipt.update(
                status="ORIGINAL_XSUM_SAVED_NATIVE_RESPONSES_SCORED",
                rouge=scores,
                original_indices=indices,
                native_responses_reused=len(indices),
                scoring_complete=True,
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
