"""Author mT5 model-card generation and unmodified multilingual ROUGE on NPU.

The publisher's 512-token encoder policy is part of this short-input protocol,
not evidence of long-context serving. Original JSONL and generated text remain
private. No Qwen, prompt rewriting, English-only scorer or CPU model fallback.
"""

import argparse
import atexit
import ast
import hashlib
import importlib
from importlib import metadata
import json
from logging import getLogger
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Dict, List


PINS = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/xlsum-author-pins.json"
)
PROTOCOL = {
    "encoder_max_length": 512,
    "decoder_max_length": 84,
    "num_beams": 4,
    "no_repeat_ngram_size": 2,
    "length_penalty": 0.6,
    "tie_word_embeddings": False,
    "dtype": "torch.float32",
    "tokenizer": "T5Tokenizer",
    "tokenizer_legacy": True,
    "seed": 1234,
}


class OriginalTokenizer:
    """Run the original slow tokenizer in its own tokenizer-only environment.

    The NPU model stays in the compatible modern Torch/Transformers process.
    JSON pipes are local IPC, not a network service. No model loads in worker.
    """

    def __init__(self, python, model_root):
        self.process = subprocess.Popen(
            [
                str(python),
                "-I",
                "-u",
                str(Path(__file__).resolve()),
                "--tokenizer-worker",
                "--model-root",
                str(model_root),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        atexit.register(self.close)
        handshake = json.loads(self.process.stdout.readline())
        if handshake != {"transformers": "4.11.3", "tokenizer_class": "T5Tokenizer"}:
            self.close()
            raise ValueError("original tokenizer-only environment required")

    def call(self, request):
        self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        response = json.loads(self.process.stdout.readline())
        if "error_type" in response:
            raise ValueError(
                "original tokenizer worker rejected request: " + response["error_type"]
            )
        return response

    def __call__(self, inputs, **kwargs):
        tensor = kwargs.pop("return_tensors", None)
        result = self.call({"method": "encode", "inputs": inputs, "kwargs": kwargs})
        if tensor is not None:
            if tensor != "pt":
                raise ValueError("only original PyTorch input representation supported")
            import torch

            result["input_ids"] = torch.tensor(result["input_ids"], dtype=torch.long)
        return result

    def decode(self, ids, **kwargs):
        return self.call({"method": "decode", "ids": ids, "kwargs": kwargs})["text"]

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)


class CacheTokenizationMismatch(ValueError):
    """Same raw task, but cached model input differs from original tokenization."""


def tokenizer_worker(model_root):
    from transformers import T5Tokenizer

    if metadata.version("transformers") != "4.11.3":
        raise ValueError("original 4.11.3 tokenizer required")
    tokenizer = T5Tokenizer.from_pretrained(str(model_root), local_files_only=True)
    print(
        json.dumps(
            {"transformers": "4.11.3", "tokenizer_class": type(tokenizer).__name__}
        ),
        flush=True,
    )
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request["method"] == "encode":
                result = {
                    "input_ids": tokenizer(request["inputs"], **request["kwargs"])[
                        "input_ids"
                    ]
                }
            elif request["method"] == "decode":
                result = {"text": tokenizer.decode(request["ids"], **request["kwargs"])}
            else:
                raise ValueError("unsupported tokenizer operation")
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps({"error_type": type(exc).__name__}), flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_files(root, files):
    root = Path(root).resolve()
    for spec in files:
        path = (root / spec["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("missing or uncontained original asset")
        if path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]:
            raise ValueError("original asset identity mismatch")


def normalized_article(text):
    return re.sub(r"\s+", " ", re.sub(r"\n+", " ", text.strip()))


def original_scoring(source_root, pins):
    """Import verified scorer; extract original metric functions, not training."""
    check_files(source_root, pins["source_files"])
    from rouge_score import rouge_scorer, scoring
    import nltk

    installed = Path(rouge_scorer.__file__).parent
    for spec in pins["source_files"]:
        relative = Path(spec["path"])
        if (
            relative.parent.name == "multilingual_rouge_scoring"
            and relative.suffix == ".py"
            and relative.name != "setup.py"
        ):
            if sha256(installed / relative.name) != spec["sha256"]:
                raise ValueError(
                    "installed scorer is not the pinned author's multilingual version"
                )
    sentence_module = ast.parse(
        (Path(source_root) / "seq2seq/sentence_splitter.py").read_text()
    )
    sentence_functions = [
        n for n in sentence_module.body if isinstance(n, ast.FunctionDef)
    ]
    namespace = {"re": re, "nltk": nltk, "NLTK_AVAILABLE": True}
    exec(
        compile(
            ast.Module(body=sentence_functions, type_ignores=[]),
            "original-sentence-splitter",
            "exec",
        ),
        namespace,
    )
    utils = ast.parse((Path(source_root) / "seq2seq/utils.py").read_text())
    selected = []
    for node in utils.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "ROUGE_KEYS" for t in node.targets
        ):
            selected.append(node)
        if isinstance(node, ast.FunctionDef) and node.name in {
            "extract_rouge_mid_statistics",
            "calculate_rouge",
        }:
            selected.append(node)
    if len(selected) != 3:
        raise ValueError("original scoring functions missing")
    namespace.update(
        Dict=Dict,
        List=List,
        rouge_scorer=rouge_scorer,
        scoring=scoring,
        logger=getLogger("xlsum-original"),
    )
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]), "original-xlsum-metrics", "exec"
        ),
        namespace,
    )
    # Fail before any generation if sentence data or English stopwords is absent.
    nltk.sent_tokenize("Dependency preflight.")
    nltk.corpus.stopwords.words("english")
    return namespace["calculate_rouge"]


def preflight_cache(root, *, row, offset, tokenizer, pins):
    """Reuse the audited corrected English preflight, never its incompatible run."""
    root = Path(root)
    receipt = json.loads((root / "receipt-public.json").read_text())
    if not (
        receipt["status"]
        == "PUBLISHER_MODEL_CARD_NPU_GENERATION_COMPLETED_SCORING_PENDING"
        and receipt["language_variant"] == "english"
        and receipt["model_revision"] == pins["model_revision"]
        and receipt.get("author_tie_word_embeddings") is False
        and receipt.get("author_length_penalty") == PROTOCOL["length_penalty"]
        and receipt.get("original_411_tokenizer_inputs_verified") is True
        and receipt["versions"]["transformers"] == "4.57.3"
        and receipt["publisher_encoder_max_length"] == 512
        and receipt["publisher_decoder_max_length"] == 84
        and receipt["publisher_num_beams"] == 4
        and receipt["publisher_no_repeat_ngram_size"] == 2
        and receipt["model_dtype"] == "torch.float32"
        and receipt["model_parameters_on_npu"] is True
    ):
        raise ValueError("incompatible preflight evidence; do not reuse")
    records = [s for s in receipt["samples"] if s["original_offset"] == offset]
    if not records:
        return None
    (sample,) = records
    path = root / f"row-{offset}.private.json"
    if sha256(path) != sample["private_record_sha256"]:
        raise ValueError("private preflight evidence changed")
    record = json.loads(path.read_text())
    normalized = normalized_article(row["text"])
    encoded = tokenizer(
        [normalized], padding="max_length", truncation=True, max_length=512
    )["input_ids"]
    if (
        record["original_row"] != row
        or record["publisher_normalized_article"] != normalized
        or record["encoder_input_ids"] != encoded
    ):
        raise ValueError("preflight does not match the original task/input")
    decoded = tokenizer.decode(
        record["generated_token_ids"],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if decoded != record["summary"] or len(record["generated_token_ids"]) > 84:
        raise ValueError("preflight output decoding mismatch")
    return record


def native_cache(root, *, asset, row, offset, tokenizer, pins, data_revision):
    """Reuse completed rows, including those preceding a terminal runtime failure."""
    root = Path(root)
    receipt = json.loads((root / "receipt-public.json").read_text())
    if not (
        receipt.get("source_id") == "XLSUM"
        and receipt.get("model_revision") == pins["model_revision"]
        and receipt.get("source_revision") == pins["source_revision"]
        and receipt.get("data_revision") == data_revision
        and receipt.get("protocol") == PROTOCOL
        and receipt.get("versions", {}).get("transformers") == "4.57.3"
    ):
        raise ValueError("incompatible native cache protocol")
    matched = [
        r
        for r in receipt["languages"]
        if r["language_variant"] == asset["language_variant"]
    ]
    if not matched:
        return None
    (language,) = matched
    if language["data_sha256"] != asset["sha256"]:
        raise ValueError("native cache original data changed")
    samples = [s for s in language["samples"] if s["original_offset"] == offset]
    if not samples:
        return None
    (sample,) = samples
    path = root / f"{asset['language_variant']}-{offset}.private.json"
    if sha256(path) != sample["private_record_sha256"]:
        raise ValueError("native cache private evidence changed")
    record = json.loads(path.read_text())
    normalized = normalized_article(row["text"])
    if (
        record["original_row"] != row
        or record["publisher_normalized_article"] != normalized
    ):
        raise ValueError("native cache does not match original request")
    ids = tokenizer(
        [normalized], padding="max_length", truncation=True, max_length=512
    )["input_ids"]
    decoded = tokenizer.decode(
        record["generated_token_ids"],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if record["encoder_input_ids"] != ids:
        raise CacheTokenizationMismatch(
            "cached input differs from original tokenizer; unchanged task needs regeneration"
        )
    if not (
        record["original_row"] == row
        and record["publisher_normalized_article"] == normalized
        and record["encoder_input_ids"] == ids
        and len(record["generated_token_ids"]) <= 84
    ):
        raise ValueError("native cache does not match original request or output")
    if record["summary"] != decoded:
        record["prior_decoder_summary_sha256"] = hashlib.sha256(
            record["summary"].encode()
        ).hexdigest()
        record["summary"] = decoded
        record["redecoded_from_saved_generated_token_ids"] = True
    return record


def run(args):
    from vllm_hust_workload_data.registry import source_spec
    from vllm_hust_workload_data.xlsum import original_rows

    pins = json.loads(PINS.read_text())
    source = source_spec("XLSUM")
    if args.output.exists():
        raise ValueError("fresh private output directory required")
    languages = set(args.language)
    known_languages = {
        f["language_variant"] for f in source["files"] if f["role"] == "data"
    }
    if languages - known_languages - {"all"}:
        raise ValueError("unknown original language variant")
    selected_files = [
        f
        for f in source["files"]
        if f["role"] == "data"
        and ("all" in languages or f["language_variant"] in languages)
    ]
    if not selected_files:
        raise ValueError("unknown original language variant")
    check_files(args.data_root, selected_files)
    check_files(args.model_root, pins["model_files"])
    calculate_rouge = original_scoring(args.source_root, pins)
    # Check every requested scorer before spending any NPU work.
    scorer_module = importlib.import_module("rouge_score.rouge_scorer")
    for asset in selected_files:
        scorer_module.RougeScorer(
            ["rouge1", "rouge2", "rougeL", "rougeLsum"],
            use_stemmer=True,
            lang=asset["language_variant"],
        )
    import numpy as np
    import torch
    import torch_npu  # noqa: F401 - register NPU backend
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, set_seed

    if metadata.version("transformers") != "4.57.3":
        raise ValueError(
            "use the declared isolated compatible Transformers 4.57.3 runtime"
        )
    config = AutoConfig.from_pretrained(args.model_root, local_files_only=True)
    if config.tie_word_embeddings is not False or config.length_penalty != 0.6:
        raise ValueError("runtime changed publisher model configuration")
    tokenizer = OriginalTokenizer(args.tokenizer_python, args.model_root)
    set_seed(1234)
    np.random.seed(1234)
    torch.npu.set_device(args.device)
    model = None
    args.output.mkdir(parents=True, mode=0o700)
    receipt = {
        "source_id": "XLSUM",
        "status": "RUNNING",
        "source_revision": pins["source_revision"],
        "data_revision": source["revision"],
        "model_revision": pins["model_revision"],
        "protocol": PROTOCOL,
        "selection": "full"
        if args.full
        else "first-three-physical-rows-before-outputs",
        "versions": {
            k: metadata.version(k)
            for k in ["transformers", "torch", "torch-npu", "nltk", "pyonmttok"]
        },
        "languages": [],
        "scoring_verified": False,
        "bootstrap_seed_per_language": 1234,
        "tokenizer_transformers": "4.11.3",
        "tokenizer_runs_in_model_free_local_worker": True,
        "source_checkbox_eligible": False,
        "raw_payloads_published": False,
        "started_at_unix": time.time(),
    }

    def save():
        (args.output / "receipt-public.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )

    save()
    try:
        for asset in selected_files:
            language = asset["language_variant"]
            # Construct the original scorer before any generation for this language.
            scorer_module = importlib.import_module("rouge_score.rouge_scorer")
            scorer_module.RougeScorer(
                ["rouge1", "rouge2", "rougeL", "rougeLsum"],
                use_stemmer=True,
                lang=language,
            )
            rows = list(original_rows(args.data_root / asset["path"]))
            if len(rows) != asset["rows"]:
                raise ValueError("original test row count mismatch")
            chosen = rows if args.full else rows[:3]
            result = {
                "language_variant": language,
                "data_sha256": asset["sha256"],
                "original_rows": len(rows),
                "selected_offsets": list(range(len(chosen))),
                "completed_rows": 0,
                "fresh_model_calls": 0,
                "reused_rows": 0,
                "samples": [],
                "scoring_verified": False,
            }
            receipt["languages"].append(result)
            save()
            predictions, targets = [], []
            for offset, row in enumerate(chosen):
                record = None
                for root in args.reuse:
                    try:
                        candidate = native_cache(
                            root,
                            asset=asset,
                            row=row,
                            offset=offset,
                            tokenizer=tokenizer,
                            pins=pins,
                            data_revision=source["revision"],
                        )
                    except CacheTokenizationMismatch:
                        if not args.rerun_tokenizer_mismatch:
                            raise
                        result.setdefault("invalidated_cache_samples", []).append(
                            {
                                "original_offset": offset,
                                "prior_run": root.name,
                                "reason": "same raw task but original-tokenizer encoder IDs differ; preserve evidence and rerun unchanged task",
                            }
                        )
                        candidate = None
                    if candidate is not None:
                        if record is not None and record != candidate:
                            raise ValueError("conflicting completed native evidence")
                        record = candidate
                if record is None and args.reuse_preflight and language == "english":
                    record = preflight_cache(
                        args.reuse_preflight,
                        row=row,
                        offset=offset,
                        tokenizer=tokenizer,
                        pins=pins,
                    )
                if record is not None:
                    result["reused_rows"] += 1
                else:
                    if model is None:
                        free, _ = torch.npu.mem_get_info()
                        if free < 8 * 1024**3:
                            raise RuntimeError(
                                "insufficient free NPU memory; do not stop or reconfigure other services"
                            )
                        model = AutoModelForSeq2SeqLM.from_pretrained(
                            args.model_root,
                            local_files_only=True,
                            dtype=torch.float32,
                            attn_implementation="eager",
                        )
                        model.eval().to(f"npu:{args.device}")
                        if not all(p.device.type == "npu" for p in model.parameters()):
                            raise RuntimeError("model is not entirely on NPU")
                    normalized = normalized_article(row["text"])
                    encoded = tokenizer(
                        [normalized],
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=512,
                    )["input_ids"]
                    generation_started = time.monotonic()
                    with torch.inference_mode():
                        generated = model.generate(
                            input_ids=encoded.to(f"npu:{args.device}"),
                            max_length=84,
                            no_repeat_ngram_size=2,
                            num_beams=4,
                        )[0]
                    torch.npu.synchronize()
                    generation_seconds = time.monotonic() - generation_started
                    if generated.device.type != "npu":
                        raise RuntimeError("no CPU generation fallback allowed")
                    tokens = generated.tolist()
                    summary = tokenizer.decode(
                        tokens,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    record = {
                        "original_row": row,
                        "publisher_normalized_article": normalized,
                        "encoder_input_ids": encoded.tolist(),
                        "generated_token_ids": tokens,
                        "summary": summary,
                    }
                    result["fresh_model_calls"] += 1
                    # Wall time is for full-vs-sample scheduling only, not a performance claim.
                    result.setdefault("fresh_generation_seconds", []).append(
                        generation_seconds
                    )
                path = args.output / f"{language}-{offset}.private.json"
                path.write_text(json.dumps(record, ensure_ascii=False) + "\n")
                predictions.append(record["summary"])
                targets.append(row["summary"])
                result["samples"].append(
                    {"original_offset": offset, "private_record_sha256": sha256(path)}
                )
                result["completed_rows"] += 1
                save()
            np.random.seed(1234)
            result["rouge"] = calculate_rouge(predictions, targets, rouge_lang=language)
            result["scoring_verified"] = True
            save()
        receipt["scoring_verified"] = True
        receipt["status"] = "ORIGINAL_MODEL_CARD_GENERATION_AND_AUTHOR_ROUGE_COMPLETED"
    except Exception as exc:
        receipt.update(
            status="FAILED_NO_REPLACEMENT_TASK",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    finally:
        receipt["finished_at_unix"] = time.time()
        save()
        tokenizer.close()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--tokenizer-worker":
        worker = argparse.ArgumentParser()
        worker.add_argument("--tokenizer-worker", action="store_true")
        worker.add_argument("--model-root", type=Path, required=True)
        tokenizer_worker(worker.parse_args().model_root)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["source-root", "model-root", "data-root", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--language",
        required=True,
        action="append",
        help="original variant name (repeatable) or all",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="all original test rows of selected variants",
    )
    parser.add_argument(
        "--reuse-preflight", type=Path, help="audited corrected English preflight only"
    )
    parser.add_argument(
        "--reuse",
        type=Path,
        action="append",
        default=[],
        help="completed native output root; checked before any new request",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--rerun-tokenizer-mismatch",
        action="store_true",
        help="explicitly regenerate the unchanged original task only when verified prior cache used different encoder IDs; never ignores corruption",
    )
    parser.add_argument(
        "--tokenizer-python",
        type=Path,
        required=True,
        help="Python executable in the original 4.11.3 tokenizer-only environment",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
