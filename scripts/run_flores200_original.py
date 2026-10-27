"""Original FLORES-200 parallel sentences and author primary chrF++ metric.

Server-only NLLB/NPU execution. Summary receipts never imply every language is
supported or the entire source is delivered. No language/script substitution.
"""

import argparse
from importlib import metadata
import json
import math
from pathlib import Path, PurePosixPath
import re
import tarfile

from prepare_flores200_original_assets import PINS, inspect_archive, verify


def original_pairs(
    archive, dataset, *, source, target, split, start=0, count=3, full=False
):
    if (
        split not in dataset["split_rows"]
        or source == target
        or any(
            not re.fullmatch(r"[a-z]{3}_[A-Z][a-z]{3}", code)
            for code in (source, target)
        )
    ):
        raise ValueError("distinct original language codes and public split required")
    if full and (start != 0 or count != 3):
        raise ValueError("full selection cannot combine with range")
    total = dataset["split_rows"][split]
    if not full and (start < 0 or count < 3 or start + count > total):
        raise ValueError("preselect at least three original positions within split")
    verify(Path(archive), dataset)
    inspect_archive(Path(archive), dataset)
    values = []
    with tarfile.open(archive) as bundle:
        members = {
            str(PurePosixPath(member.name)): member
            for member in bundle.getmembers()
            if member.isfile()
        }
        for code in (source, target):
            member = members[f"flores200_dataset/{split}/{code}.{split}"]
            lines = bundle.extractfile(member).read().decode("utf-8").split("\n")
            if lines[-1] == "":
                lines.pop()
            if len(lines) != total:
                raise ValueError("original parallel line count mismatch")
            values.append(lines)
    positions = range(total) if full else range(start, start + count)
    return [
        {"row_index": i, "source": values[0][i], "reference": values[1][i]}
        for i in positions
    ]


def execute(pairs, *, translate, metric, output, source, target, split):
    if len(pairs) < 3 or len({pair["row_index"] for pair in pairs}) != len(pairs):
        raise ValueError(
            "at least three distinct preselected original positions required"
        )
    output = Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    receipt = {
        "status": "RUNNING_NOT_COMPLETE",
        "source_language": source,
        "target_language": target,
        "split": split,
        "selected_original_positions": [p["row_index"] for p in pairs],
        "completed_original_positions": [],
        "metric": "author_primary_chrf++_word_order_2",
        "native_execution_complete": False,
        "source_checkbox_eligible": False,
        "language_substitution": False,
        "truncation": False,
        "performance_claim": False,
    }

    def save():
        (output / "receipt-private.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )

    save()
    try:
        hypotheses, references = [], []
        with (output / "predictions-private.jsonl").open(
            "x", encoding="utf-8"
        ) as stream:
            for pair in pairs:
                prediction = translate(pair["source"])
                if not isinstance(prediction, str):
                    raise ValueError("translation output must be a string")
                hypotheses.append(prediction)
                references.append(pair["reference"])
                stream.write(
                    json.dumps({**pair, "prediction": prediction}, ensure_ascii=False)
                    + "\n"
                )
                stream.flush()
                receipt["completed_original_positions"].append(pair["row_index"])
                save()
        score = metric.corpus_score(hypotheses, [references])
        if not math.isfinite(score.score) or not 0 <= score.score <= 100:
            raise ValueError("invalid author primary metric result")
        receipt.update(
            status="SELECTED_ORIGINAL_PAIR_AND_AUTHOR_PRIMARY_METRIC_COMPLETE",
            chrf_plus_plus=score.score,
            metric_signature=str(metric.get_signature()),
            native_execution_complete=True,
        )
    except Exception as exc:
        receipt.update(status="FAILED_NOT_COMPLETE", error_type=type(exc).__name__)
        raise
    finally:
        save()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-language", required=True)
    parser.add_argument("--target-language", required=True)
    parser.add_argument("--split", choices=["dev", "devtest"], default="devtest")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--npu-device", type=int, required=True)
    args = parser.parse_args()
    pins = json.loads(PINS.read_text("utf-8"))
    if any(
        code in pins["model"]["original_codes_without_language_token"]
        for code in (args.source_language, args.target_language)
    ):
        raise ValueError("model lacks original language/script token; no substitution")
    output, model_root = args.output.resolve(), args.model_root.resolve()
    if (
        output.exists()
        or output.is_relative_to(model_root)
        or model_root.is_relative_to(output)
    ):
        raise ValueError("fresh private output separate from original model required")
    for item in pins["model"]["files"]:
        verify(model_root / item["path"], item)
    pairs = original_pairs(
        args.archive,
        pins["dataset"],
        source=args.source_language,
        target=args.target_language,
        split=args.split,
        start=args.start,
        count=args.count,
        full=args.full,
    )
    import torch
    import torch_npu  # noqa: F401 - registers NPU backend
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from sacrebleu.metrics import CHRF

    if args.npu_device < 0 or not torch.npu.is_available():
        raise RuntimeError("available NPU required; no CPU/CUDA fallback")
    tokenizer = AutoTokenizer.from_pretrained(
        model_root,
        local_files_only=True,
        trust_remote_code=False,
        src_lang=args.source_language,
    )
    for code in (args.source_language, args.target_language):
        if code not in tokenizer.all_special_tokens:
            raise ValueError(
                "model lacks original language/script token; no substitution"
            )
    device = f"npu:{args.npu_device}"
    model = (
        AutoModelForSeq2SeqLM.from_pretrained(
            model_root, local_files_only=True, trust_remote_code=False
        )
        .to(device)
        .eval()
    )
    target_id = tokenizer.convert_tokens_to_ids(args.target_language)

    def translate(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=False)
        if inputs["input_ids"].shape[-1] > model.config.max_position_embeddings:
            raise ValueError(
                "complete original sentence exceeds model context; no truncation"
            )
        with torch.inference_mode():
            result = model.generate(
                **{key: value.to(device) for key, value in inputs.items()},
                forced_bos_token_id=target_id,
            )
        return tokenizer.batch_decode(result, skip_special_tokens=True)[0]

    receipt = execute(
        pairs,
        translate=translate,
        metric=CHRF(word_order=2),
        output=output,
        source=args.source_language,
        target=args.target_language,
        split=args.split,
    )
    try:
        verify(args.archive, pins["dataset"])
        for item in pins["model"]["files"]:
            verify(model_root / item["path"], item)
    except Exception:
        receipt.update(
            status="FINAL_IDENTITY_RECHECK_FAILED", native_execution_complete=False
        )
        (output / "receipt-private.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        raise
    receipt.update(
        model_repo=pins["model"]["repo_id"],
        model_revision=pins["model"]["revision"],
        noncommercial_research_only=True,
        generation_config=model.generation_config.to_dict(),
        versions={
            name: metadata.version(name)
            for name in (
                "transformers",
                "sacrebleu",
                "torch",
                "torch-npu",
                "sentencepiece",
            )
        },
    )
    (output / "receipt-private.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
