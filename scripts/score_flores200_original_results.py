"""Score saved original FLORES translations offline; no model or network calls."""

import argparse
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path

from run_flores200_original import PINS, original_pairs, verify


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_saved(path, archive, *, source, target, split, dataset):
    """Every saved row must match the unchanged selected original pair."""
    rows = [json.loads(line) for line in Path(path).read_text("utf-8").splitlines()]
    if not rows or any(
        not isinstance(row, dict)
        or set(row) != {"source", "reference", "prediction", "row_index"}
        or type(row["row_index"]) is not int
        or not isinstance(row["prediction"], str)
        for row in rows
    ):
        raise ValueError("original saved translation rows required")
    first = rows[0]["row_index"]
    pairs = original_pairs(
        archive,
        dataset,
        source=source,
        target=target,
        split=split,
        start=first,
        count=len(rows),
        full=False,
    )
    if len(pairs) != len(rows) or any(
        {key: row[key] for key in ("source", "reference", "row_index")} != pair
        for row, pair in zip(rows, pairs, strict=True)
    ):
        raise ValueError("saved output changed original pair/order; no substitution")
    return rows


def calculate(rows, *, chrf, bleu, pieces):
    hypotheses = [row["prediction"] for row in rows]
    references = [row["reference"] for row in rows]
    primary = chrf.corpus_score(hypotheses, [references])
    # The author's SPM pieces are evaluated with no additional tokenization.
    # Record this external-tokenization protocol, not a false tok:flores200 tag.
    secondary = bleu.corpus_score(
        [" ".join(pieces(text)) for text in hypotheses],
        [[" ".join(pieces(text)) for text in references]],
    )
    if any(
        not math.isfinite(item.score) or not 0 <= item.score <= 100
        for item in (primary, secondary)
    ):
        raise ValueError("invalid original metric result")
    return {
        "chrf_plus_plus": primary.score,
        "chrf_signature": str(chrf.get_signature()),
        "spbleu": secondary.score,
        "spbleu_signature_after_external_original_spm": str(bleu.get_signature()),
        "spbleu_tokenization": "original FLORES200 SentencePiece EncodeAsPieces on both sides; sacrebleu tokenize=none",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--spm", type=Path, required=True)
    parser.add_argument("--source-language", required=True)
    parser.add_argument("--target-language", required=True)
    parser.add_argument("--split", choices=["dev", "devtest"], default="devtest")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or any(
        output == path.resolve() or output in path.resolve().parents
        for path in (args.predictions, args.archive, args.spm)
    ):
        raise ValueError("fresh output separate from original inputs required")
    pins = json.loads(PINS.read_text("utf-8"))
    spm_spec = next(
        item
        for item in pins["model"]["files"]
        if item["path"] == "sentencepiece.bpe.model"
    )
    # Author-linked scoring SPM and this publisher model asset have identical
    # observed SHA256/size. Do not accept another model's tokenizer by filename.
    verify(args.spm, spm_spec)
    before = digest(args.predictions)
    rows = load_saved(
        args.predictions,
        args.archive,
        source=args.source_language,
        target=args.target_language,
        split=args.split,
        dataset=pins["dataset"],
    )
    import sentencepiece
    from sacrebleu.metrics import BLEU, CHRF

    spm = sentencepiece.SentencePieceProcessor(model_file=str(args.spm))
    result = calculate(
        rows,
        chrf=CHRF(word_order=2),
        bleu=BLEU(tokenize="none"),
        pieces=spm.EncodeAsPieces,
    )
    verify(args.spm, spm_spec)
    verify(args.archive, pins["dataset"])
    if digest(args.predictions) != before:
        raise ValueError("saved predictions changed while scoring")
    receipt = {
        "source_id": "FLORES200",
        "status": "SAVED_ORIGINAL_TRANSLATIONS_PRIMARY_AND_SPBLEU_SCORED",
        "source_language": args.source_language,
        "target_language": args.target_language,
        "split": args.split,
        "original_positions": [row["row_index"] for row in rows],
        "predictions_sha256": before,
        "spm_sha256": digest(args.spm),
        "versions": {
            name: metadata.version(name) for name in ("sacrebleu", "sentencepiece")
        },
        "new_model_calls": 0,
        "source_checkbox_eligible": False,
        "performance_claim": False,
        **result,
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
