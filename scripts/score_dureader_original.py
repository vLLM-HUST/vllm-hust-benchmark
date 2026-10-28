"""Run publisher metrics with an explicit Python3 dictionary-order adapter.

No metric formula, reference normalization or prediction content is changed.
The old converted BLEU wrapper compares key lists instead of key sets: align
prediction dictionary insertion order to the original reference key order.
All upstream software bytes must match the preparation receipt before import.
"""

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys


def load_scorer(root):
    root = Path(root).resolve()
    receipt = json.loads((root / "receipt-public.json").read_text("utf-8"))
    from prepare_dureader_official_scorer import FILES

    if {item["path"] for item in receipt["files"]} != set(FILES):
        raise ValueError("complete pinned scoring software receipt required")
    for item in receipt["files"]:
        path = (root / item["path"]).resolve()
        if (
            not path.is_relative_to(root)
            or hashlib.sha256(path.read_bytes()).hexdigest() != item["installed_sha256"]
        ):
            raise ValueError(
                "publisher scoring software differs from preparation receipt"
            )
    existing = sys.modules.get("dureader_official.dureader_eval")
    if (
        existing
        and Path(existing.__file__).resolve()
        != root / "dureader_official/dureader_eval.py"
    ):
        raise ValueError("different scoring software already imported")
    sys.path.insert(0, str(root))
    return importlib.import_module("dureader_official.dureader_eval")


@contextmanager
def aligned_metric_keys(module):
    original = module.compute_bleu_rouge

    def aligned(predictions, references, bleu_order=4):
        if set(predictions) != set(references):
            raise ValueError("prediction/reference metric key sets differ")
        # Reorder only; original object values, grouping and metric math retained.
        ordered = {key: predictions[key] for key in references}
        return original(ordered, references, bleu_order)

    module.compute_bleu_rouge = aligned
    try:
        yield
    finally:
        module.compute_bleu_rouge = original


def score(module, predictions, references):
    if set(predictions) - set(references):
        raise ValueError("prediction has unknown original question identities")
    with aligned_metric_keys(module):
        return {
            task: module.get_metrics(predictions, references, task, "both")
            for task in ("main", "all", "description", "entity", "yesno")
        }


def metric_report(scores):
    """Encode upstream undefined values as null, never as an invented zero."""
    undefined = []

    def clean(value, path):
        if isinstance(value, dict):
            return {key: clean(item, path + [key]) for key, item in value.items()}
        if isinstance(value, float) and not math.isfinite(value):
            undefined.append("/".join(path))
            return None
        return value

    return {
        "scores": clean(scores, []),
        "undefined_metrics": undefined,
        "undefined_metric_encoding": "null means upstream non-finite/undefined, not zero; original native raw scores remain in historical private receipts",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer-root", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh output file required")
    module = load_scorer(args.scorer_root)
    predictions = module.read_file(str(args.predictions), "all")
    references = module.read_file(str(args.references), "all", is_ref=True)
    result = {
        "metric_key_alignment": "prediction insertion order aligned to original reference key order, no content or formula change",
        **metric_report(score(module, predictions, references)),
        "model_calls": 0,
    }
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
