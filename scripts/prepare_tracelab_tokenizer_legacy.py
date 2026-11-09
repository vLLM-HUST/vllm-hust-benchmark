"""Prepare a separate lossless BPE-merge serialization for tokenizers0.19.

The original tokenizer, vocabulary, IDs, merge order and all other fields stay
unchanged. This is not a different tokenizer or a native replay qualification.
Default plan-only; no dependencies, model calls or raw task downloads.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def legacy_view(document):
    model = document.get("model", {})
    if model.get("type") != "BPE" or not isinstance(model.get("merges"), list):
        raise ValueError("BPE model with explicit merges required")
    merges = model["merges"]
    if not merges:
        raise ValueError("nonempty original merge list required")
    pairs = []
    for item in merges:
        pair = item.split(" ") if isinstance(item, str) else item
        if not (
            isinstance(pair, list)
            and len(pair) == 2
            and all(
                isinstance(token, str) and token and not any(c.isspace() for c in token)
                for token in pair
            )
        ):
            raise ValueError("unambiguous two-token merge serialization required")
        pairs.append(pair)
    converted = deepcopy(document)
    converted["model"]["merges"] = [" ".join(pair) for pair in pairs]
    restored = deepcopy(converted)
    restored["model"]["merges"] = deepcopy(merges)
    if restored != document:
        raise ValueError("only merge serialization may change")
    return converted


def prepare(tokenizer, expected_sha256, output, *, write=False):
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("explicit original tokenizer SHA256 required")
    source = Path(tokenizer).resolve(strict=True)
    target = Path(output).resolve()
    if target.exists() or target.is_relative_to(ROOT) or target == source:
        raise ValueError("fresh external compatibility file required")
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("original tokenizer identity mismatch")
    original = json.loads(raw)
    converted = legacy_view(original)
    encoded = (json.dumps(converted, ensure_ascii=False) + "\n").encode("utf-8")
    report = {
        "source_id": "TRACELAB",
        "status": "LEGACY_SERIALIZATION_PREPARED_NOT_RUNTIME_QUALIFIED"
        if write
        else "PLAN_ONLY",
        "original_tokenizer_sha256": expected_sha256,
        "compatibility_tokenizer_sha256": hashlib.sha256(encoded).hexdigest(),
        "merge_count": len(original["model"]["merges"]),
        "only_merge_serialization_changed": True,
        "vocabulary_ids_and_merge_order_preserved": True,
        "runtime_encoding_equivalence_verified": False,
        "native_cache_probe_or_replay_executed": False,
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }
    if write:
        # Parent is supplied by the operator; never create broad output trees.
        with target.open("xb") as stream:
            stream.write(encoded)
    if source.read_bytes() != raw:
        raise ValueError("original tokenizer changed during preparation")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            prepare(
                args.tokenizer, args.expected_sha256, args.output, write=args.write
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
