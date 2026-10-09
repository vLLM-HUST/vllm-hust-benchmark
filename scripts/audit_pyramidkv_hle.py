"""Inventory HLE-Verified's pinned source rows and image requirements only.

No solver prompts, execution subset, judge calls, scores, or rights clearance are
created. The canonical license-review gate remains open.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

REVISION = "b705e0fb541c025a1532ce0d60d70ae2f53b00e0"  # pragma: allowlist secret
SOURCES = {
    "Gold_subset": (
        668,
        "f03c2632400970ffe5167d5ecde169ad38d34ce4dceb2eb8cfb54ece9c5fae00",  # pragma: allowlist secret -- public source SHA-256
    ),
    "Revision_subset": (
        1143,
        "a5a1b1d84edfb90844dfd2f506f331e69d58cb8ef9488ef5f4ebd420fb30cec6",  # pragma: allowlist secret -- public source SHA-256
    ),
    "Uncertain_subset": (
        689,
        "ef6979faa200d8106de1d31567ec7a5c1243eb6bb0b299a2367b1fb4014f1e5a",  # pragma: allowlist secret -- public source SHA-256
    ),
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def summarize_row(row: dict, subset: str, index: int) -> dict:
    for key in ("id", "question", "image", "answer_type"):
        if not isinstance(row.get(key), str):
            raise TypeError(f"Unexpected HLE field type: {key}")
    answer = row.get("answer")
    if type(answer) not in (str, int, bool, float):
        raise TypeError("Unexpected HLE answer type")
    if isinstance(answer, float) and not math.isfinite(answer):
        raise ValueError("Nonfinite HLE answer")
    if not row["id"] or not row["question"] or answer == "":
        raise ValueError("Missing task content")
    # Only problem image fields are solver inputs; rationale images are gold
    # explanation material and must never become solver prompt images.
    preview = row.get("image_preview")
    has_image = bool(row["image"].strip())
    has_preview = preview not in (None, "", {})
    return {
        "id": row["id"],
        "source_subset": subset,
        "source_index": index,
        "verified_class": row["Verified_Classes"],
        "category": row["category"],
        "answer_type": row["answer_type"],
        "source_row_sha256": sha(canonical(row)),
        "question_sha256": sha(row["question"].encode()),
        "answer_sha256": sha(canonical(answer)),
        "answer_value_type": type(answer).__name__,
        "problem_image_present": has_image,
        "problem_image_sha256": sha(row["image"].encode()) if has_image else None,
        "image_preview_present": has_preview,
        "image_preview_sha256": sha(canonical(preview)) if has_preview else None,
        "requires_image_handling": has_image or has_preview,
        "rationale_image_present": bool(row.get("rationale_image")),
    }


def audit(args) -> None:
    inventories = {}
    seen = set()
    for subset, (count, digest) in SOURCES.items():
        path = args.assets / (subset + ".jsonl")
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        if hasher.hexdigest() != digest:
            raise ValueError(f"Source SHA-256 mismatch: {subset}")
        items = []
        with path.open() as stream:
            for index, line in enumerate(stream):
                row = summarize_row(json.loads(line), subset, index)
                if row["id"] in seen:
                    raise ValueError("Duplicate HLE task ID across source subsets")
                seen.add(row["id"])
                items.append(row)
        if len(items) != count:
            raise ValueError("Unexpected HLE subset cardinality")
        inventories[subset] = items
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {
        "repository": "https://github.com/SKYLENAGE-AI/HLE-Verified",
        "revision": REVISION,
        "revision_namespace": "GitHub source commit; not a Hugging Face revision",
        "scope": "source/modality inventory only, no selected execution subset",
        "source_readme_sha256": sha((args.assets / "README.md").read_bytes()),
        "source_tree_sha256": sha((args.assets / "tree.json").read_bytes()),
        "sources": {
            name: {"rows": count, "sha256": digest}
            for name, (count, digest) in SOURCES.items()
        },
        "subsets": {},
        "solver_calls": 0,
        "judge_calls": 0,
        "scores": None,
        "license_review": "open: pinned repository tree has no license file and README makes no explicit license grant",
        "runtime": "current qualified PyramidKV service disables image/video inputs; image-bearing tasks must not be silently dropped or relabeled text-only",
        "execution_subset": None,
        "next_steps": [
            "Complete canonical rights review",
            "Choose named full or Gold execution scope",
            "Pin judge and image-aware solver protocol",
            "Validate multimodal runtime before tasks with images",
        ],
    }
    for subset, items in inventories.items():
        name = subset + "-inventory.jsonl"
        raw = b"".join(canonical(item) + b"\n" for item in items)
        (args.output / name).write_bytes(raw)
        summary["subsets"][subset] = {
            "tasks": len(items),
            "image_bearing": sum(x["requires_image_handling"] for x in items),
            "image_field": sum(x["problem_image_present"] for x in items),
            "preview_only": sum(
                x["image_preview_present"] and not x["problem_image_present"]
                for x in items
            ),
            "answer_types": dict(Counter(x["answer_type"] for x in items)),
            "answer_value_types": dict(Counter(x["answer_value_type"] for x in items)),
            "classes": dict(Counter(x["verified_class"] for x in items)),
            "inventory_sha256": sha(raw),
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["subsets"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args())
