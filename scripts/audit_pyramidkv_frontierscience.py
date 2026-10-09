"""Freeze input-only FrontierScience manifests; never infer scores or readiness."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

REVISION = "25ed67db7da8f4591484e764008ff585544f5a30"  # pragma: allowlist secret
SOURCES = {
    "olympiad": (
        100,
        "450e7fceb53c8140283a752f0d62a3abc69a85e252d4dbbd5d9dfbfb762db4e4",  # pragma: allowlist secret -- public dataset SHA-256
    ),
    "research": (
        60,
        "96c0434abfcbadd6ef6f59a03cc374be4caf9c1f2d5e62d8fe921e768f66aa46",  # pragma: allowlist secret -- public dataset SHA-256
    ),
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked_rows(source: Path, track: str) -> list[dict]:
    expected_count, expected_hash = SOURCES[track]
    data = source.read_bytes()
    if sha(data) != expected_hash:
        raise ValueError(f"Source SHA-256 mismatch: {track}")
    rows = [json.loads(line) for line in data.splitlines()]
    if len(rows) != expected_count:
        raise ValueError(f"Unexpected task count: {track}")
    return rows


def audit(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    # Validate both inputs before writing a candidate manifest.
    sources = {
        track: checked_rows(args.assets / track / "test.jsonl", track)
        for track in SOURCES
    }
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_version": 1,
        "status": "input-contract-only; scorer and execution contract blocked",
        "dataset": "openai/frontierscience",
        "revision": REVISION,
        "template": "Qwen user message containing only source problem; no demonstrations",
        "enable_thinking": False,
        "add_generation_prompt": True,
        "add_special_tokens": False,
        "threshold": 4096,
        "candidate_only": True,
        "model_revision": args.model_revision,
        "tokenizer_files": {
            path.name: sha(path.read_bytes())
            for path in sorted(Path(args.model).glob("*"))
            if path.name
            in {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}
        },
        "source_answer_used_in_prompt": False,
        "solver_calls": 0,
        "judge_calls": 0,
        "scores": None,
        "tracks": {},
    }
    for track, rows in sources.items():
        inventory = []
        for index, row in enumerate(rows):
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["problem"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
            inventory.append(
                {
                    "task_id": f"{track}/test/{index:03d}",
                    "track": track,
                    "source_index": index,
                    "task_group_id": row["task_group_id"],
                    "subject": row["subject"],
                    "source_row_sha256": sha(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                    ),
                    "problem_sha256": sha(row["problem"].encode()),
                    "reference_sha256": sha(row["answer"].encode()),
                    "rendered_prompt_sha256": sha(rendered.encode()),
                    "prompt_tokens": tokens,
                    "prefill_threshold_exceeded": tokens > 4096,
                }
            )
        content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in inventory)
        name = f"{track}-inventory.jsonl"
        (args.output / name).write_text(content)
        lengths = [row["prompt_tokens"] for row in inventory]
        summary["tracks"][track] = {
            "tasks": len(rows),
            "unique_task_group_ids": len({row["task_group_id"] for row in rows}),
            "min_prompt_tokens": min(lengths),
            "max_prompt_tokens": max(lengths),
            "threshold_exceeded": sum(n > 4096 for n in lengths),
            "subjects": dict(sorted(Counter(row["subject"] for row in rows).items())),
            "source_sha256": SOURCES[track][1],
            "inventory_sha256": sha(content.encode()),
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["tracks"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args())
