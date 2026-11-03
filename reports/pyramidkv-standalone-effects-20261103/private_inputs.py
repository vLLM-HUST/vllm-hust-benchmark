"""Verify local official references and rebuild request bytes without publishing them."""

import json
from pathlib import Path

from evidence import digest, read_json, require


def load_references(root: Path, manifest: dict) -> dict:
    references = {}
    for task in ("narrativeqa", "qasper", "2wikimqa"):
        path = root / "data" / f"{task}.jsonl"
        require(path.is_file(), f"Private dataset is missing: {task}")
        content = path.read_bytes()
        require(
            digest(content) == manifest["dataset_files_sha256"][task],
            f"Private dataset digest differs: {task}",
        )
        rows = [read_json(line) for line in content.splitlines() if line.strip()]
        require(len(rows) >= 100, f"Private dataset has fewer than 100 rows: {task}")
        for index, row in enumerate(rows[:100]):
            answers = row.get("answers")
            require(
                isinstance(answers, list)
                and answers
                and all(isinstance(answer, str) for answer in answers),
                f"Private reference answers are invalid: {task}-{index:03d}",
            )
            references[f"{task}-{index:03d}"] = {
                "dataset_id": row.get("_id"),
                "answers": answers,
            }
    return references


def request_bytes(case: dict) -> bytes:
    payload = {
        "model": "qwen35-pyramidkv",
        "prompt": case["prompt"],
        "max_tokens": case["max_tokens"],
        "ignore_eos": case["ignore_eos"],
        "temperature": 0,
        "seed": 17,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if "cache_salt" in case:
        payload["cache_salt"] = case["cache_salt"]
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()
