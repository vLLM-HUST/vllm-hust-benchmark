"""Verify local official references and rebuild request bytes without publishing them."""

import json
import re
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


CASE_METADATA = {
    "case_id",
    "kind",
    "task",
    "dataset_id",
    "subset",
    "original_instruction_tokens",
    "truncated",
    "max_tokens",
    "ignore_eos",
    "prompt_tokens",
    "prompt_sha256",
    "cache_salt",
    "require_uncached",
}


def public_plan(cases: list[dict]) -> tuple[list[dict], list[dict]]:
    index, synthetic, seen = [], [], set()
    for case in cases:
        key = case["case_id"]
        require(
            isinstance(key, str)
            and re.fullmatch(r"[A-Za-z0-9_-]+", key)
            and key not in seen,
            "Private plan has duplicate or unsafe case IDs",
        )
        seen.add(key)
        quality = case.get("kind") == "quality"
        allowed = CASE_METADATA | {"prompt"} | ({"answers"} if quality else set())
        require(
            not set(case) - allowed,
            "Private plan has unrecognized or misplaced reference fields",
        )
        require(
            case.get("kind")
            in {
                "quality",
                "warmup",
                "boundary",
                "performance",
                "capacity",
                "repeatability",
            },
            "Private plan has an unknown case kind",
        )
        require(
            isinstance(case.get("prompt"), list)
            and case["prompt"]
            and all(type(token) is int and token >= 0 for token in case["prompt"]),
            "Private plan prompt must contain token IDs",
        )
        if quality:
            require(
                isinstance(case.get("answers"), list)
                and case["answers"]
                and all(isinstance(answer, str) for answer in case["answers"]),
                "Private plan reference answers are missing or invalid",
            )
        index.append(
            {field: value for field, value in case.items() if field in CASE_METADATA}
        )
        if not quality:
            synthetic.append(case.copy())
    return index, synthetic


def reconstruct_cases(
    index: list[dict],
    synthetic: list[dict],
    quality_cases: list[dict],
    expected_sha256: str,
) -> list[dict]:
    private = [case for case in quality_cases if case["kind"] == "quality"]
    combined = [*private, *synthetic]
    lookup = {case["case_id"]: case for case in combined}
    require(
        len(lookup) == len(combined), "Private plan reconstruction has duplicate cases"
    )
    require(
        all(row["case_id"] in lookup for row in index),
        "Private plan reconstruction lacks cases",
    )
    require(
        {case["case_id"] for case in synthetic}
        == {row["case_id"] for row in index if row["kind"] != "quality"},
        "Private plan synthetic membership differs",
    )
    cases = []
    for row in index:
        case = lookup[row["case_id"]].copy()
        if case["kind"] == "quality":
            # Service smoke retains the original case's API salt. Prompt,
            # references and generation limits still come from the pinned QA plan.
            for field in ("cache_salt", "require_uncached"):
                if field in row:
                    case[field] = row[field]
        cases.append(case)
    actual_index, _ = public_plan(cases)
    require(
        actual_index == index, "Private plan metadata differs from the public index"
    )
    content = (json.dumps(cases, ensure_ascii=False, indent=2) + "\n").encode()
    require(
        digest(content) == expected_sha256,
        "Private plan case digest differs from the frozen original",
    )
    return cases
