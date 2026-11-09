"""Summarize paired short-answer quality, reuse sources and timing receipts."""

import argparse
import gzip
import json
import re
import statistics
from pathlib import Path

REUSE = re.compile(
    r"SimLLM: reusing (?P<skipped>\d+) semantic tokens for request "
    r"(?P<target>\S+) \(source=(?P<source>\S+), cosine=(?P<cosine>[0-9.]+)\)"
)
ARMS = ("baseline", "approximate", "chunked_baseline", "native_apc", "rollback")
CHOICES = {
    "number": ("17", "29", "43"),
    "negation": ("允许", "禁止"),
    "subject": ("张岚", "李诚", "王宁"),
    "instruction": ("A", "B", "C"),
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def resolve(server_id: str, rows: dict[str, dict]) -> str:
    candidates = [
        row["id"] for rid, row in rows.items() if server_id.startswith(rid + "-")
    ]
    if len(candidates) != 1:
        raise ValueError(f"Server ID resolution failed: {server_id}")
    return candidates[0]


def extract_answer(text: str, kind: str) -> str | None:
    cleaned = text.strip().upper() if kind == "instruction" else text.strip()
    if kind == "negation":
        cleaned = cleaned.replace("不允许", "禁止")
    if kind == "instruction":
        hits = set(re.findall(r"(?<![A-Z])[ABC](?![A-Z])", cleaned))
    else:
        hits = {choice for choice in CHOICES[kind] if choice in cleaned}
    return next(iter(hits)) if len(hits) == 1 else None


def summarize(root: Path, workload: Path) -> dict:
    inputs = load_jsonl(workload)
    if len(inputs) != 12 or len({row["id"] for row in inputs}) != 12:
        raise ValueError("Expected 12 unique workload requests")
    by_id = {row["id"]: row for row in inputs}
    outputs: dict[str, dict[str, dict]] = {}
    summaries: dict[str, dict] = {}
    for arm in ARMS:
        rows = load_jsonl(root / arm / "outputs.jsonl")
        summaries[arm] = next(row for row in rows if row.get("phase") == "summary")
        items = [row for row in rows if row.get("phase") != "summary"]
        if len(items) != 12 or {row["id"] for row in items} != set(by_id):
            raise ValueError(f"Incomplete or mismatched arm: {arm}")
        outputs[arm] = {row["id"]: row for row in items}

    by_request = {row["request_id"]: row for row in outputs["approximate"].values()}
    reuse = {}
    with gzip.open(root / "approximate/server.log.gz", "rt", errors="replace") as log:
        lines = log.readlines()
    for line in lines:
        match = REUSE.search(line)
        if not match:
            continue
        event = match.groupdict()
        target = resolve(event["target"], by_request)
        source = resolve(event["source"], by_request)
        if target in reuse:
            raise ValueError(f"Duplicate reuse event: {target}")
        reuse[target] = {
            "source_id": source,
            "skipped_tokens": int(event["skipped"]),
            "cosine": float(event["cosine"]),
        }

    comparisons = []
    for item in inputs:
        rid = item["id"]
        entry = {
            "id": rid,
            "change_kind": item["change_kind"],
            "variant": item["variant"],
            "expected": item["expected"],
            "reuse": reuse.get(rid),
            "arms": {},
        }
        for arm in ARMS:
            row = outputs[arm][rid]
            answer = extract_answer(row["output_text"], item["change_kind"])
            entry["arms"][arm] = {
                "output_text": row["output_text"],
                "extracted_answer": answer,
                "correct": answer == item["expected"],
                "ttft_s": row["ttft_s"],
                "latency_s": row["latency_s"],
                "prompt_tokens": row["prompt_tokens"],
                "completion_tokens": row["completion_tokens"],
            }
        comparisons.append(entry)

    return {
        "scope": "12-request exploratory paired diagnostic; not a formal quality or throughput benchmark",
        "stages": json.loads((root / "summary.json").read_text()),
        "request_count": len(comparisons),
        "reuse_hits": len(reuse),
        "reuse_from_prior_reuse_target": sum(
            value["source_id"] in reuse for value in reuse.values()
        ),
        "reused_prompt_tokens": sum(
            value["skipped_tokens"] for value in reuse.values()
        ),
        "correct_by_arm": {
            arm: sum(row["arms"][arm]["correct"] for row in comparisons) for arm in ARMS
        },
        "output_equal_baseline_by_arm": {
            arm: sum(
                row["arms"][arm]["output_text"]
                == row["arms"]["baseline"]["output_text"]
                for row in comparisons
            )
            for arm in ARMS
        },
        "apc_equal_chunked_baseline": sum(
            row["arms"]["native_apc"]["output_text"]
            == row["arms"]["chunked_baseline"]["output_text"]
            for row in comparisons
        ),
        "client_summaries": summaries,
        "median_ttft_s_by_arm": {
            arm: statistics.median(row["arms"][arm]["ttft_s"] for row in comparisons)
            for arm in ARMS
        },
        "median_ttft_s_on_reuse_targets": {
            arm: (
                statistics.median(
                    row["arms"][arm]["ttft_s"]
                    for row in comparisons
                    if row["reuse"] is not None
                )
                if reuse
                else None
            )
            for arm in ARMS
        },
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("workload", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(summarize(args.run, args.workload), ensure_ascii=False, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
