"""Recompute the issue #3 latency-admission diagnostic from raw receipts."""

from __future__ import annotations

import gzip
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
ARMS = {
    "short": (
        "workload.jsonl",
        {
            "baseline": "run1/baseline",
            "approximate512": "run1/approximate512",
            "approximate256": "run1/approximate256",
            "approximate128": "run1/approximate128",
            "rollback": "run1/rollback",
            "guarded_short": "run-guarded-short/guarded_short",
        },
    ),
    "seven_k": (
        "workload_long.jsonl",
        {
            "baseline": "run-long/baseline",
            "approximate512": "run-long/approximate512",
        },
    ),
    "sixteen_k": (
        "workload_16k.jsonl",
        {
            "first_baseline": "run-16k/baseline",
            "first_approximate512": "run-16k/approximate512",
            "baseline": "run-guarded-long-r2/baseline",
            "guarded_long": "run-guarded-long-r2/guarded_long",
        },
    ),
}
HIT = re.compile(
    r"SimLLM: reusing (\d+) semantic tokens for request (\S+) "
    r"\(source=(\S+), cosine=([^)]*)\)"
)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def summarize_arm(path: Path, workload: list[dict]) -> tuple[dict, list[dict]]:
    raw = read_rows(path / "outputs.jsonl")
    rows = [row for row in raw if row.get("phase") != "summary"]
    client_summary = next(row for row in raw if row.get("phase") == "summary")
    if [row["id"] for row in rows] != [row["id"] for row in workload]:
        raise AssertionError(f"Input/output IDs differ: {path}")
    expected = {row["id"]: row["expected"] for row in workload}
    by_request_id = {row["request_id"]: row["id"] for row in rows}

    def client_id(worker_id: str) -> str:
        matches = [
            item_id
            for request_id, item_id in by_request_id.items()
            if worker_id == request_id or worker_id.startswith(request_id + "-")
        ]
        if len(matches) != 1:
            raise AssertionError(f"Unmapped cache event: {path}: {worker_id}")
        return matches[0]

    hits = {}
    with gzip.open(path / "server.log.gz", "rt", errors="replace") as source:
        server_log = source.read()
    for skipped, request_id, source_id, cosine in HIT.findall(server_log):
        hits[client_id(request_id)] = {
            "source_id": client_id(source_id),
            "skipped_tokens": int(skipped),
            "cosine": float(cosine),
        }
    # Client outputs do not repeat workload variant; use stable request IDs.
    targets = [row for row in rows if not row["id"].endswith("_0")]
    details = [
        {
            "id": row["id"],
            "prompt_tokens": row["prompt_tokens"],
            "completion_tokens": row["completion_tokens"],
            "ttft_s": row["ttft_s"],
            "output_text": row["output_text"],
            "expected": expected[row["id"]],
            "correct": row["output_text"].strip() == expected[row["id"]],
            "reuse": hits.get(row["id"]),
        }
        for row in rows
    ]
    summary = {
        "requests": len(rows),
        "correct": sum(row["correct"] for row in details),
        "prompt_tokens_min": min(row["prompt_tokens"] for row in rows),
        "prompt_tokens_max": max(row["prompt_tokens"] for row in rows),
        "target_ttft_median_s": round(
            statistics.median(row["ttft_s"] for row in targets), 4
        ),
        "target_ttft_mean_s": round(
            statistics.mean(row["ttft_s"] for row in targets), 4
        ),
        "requests_per_s": client_summary["requests_per_s"],
        "semantic_hits": len(hits),
        "skipped_tokens": sum(hit["skipped_tokens"] for hit in hits.values()),
    }
    return summary, details


def main() -> None:
    report = {}
    for experiment, (input_name, arms) in ARMS.items():
        workload = read_rows(EVIDENCE / input_name)
        results = {}
        for arm, relative_path in arms.items():
            summary, details = summarize_arm(EVIDENCE / relative_path, workload)
            results[arm] = {"summary": summary, "requests": details}
        if experiment == "short":
            baseline = [row["output_text"] for row in results["baseline"]["requests"]]
            for arm in ("rollback", "guarded_short"):
                if [row["output_text"] for row in results[arm]["requests"]] != baseline:
                    raise AssertionError(f"{arm} differs from short baseline")
        report[experiment] = results
    (ROOT / "comparisons.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    for experiment, arms in report.items():
        for arm, value in arms.items():
            print(experiment, arm, value["summary"])


if __name__ == "__main__":
    main()
