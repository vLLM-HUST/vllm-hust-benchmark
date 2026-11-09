"""Recover fixed018 long-input coverage without relabelling mixed loads as a4K/16K pair."""

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "reports/long-context-fixed018-20261029"


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def summarize(rows, seconds=900):
    closed = [
        r for r in rows if r["success"] and 0 <= r["start"] <= r["end"] <= seconds
    ]
    long = [r for r in closed if r["prompt_tokens"] >= 16384]
    windows = []
    for start in range(0, seconds, 60):
        # Completion-time diagnostic bins. Crossing requests are not silently
        # recast as an entirely in-window successful cost cohort.
        cohort = [r for r in long if start < r["end"] <= start + 60]
        ttft = [r["first_token"] - r["start"] for r in cohort]
        tpot = [
            (r["last_token"] - r["first_token"]) / (r["output_tokens"] - 1)
            for r in cohort
            if r["output_tokens"] > 1
        ]
        windows.append(
            {
                "start": start,
                "end": start + 60,
                "long_input_completed_requests": len(cohort),
                "ttft_p95_seconds": percentile(ttft, 0.95),
                "ttft_p99_seconds": percentile(ttft, 0.99),
                "tpot_p95_seconds": percentile(tpot, 0.95),
                "tpot_p99_seconds": percentile(tpot, 0.99),
            }
        )
    return {
        "requests_including_drain": len(rows),
        "failed_including_drain": sum(not r["success"] for r in rows),
        "input_max_tokens": max(r["prompt_tokens"] for r in rows),
        "input_ge16k_including_drain": sum(r["prompt_tokens"] >= 16384 for r in rows),
        "successful_closed_requests": len(closed),
        "input_ge16k_successful_closed_requests": len(long),
        "completion_time_diagnostic_windows": windows,
        "fixed_4k_16k_pair_pass": None,
        "retrieval_correctness_pass": None,
        "five_dataset_stability_pass": None,
    }


def recover(base):
    arms = []
    for concurrency in (1, 2, 4, 8, 16):
        directory = base / f"C{concurrency}-900s"
        manifest = {
            line.split(maxsplit=1)[1].removeprefix("./"): line.split()[0]
            for line in (directory / "SHA256SUMS").read_text().splitlines()
        }
        names = [
            "server-metadata.json",
            "server-window.log",
            "workload-result/requests.jsonl",
            "workload-result/summary.json",
        ]
        hashes = {}
        for name in names:
            hashes[name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
            if hashes[name] != manifest[name]:
                raise ValueError("Original member SHA mismatch: " + name)
        metadata = json.loads((directory / names[0]).read_text())
        runtime = metadata["runtime"]
        if (
            runtime["vllm_release"] != "0.18.0"
            or runtime["vllm_ascend_release"] != "0.18.0"
        ):
            raise ValueError("Not the fixed B0")
        raw = [
            json.loads(line) for line in (directory / names[2]).read_text().splitlines()
        ]
        rows = [
            {
                **{
                    key: r[key]
                    for key in (
                        "start",
                        "end",
                        "first_token",
                        "last_token",
                        "success",
                        "prompt_tokens",
                        "finish_reason",
                    )
                },
                "output_tokens": r["usage"]["completion_tokens"],
            }
            for r in raw
        ]
        original = json.loads((directory / names[3]).read_text())
        derived = summarize(rows)
        if (
            derived["requests_including_drain"] != original["requests_started"]
            or derived["failed_including_drain"] != original["failed_requests"]
        ):
            raise ValueError("Original aggregate mismatch")
        arms.append(
            {
                "concurrency": concurrency,
                "seconds": 900,
                "runtime": runtime,
                "source_sha256": hashes,
                "requests": rows,
                "summary": derived,
            }
        )
    return {
        "schema": "fixed018-long-input-coverage/v1",
        "new_model_runs": 0,
        "role": "B0",
        "arms": arms,
        "limits": [
            "Existing SWE-prefix-reuse mixed-length sequential dialogues, not fixed4K/16K matched arrivals or a scored retrieval pair.",
            "Request diagnostics include explicit finish reasons; length termination alone is not silent truncation. Service-log event review is still separate.",
            "60s bins group successful requests by completion time inside the900s run; no cross-boundary token cost or window-throughput claim.",
            "Linear interpolated percentiles retain sample counts and null empty bins; small bins do not establish stable tail estimates.",
            "No source prompts, answers or token IDs in this public projection; original files remain unchanged.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = DEST / "coverage.json"
    if args.raw:
        if args.check:
            parser.error("Use --raw for recovery or --check for public recomputation")
        DEST.mkdir(exist_ok=True)
        target.write_text(json.dumps(recover(args.raw), indent=2) + "\n")
    data = json.loads(target.read_text())
    for arm in data["arms"]:
        assert arm["summary"] == summarize(arm["requests"])
    print(
        "Fixed018 long-input coverage recomputed; controlled pairing remains incomplete"
    )


if __name__ == "__main__":
    main()
