"""CPU-only workload, schedule, and measurement contracts for the new campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
import math


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def requests(prompts, outputs, arrivals):
    if len(outputs) == 1:
        outputs = outputs * len(prompts)
    if len(arrivals) == 1:
        arrivals = arrivals * len(prompts)
    if not prompts or len(prompts) != len(outputs) or len(prompts) != len(arrivals):
        raise ValueError("prompt, output and arrival vectors must agree")
    if any(type(n) is not int or n <= 0 for n in prompts + outputs) or any(
        not math.isfinite(t) or t < 0 for t in arrivals
    ):
        raise ValueError("lengths must be positive and arrivals nonnegative")
    return [
        dict(index=i, prompt_tokens=p, output_tokens=o, arrival_ms=t, token_id=1000 + i)
        for i, (p, o, t) in enumerate(zip(prompts, outputs, arrivals))
    ]


def request_metrics(submitted_ns, events, expected_tokens):
    """Output-chunk times, not fabricated per-token timestamps for bundled output."""
    if not events or not events[-1]["finished"]:
        raise ValueError("request did not finish")
    previous = 0
    previous_time = submitted_ns
    chunks = []
    for row in events:
        n = row["output_tokens"]
        if n < previous or row["monotonic_ns"] < previous_time:
            raise ValueError("nonmonotonic output")
        if n > previous:
            chunks.append(
                dict(
                    time_ms=(row["monotonic_ns"] - submitted_ns) / 1e6,
                    new_tokens=n - previous,
                    total_tokens=n,
                )
            )
        previous = n
        previous_time = row["monotonic_ns"]
    if previous != expected_tokens or not chunks:
        raise ValueError("unexpected output length")
    return dict(
        first_output_ms=chunks[0]["time_ms"],
        completion_ms=(events[-1]["monotonic_ns"] - submitted_ns) / 1e6,
        output_chunks=chunks,
        inter_output_ms=[
            b["time_ms"] - a["time_ms"] for a, b in zip(chunks, chunks[1:])
        ],
    )


def balanced_cells(repetitions=5):
    """Alternating rotations avoid fixing an arm to one position in every block."""
    if repetitions < 2:
        raise ValueError("at least two independent processes per arm")
    arms = [("stock", 32), ("linked", 32), ("linked", 128), ("stock", 128)]
    result = []
    for block in range(repetitions):
        offset = block % len(arms)
        order = arms[offset:] + arms[:offset]
        if block % 2:
            order.reverse()
        for observer, budget in order:
            result.append(
                dict(
                    name=f"b{block}-{observer}{budget}",
                    block=block,
                    observation=observer,
                    token_budget=budget,
                )
            )
    return result


def process_summary(batches):
    if not batches:
        raise ValueError("no measured batches")
    reference = batches[0]["outputs"]
    if any(b["outputs"] != reference for b in batches):
        raise ValueError("token outputs changed across batches")
    return dict(
        batch_ms=statistics.median(b["elapsed_ms"] for b in batches),
        requests={
            key: {
                metric: statistics.median(b["requests"][key][metric] for b in batches)
                for metric in ("first_output_ms", "completion_ms")
            }
            for key in batches[0]["requests"]
        },
        outputs=reference,
    )


def validate_context(directory, require_execution=True):
    paths = sorted(Path(directory).glob("*.jsonl"))
    if not paths:
        raise ValueError("missing context files")
    counts = {}
    for path in paths:
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        if rows[0]["type"] != "session" or rows[-1]["type"] != "summary":
            raise ValueError("unclosed producer")
        if (
            rows[-1].get("dropped_records") != 0
            or rows[-1].get("written_records") != len(rows) - 2
        ):
            raise ValueError("lost context records")
        for row in rows[1:-1]:
            counts[row["type"]] = counts.get(row["type"], 0) + 1
    if not counts.get("step") or (require_execution and not counts.get("execution")):
        raise ValueError(
            "linked capture requires both scheduler and worker execution records"
        )
    return counts
