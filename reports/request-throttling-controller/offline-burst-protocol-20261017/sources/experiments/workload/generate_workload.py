#!/usr/bin/env python3
"""Build a synthetic request dataset. It only writes data; it sends nothing.

    python experiments/workload/generate_workload.py --name gamma-cv2-zipf-medium \\
        --requests 1000 --request-rate 4 --arrival-process gamma \\
        --coefficient-of-variation 2 \\
        --input-length-distribution zipf --input-length-mean 256 \\
        --output-length-distribution zipf --output-length-mean 256

Writes two files into data/workloads/<name>/:

    requests.jsonl  one request per line, in arrival order:
                    {"timestamp": 0, "input_length": 212, "output_length": 97}
    manifest.json   every parameter, summary statistics, sha256 of requests.jsonl

The record fields are Mooncake's trace format (kvcache-ai/Mooncake,
FAST25-release): `timestamp` is the arrival time in milliseconds from the first
request, `input_length` and `output_length` are token counts. There is no text:
whoever sends the requests turns a length into a prompt. Using a published
format means the same sender can also replay Mooncake's real traces.

Reproducibility: the same parameters give byte-identical files. Each dimension
has its own random stream, seeded "<seed>/arrival", "<seed>/input_length",
"<seed>/output_length", so changing one dimension (say the CV) does not change
the others (the lengths stay the same request by request).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
from pathlib import Path

from arrival_process import ARRIVAL_PROCESSES, arrival_times_seconds
from length_distribution import (
    LENGTH_DISTRIBUTIONS,
    LLUMNIX_ZIPF_ALPHA,
    sample_lengths,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKLOADS_DIRECTORY = REPOSITORY_ROOT / "data" / "workloads"


def dimension_random(seed: int, dimension: str) -> random.Random:
    """A random stream for one dimension; a string seed is hashed (SHA-512) by
    Python, so it is the same on every run and every machine."""
    return random.Random(f"{seed}/{dimension}")


def zipf_alpha_for(mean: int, maximum: int, alpha: float | None) -> float:
    """An explicit alpha wins; otherwise Llumnix's table, otherwise 1.0 (as Llumnix)."""
    if alpha is not None:
        return alpha
    return LLUMNIX_ZIPF_ALPHA.get((mean, maximum), 1.0)


def lengths_for(arguments: argparse.Namespace, which: str) -> tuple[list[int], dict]:
    """`which` is "input" or "output"; returns the lengths and their parameters."""
    distribution = getattr(arguments, f"{which}_length_distribution")
    mean = getattr(arguments, f"{which}_length_mean")
    minimum = getattr(arguments, f"{which}_length_minimum")
    maximum = getattr(arguments, f"{which}_length_maximum")
    alpha = zipf_alpha_for(mean, maximum, getattr(arguments, f"{which}_zipf_alpha"))
    lengths = sample_lengths(
        dimension_random(arguments.seed, f"{which}_length"),
        arguments.requests,
        distribution,
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        zipf_alpha=alpha,
    )
    parameters = {"distribution": distribution, "maximum": maximum}
    if distribution == "uniform":
        parameters["minimum"] = minimum
    else:
        parameters.update(mean=mean, zipf_alpha=alpha)
    return lengths, parameters


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def describe(values: list[float]) -> dict:
    mean = statistics.fmean(values)
    return {
        "mean": round(mean, 3),
        "coefficient_of_variation": round(statistics.pstdev(values) / mean, 3)
        if mean
        else 0.0,
        "p50": percentile(values, 0.50),
        "p99": percentile(values, 0.99),
        "minimum": min(values),
        "maximum": max(values),
    }


def git_state() -> dict:
    def run(*command: str) -> str:
        return subprocess.run(
            command, cwd=REPOSITORY_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "parent_sha": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def build(arguments: argparse.Namespace) -> tuple[list[dict], dict]:
    arrivals = arrival_times_seconds(
        dimension_random(arguments.seed, "arrival"),
        arguments.requests,
        arguments.arrival_process,
        arguments.request_rate,
        arguments.coefficient_of_variation,
    )
    input_lengths, input_parameters = lengths_for(arguments, "input")
    output_lengths, output_parameters = lengths_for(arguments, "output")
    requests = [
        {
            "timestamp": round(arrival * 1000),  # milliseconds, Mooncake's unit
            "input_length": input_length,
            "output_length": output_length,
        }
        for arrival, input_length, output_length in zip(
            arrivals, input_lengths, output_lengths, strict=True
        )
    ]
    gaps_seconds = [
        later - earlier for earlier, later in zip(arrivals, arrivals[1:], strict=False)
    ]
    parameters = {
        "seed": arguments.seed,
        "requests": arguments.requests,
        "arrival": {
            "process": arguments.arrival_process,
            "request_rate_per_second": arguments.request_rate,
            "coefficient_of_variation": arguments.coefficient_of_variation,
        },
        "input_length": input_parameters,
        "output_length": output_parameters,
    }
    summary = {
        "duration_seconds": round(arrivals[-1], 3),
        "inter_arrival_seconds": {
            key: round(value, 4) for key, value in describe(gaps_seconds).items()
        },
        "input_length_tokens": describe(input_lengths),
        "output_length_tokens": describe(output_lengths),
    }
    return requests, {"parameters": parameters, "summary": summary}


def write(name: str, requests: list[dict], description: dict) -> Path:
    directory = WORKLOADS_DIRECTORY / name
    directory.mkdir(parents=True, exist_ok=True)
    lines = "".join(
        json.dumps(request, separators=(",", ":")) + "\n" for request in requests
    )
    (directory / "requests.jsonl").write_text(lines)
    manifest = {
        "what_this_is": (
            "synthetic request dataset: arrival times and token counts only, "
            "no text; generated, not measured"
        ),
        "format": "Mooncake trace (timestamp ms, input_length, output_length)",
        "generator": "experiments/workload/generate_workload.py",
        **git_state(),
        **description,
        "output": "requests.jsonl",
        "output_sha256": hashlib.sha256(lines.encode()).hexdigest(),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="experiments/workload/generate_workload.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--name", required=True, help="directory under data/workloads/")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument(
        "--request-rate", type=float, default=4.0, help="mean requests per second"
    )
    parser.add_argument(
        "--arrival-process", choices=ARRIVAL_PROCESSES, default="poisson"
    )
    parser.add_argument(
        "--coefficient-of-variation",
        type=float,
        default=1.0,
        help="of the inter-arrival time; used by --arrival-process gamma only",
    )
    for which, default_mean in (("input", 256), ("output", 256)):
        parser.add_argument(
            f"--{which}-length-distribution",
            choices=LENGTH_DISTRIBUTIONS,
            default="zipf",
        )
        parser.add_argument(
            f"--{which}-length-mean", type=int, default=default_mean, help="zipf"
        )
        parser.add_argument(
            f"--{which}-length-minimum", type=int, default=1, help="uniform"
        )
        parser.add_argument(
            f"--{which}-length-maximum",
            type=int,
            default=6144,
            help="both; Llumnix's paper settings use 6144",
        )
        parser.add_argument(
            f"--{which}-zipf-alpha",
            type=float,
            default=None,
            help="default: Llumnix's table for (mean, maximum), else 1.0",
        )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    requests, description = build(arguments)
    directory = write(arguments.name, requests, description)
    print(f"wrote {len(requests)} requests to {directory.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    main()
