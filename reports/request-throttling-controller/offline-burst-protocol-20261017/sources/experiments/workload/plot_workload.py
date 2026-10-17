#!/usr/bin/env python3
"""Figures that explain the workload itself and the series that varies only the
coefficient of variation of the inter-arrival time, for
documents/design/workload_microbenchmark.md. Every number is computed here from
the dataset and result files; nothing is typed in.

    python experiments/workload/plot_workload.py \\
        --series data/from_ascend86/workload/arrival-cv-series \\
        --out documents/design/figures/workload_microbenchmark

Writes
    arrival_by_coefficient_of_variation.png   first 60 s of arrivals, each dataset
    output_lengths.png    histogram of output lengths (Zipf, Llumnix "Medium")
    time_to_first_token_by_coefficient_of_variation.png   median, 90th and 99th
                          percentile of time to first token against it
Needs matplotlib.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plot_run import AQUA, BLUE, INK, ORANGE, style  # noqa: I001 - shared palette

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COEFFICIENTS = ("0.5", "1", "2", "4")
DATASET = "gamma-cv{coefficient}-rate2-zipf-medium-n600"  # directory names as generated


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def arrival_figure(pyplot, out: Path) -> None:
    figure, axes = pyplot.subplots(len(COEFFICIENTS), 1, figsize=(11, 4.2), sharex=True)
    for axis, coefficient in zip(axes, COEFFICIENTS, strict=True):
        requests = read_jsonl(
            REPOSITORY_ROOT
            / "data/workloads"
            / DATASET.format(coefficient=coefficient)
            / "requests.jsonl"
        )
        times = [r["timestamp"] / 1000 for r in requests if r["timestamp"] <= 60_000]
        axis.eventplot(times, colors=BLUE, linelengths=0.8, linewidths=1.2)
        axis.set_yticks([])
        axis.set_ylabel(
            f"coefficient of\nvariation = {coefficient}\n{len(times)} requests",
            color=INK,
            rotation=0,
            ha="right",
            va="center",
        )
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.tick_params(colors=INK)
    axes[-1].set_xlabel("arrival time (s); first 60 s of each dataset", color=INK)
    figure.suptitle(
        "Same average rate (2 requests/s), different burstiness: one tick per request",
        color=INK,
    )
    figure.tight_layout()
    figure.savefig(out, dpi=130)
    pyplot.close(figure)


def length_figure(pyplot, out: Path) -> None:
    requests = read_jsonl(
        REPOSITORY_ROOT
        / "data/workloads"
        / DATASET.format(coefficient="1")
        / "requests.jsonl"
    )
    lengths = [r["output_length"] for r in requests]
    mean = sum(lengths) / len(lengths)
    median = percentile(lengths, 0.5)
    p99 = percentile(lengths, 0.99)
    figure, axis = pyplot.subplots(figsize=(11, 3.6))
    bins = [2**exponent for exponent in range(3, 14)]
    axis.hist(lengths, bins=bins, color=BLUE, edgecolor="white", linewidth=2)
    axis.set_xscale("log", base=2)
    for value, label, colour in (
        (median, f"median {median}", AQUA),
        (mean, f"mean {mean:.0f}", ORANGE),
        (p99, f"99th percentile {p99}", INK),
    ):
        axis.axvline(value, color=colour, linewidth=2)
        axis.text(value * 1.05, axis.get_ylim()[1] * 0.85, label, color=INK)
    style(axis, "requests")
    axis.set_xlabel("output length (tokens, log scale)", color=INK)
    axis.set_title(
        f"{len(lengths)} output lengths, Zipf with mean 256 (Llumnix 'Medium')",
        color=INK,
    )
    figure.tight_layout()
    figure.savefig(out, dpi=130)
    pyplot.close(figure)


def ttft_figure(pyplot, series: Path, out: Path) -> None:
    coefficient_values = []
    by_percentile = {"median": [], "90th percentile": [], "99th percentile": []}
    for coefficient in COEFFICIENTS:
        records = read_jsonl(
            series / DATASET.format(coefficient=coefficient) / "requests.jsonl"
        )
        ttft = [
            r["first_token_seconds"] - r["sent_seconds"]
            for r in records
            if r["status"] == "ok" and r["first_token_seconds"] is not None
        ]
        coefficient_values.append(float(coefficient))
        for name, fraction in (
            ("median", 0.5),
            ("90th percentile", 0.9),
            ("99th percentile", 0.99),
        ):
            by_percentile[name].append(percentile(ttft, fraction))
    figure, axis = pyplot.subplots(figsize=(10, 4))
    for (name, values), colour in zip(
        by_percentile.items(), (AQUA, BLUE, ORANGE), strict=True
    ):
        axis.plot(coefficient_values, values, marker="o", markersize=8, linewidth=2,
                  color=colour)  # fmt: skip
        axis.annotate(
            f"{name} {values[-1]:.2f} s",
            (coefficient_values[-1], values[-1]),
            textcoords="offset points",
            xytext=(8, -4),
            color=INK,
        )
        axis.annotate(
            f"{values[0]:.2f}",
            (coefficient_values[0], values[0]),
            textcoords="offset points",
            xytext=(-34, -4),
            color=INK,
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(coefficient_values, [str(value) for value in coefficient_values])
    axis.set_xlim(0.35, 9)
    style(axis, "time to first token (s)")
    axis.set_xlabel("coefficient of variation of the inter-arrival time", color=INK)
    axis.set_title("600 requests per point, one run each (single seed)", color=INK)
    figure.tight_layout()
    figure.savefig(out, dpi=130)
    pyplot.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    arguments.out.mkdir(parents=True, exist_ok=True)
    arrival_figure(pyplot, arguments.out / "arrival_by_coefficient_of_variation.png")
    length_figure(pyplot, arguments.out / "output_lengths.png")
    ttft_figure(
        pyplot,
        arguments.series,
        arguments.out / "time_to_first_token_by_coefficient_of_variation.png",
    )
    print(f"wrote 3 figures to {arguments.out}")


if __name__ == "__main__":
    main()
