#!/usr/bin/env python3
"""Throughput, latency and cache use against concurrency, one line per
configuration -- the form of the ThunderAgent paper's Figure 1a/1b.

    python experiments/workload/plot_concurrency_sweep.py \\
        data/results/workload/<experiment> [--window-start 180 --window-end 600]

Reads <experiment>/concurrency_<C>/<configuration>/{rounds.jsonl,
engine_metrics_raw.jsonl.gz} and writes <experiment>/figures/concurrency_sweep.png
and concurrency_sweep.md (the numbers behind every point).

Definitions (all over the window [start, end] seconds, where the load is steady):
    steps per minute     rounds that finished inside the window / window minutes
                         (the paper: "one step includes a reasoning and acting
                         period"; here a step is one LLM round of a program)
    prompt tokens per minute   engine prompt tokens of those same rounds / window
                         minutes: the work behind the steps. Steps per minute
                         alone is biased when the window catches programs in
                         their early (short) rounds, as at high concurrency
    step latency         finished - sent of those rounds, seconds (includes any
                         time held by a router: the clock starts at the send)
    prompt KV from cache vLLM counter prompt_tokens_cached_total, increase over the
                         window, x bytes of KV per token, per minute
    prompt KV recomputed (prompt_tokens_total - prompt_tokens_cached_total)
                         increase, x bytes per token, per minute
KV bytes per token: 196,608 (Qwen2.5-14B, bf16: 2 x 48 layers x 8 KV heads x 128 x 2 B).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from plot_observability import engine_series, read_jsonl, seconds_axis
from plot_run import AQUA, BLUE, INK, ORANGE, YELLOW, style

KV_BYTES_PER_TOKEN = 196_608
COLOURS = {
    "vllm-direct": BLUE,
    "thunderagent-with-throttling": ORANGE,
    "thunderagent-proxy-only": AQUA,
    "official-thunderagent": ORANGE,
    "request-throttling": AQUA,
}
DISPLAY_NAMES = {"vllm-direct": "vLLM", "official-thunderagent": "ThunderAgent",
                 "request-throttling": "Request Throttling"}


def point(run: Path, start: float, end: float) -> dict:
    rounds = [r for r in read_jsonl(run / "rounds.jsonl") if r["status"] == "ok"]
    inside = [r for r in rounds if start <= r["finished_seconds"] <= end]
    minutes = (end - start) / 60
    latencies = sorted(r["finished_seconds"] - r["sent_seconds"] for r in inside)
    series = engine_series(read_jsonl(run / "engine_metrics_raw.jsonl.gz", True))
    prompt = cached = 0.0
    for samples in series.values():
        first = next(x for x in samples if x["t"] >= start)
        last = [x for x in samples if x["t"] <= end][-1]
        prompt += (
            last["scalars"]["vllm:prompt_tokens_total"]
            - first["scalars"]["vllm:prompt_tokens_total"]
        )
        cached += (
            last["scalars"]["vllm:prompt_tokens_cached_total"]
            - first["scalars"]["vllm:prompt_tokens_cached_total"]
        )
    gib = KV_BYTES_PER_TOKEN / 2**30
    return {
        "steps_per_minute": len(inside) / minutes,
        "steps": len(inside),
        "prompt_tokens_per_minute": sum(r["engine_prompt_tokens"] for r in inside)
        / minutes,
        "step_latency_median_s": latencies[len(latencies) // 2] if latencies else None,
        "step_latency_p90_s": latencies[int(0.9 * (len(latencies) - 1))]
        if latencies
        else None,
        "cached_gib_per_minute": cached * gib / minutes,
        "recomputed_gib_per_minute": (prompt - cached) * gib / minutes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--window-start", type=float, default=180.0)
    parser.add_argument("--window-end", type=float, default=600.0)
    arguments = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    table: dict[str, dict[int, dict]] = {}
    for run in sorted(arguments.experiment.glob("concurrency_*/*/rounds.jsonl")):
        run = run.parent
        concurrency = int(re.search(r"concurrency_(\d+)", str(run))[1])
        table.setdefault(run.name, {})[concurrency] = point(
            run, arguments.window_start, arguments.window_end
        )
    figure, axes = pyplot.subplots(1, 4, figsize=(22, 5))
    for configuration, points in sorted(table.items()):
        cs = sorted(points)
        colour = COLOURS.get(configuration, YELLOW)
        display = DISPLAY_NAMES.get(configuration, configuration)
        axes[0].plot(
            cs,
            [points[c]["steps_per_minute"] for c in cs],
            marker="s",
            color=colour,
            linewidth=2,
            label=display,
        )
        axes[1].plot(
            cs,
            [points[c]["prompt_tokens_per_minute"] / 1000 for c in cs],
            marker="s",
            color=colour,
            linewidth=2,
            label=display,
        )
        axes[2].plot(
            cs,
            [points[c]["step_latency_median_s"] for c in cs],
            marker="o",
            color=colour,
            linewidth=2,
            label=f"{display}: median",
        )
        axes[2].plot(
            cs,
            [points[c]["step_latency_p90_s"] for c in cs],
            marker="o",
            color=colour,
            linewidth=1,
            linestyle="--",
            label=f"{display}: 90th percentile",
        )
        axes[3].plot(
            cs,
            [points[c]["recomputed_gib_per_minute"] for c in cs],
            marker="^",
            color=colour,
            linewidth=2,
            label=f"{display}: recomputed",
        )
        axes[3].plot(
            cs,
            [points[c]["cached_gib_per_minute"] for c in cs],
            marker="v",
            color=colour,
            linewidth=1,
            linestyle="--",
            label=f"{display}: from cache",
        )
    style(axes[0], "steps per minute\n(rounds finished in the window)")
    style(axes[1], "prompt tokens of those steps\nper minute (thousands)")
    seconds_axis(axes[2], "step latency, send to last token (s)")
    style(axes[3], "prompt KV per minute (GiB)")
    for axis in axes:
        axis.set_xlabel("concurrent programs (closed loop)", color=INK)
        axis.set_xscale("log", base=2)
        concurrencies = sorted({c for p in table.values() for c in p})
        axis.set_xticks(concurrencies, [str(c) for c in concurrencies])
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle(
        f"{arguments.experiment.name}: window {arguments.window_start:.0f}-"
        f"{arguments.window_end:.0f} s",
        color=INK,
    )
    figure.tight_layout()
    out = arguments.experiment / "figures"
    out.mkdir(exist_ok=True)
    figure.savefig(out / "concurrency_sweep.png", dpi=120)
    lines = [
        "| configuration | concurrency | steps/min | steps | prompt tokens/min "
        "| step latency median (s) | p90 (s) | recomputed GiB/min "
        "| from cache GiB/min |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for configuration, points in sorted(table.items()):
        for c in sorted(points):
            p = points[c]
            lines.append(
                f"| {configuration} | {c} | {p['steps_per_minute']:.1f} | {p['steps']} "
                f"| {p['prompt_tokens_per_minute']:,.0f} "
                f"| {p['step_latency_median_s']:.1f} | {p['step_latency_p90_s']:.1f} "
                f"| {p['recomputed_gib_per_minute']:.1f} "
                f"| {p['cached_gib_per_minute']:.1f} |"
            )
    (out / "concurrency_sweep.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", out / "concurrency_sweep.png")


if __name__ == "__main__":
    main()
