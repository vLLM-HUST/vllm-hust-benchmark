#!/usr/bin/env python3
"""The same ground truth for two (or more) configurations, on one time axis and
one scale per panel, so a difference is a difference in the data, not in the
axes.

    python experiments/workload/plot_observability_compare.py \\
        data/results/workload/<experiment>

Reads every <experiment>/<configuration>/ that plot_observability.py reads and
writes <experiment>/figures/E1_configurations.png. Panels, all engines summed:
    KV in use (tokens) and total capacity
    requests running; requests waiting for KV ("capacity" reason)
    preemptions so far
    prefill and decode tokens/s (10 s window)
    rounds completed so far; programs completed so far
    time to first token: median and maximum per 60 s of send time
    longest gap between streamed tokens per 10 s
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from plot_observability import (
    FLOOR_SECONDS,
    WINDOW,
    counter_rate,
    engine_series,
    grouped,
    read_jsonl,
    seconds_axis,
    with_gaps,
)
from plot_run import BLUE, INK, ORANGE, style

COLOURS = (BLUE, ORANGE)


def summed(series, value_of, step=1.0):
    """Add a per-engine quantity across engines on a 1 s grid."""
    end = max(s[-1]["t"] for s in series.values())
    grid = [i * step for i in range(int(end / step) + 1)]
    totals = [0.0] * len(grid)
    for samples in series.values():
        position = -1
        for index, moment in enumerate(grid):
            while position + 1 < len(samples) and samples[position + 1]["t"] <= moment:
                position += 1
            if position >= 0:
                totals[index] += value_of(samples[position], samples)
    return grid, totals


def summed_rate(series, name):
    rates = {e: counter_rate(s, name) for e, s in series.items()}
    end = max(t[-1] for t, _ in rates.values() if t)
    grid = [float(i) for i in range(int(end) + 1)]
    totals = [0.0] * len(grid)
    for times, values in rates.values():
        position = -1
        for index, moment in enumerate(grid):
            while position + 1 < len(times) and times[position + 1] <= moment:
                position += 1
            if position >= 0:
                totals[index] += values[position]
    return grid, totals


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    experiment = Path(sys.argv[1])
    runs = sorted(p.parent for p in experiment.glob("*/engine_metrics_raw.jsonl.gz"))
    figure, axes = pyplot.subplots(10, 1, figsize=(14, 30), sharex=True)
    for colour, run in zip(COLOURS, runs, strict=False):
        name = run.name
        summary = json.loads((run / "summary.json").read_text())
        capacity = {
            e["endpoint"]: e["kv_capacity_tokens_device"] + e["kv_capacity_tokens_host"]
            for e in summary.get("engines", [])
        }
        series = engine_series(read_jsonl(run / "engine_metrics_raw.jsonl.gz", True))
        rounds = read_jsonl(run / "rounds.jsonl")
        tokens = read_jsonl(run / "tokens.jsonl")

        def kv(sample, samples, capacity=capacity, series=series):
            engine = next(e for e, s in series.items() if s is samples)
            return (
                capacity.get(engine, 0) * sample["scalars"]["vllm:kv_cache_usage_perc"]
            )

        axes[0].plot(*summed(series, kv), color=colour, linewidth=1, label=name)
        axes[0].axhline(sum(capacity.values()), color=INK, linestyle="--", linewidth=1)
        axes[1].plot(
            *summed(series, lambda x, _: x["scalars"]["vllm:num_requests_running"]),
            color=colour,
            linewidth=1,
            label=name,
        )
        axes[2].plot(
            *summed(
                series,
                lambda x, _: x["labelled"].get(
                    ("vllm:num_requests_waiting_by_reason", "capacity"), 0
                ),
            ),
            color=colour,
            linewidth=1,
            label=name,
        )
        axes[3].plot(
            *summed(
                series,
                lambda x, s: (
                    x["scalars"]["vllm:num_preemptions_total"]
                    - s[0]["scalars"]["vllm:num_preemptions_total"]
                ),
            ),
            color=colour,
            label=name,
        )
        axes[4].plot(
            *summed_rate(series, "vllm:prompt_tokens_total"),
            color=colour,
            linewidth=1,
            label=name,
        )
        axes[5].plot(
            *summed_rate(series, "vllm:generation_tokens_total"),
            color=colour,
            linewidth=1,
            label=name,
        )
        finished = sorted(r["finished_seconds"] for r in rounds if r["status"] == "ok")
        axes[6].step(finished, range(1, len(finished) + 1), color=colour, label=name)
        last_round = defaultdict(float)
        planned = {}
        for r in rounds:
            last_round[r["session_id"]] = max(
                last_round[r["session_id"]], r["finished_seconds"]
            )
        workload = Path(summary["parameters"]["workload"]).name
        sessions_csv = experiment.parents[2] / "workloads" / workload / "sessions.csv"
        if sessions_csv.exists():
            for line in sessions_csv.read_text().splitlines()[1:]:
                session = line.split(",")[0]
                planned[session] = planned.get(session, 0) + 1
        done_rounds = defaultdict(int)
        for r in rounds:
            if r["status"] == "ok":
                done_rounds[r["session_id"]] += 1
        completed = sorted(
            last_round[s]
            for s in last_round
            if planned and done_rounds[s] == planned.get(s, -1)
        )
        axes[7].step(completed, range(1, len(completed) + 1), color=colour, label=name)
        ok = [r for r in rounds if r["status"] == "ok" and r["first_token_seconds"]]
        rows = grouped(
            [
                (r["sent_seconds"], r["first_token_seconds"] - r["sent_seconds"])
                for r in ok
            ],
            60,
        )
        times = [x[0] for x in rows]
        axes[8].plot(
            *with_gaps(times, [max(x[1], FLOOR_SECONDS) for x in rows], 120),
            color=colour,
            linestyle="--",
            label=f"{name}: median",
        )
        axes[8].plot(
            *with_gaps(times, [x[3] for x in rows], 120),
            color=colour,
            label=f"{name}: maximum",
        )
        gaps = []
        for line in tokens:
            for (a, _), (b, count) in zip(
                line["chunks"], line["chunks"][1:], strict=False
            ):
                gaps.append((b, (b - a) / max(1, count)))
        rows = grouped(gaps, WINDOW)
        axes[9].plot(
            *with_gaps([x[0] for x in rows], [x[3] for x in rows], 2 * WINDOW),
            color=colour,
            linewidth=1,
            label=name,
        )
    labels = (
        "KV cache in use, all engines\n(tokens; dashed: capacity)",
        "requests running,\nall engines",
        "requests waiting for KV\n(reason 'capacity'), all engines",
        "preemptions so far,\nall engines",
        f"prefill: prompt tokens/s\n({WINDOW:.0f} s window)",
        f"decode: output tokens/s\n({WINDOW:.0f} s window)",
        "rounds completed so far",
        "programs completed so far",
        None,
        None,
    )
    for axis, label in zip(axes, labels, strict=True):
        if label:
            style(axis, label)
        axis.legend(frameon=False, fontsize=8, loc="upper left")
    seconds_axis(axes[8], "time to first token\n(s, 60 s groups)")
    seconds_axis(
        axes[9], f"longest gap between streamed\ntokens (s, {WINDOW:.0f} s groups)"
    )
    axes[-1].set_xlabel("seconds since the run started", color=INK)
    figure.suptitle(experiment.name + "  --  E1 configurations compared", color=INK)
    figure.tight_layout()
    (experiment / "figures").mkdir(exist_ok=True)
    out = experiment / "figures" / "E1_configurations.png"
    figure.savefig(out, dpi=100)
    print("wrote", out)
    plot_waiting_per_program(pyplot, runs, experiment / "figures")


def plot_waiting_per_program(pyplot, runs, figures: Path) -> None:
    """E2: how long each program waited in total, all programs sorted.

    Waiting = time to first token summed over the program's rounds, measured by
    the client from the moment it sent the round, so a hold inside ThunderAgent
    is included (checked: every held request's client time to first token is at
    least its hold). One point per program; the rank on the x axis, the seconds
    on the y axis. Comparing medians of single requests hides where the waiting
    goes; this shows every program's share."""
    figure, axis = pyplot.subplots(figsize=(13, 5))
    for colour, run in zip(COLOURS, runs, strict=False):
        per_program = defaultdict(float)
        for r in read_jsonl(run / "rounds.jsonl"):
            if r["status"] == "ok" and r["first_token_seconds"]:
                per_program[r["session_id"]] += (
                    r["first_token_seconds"] - r["sent_seconds"]
                )
        waiting = sorted(per_program.values())
        total = sum(waiting)
        axis.plot(
            range(1, len(waiting) + 1),
            waiting,
            color=colour,
            marker="o",
            markersize=3,
            label=f"{run.name}: {len(waiting)} programs, {total:,.0f} s in total",
        )
    style(axis, "time the program spent waiting\nfor first tokens, all rounds (s)")
    axis.set_xlabel("programs, sorted from least to most waiting", color=INK)
    axis.legend(frameon=False, fontsize=9, loc="upper left")
    figure.suptitle(
        "E2  waiting per program (client-measured, holds included)", color=INK
    )
    figure.tight_layout()
    out = figures / "E2_waiting_per_program.png"
    figure.savefig(out, dpi=110)
    pyplot.close(figure)
    print("wrote", out)


if __name__ == "__main__":
    main()
