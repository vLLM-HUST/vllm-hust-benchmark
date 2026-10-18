"""Figures about the size of a step, which "steps per minute" ignores.

    figure1a_standard_steps.png   steps per minute against concurrency, as
                                  measured (dashed) and counted in standard-size
                                  steps (solid) -- the Figure 1a form
    step_size_bias_<configuration>.png   why the two differ: later rounds carry
                                  more context; which rounds the window's steps
                                  came from at each concurrency; whole programs
                                  finished over time

Called by experiment_report.py; the numbers come from experiment_tables.py.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from pathlib import Path

from experiment_tables import throughput
from plot_run import AQUA, BLUE, INK, ORANGE, YELLOW, style

CONFIGURATION_COLOURS = {"vllm-direct": BLUE, "thunderagent-with-throttling": ORANGE}
LINE_COLOURS = (BLUE, AQUA, YELLOW, ORANGE, INK)


def figure1a(pyplot, points, start, end, out: Path) -> Path:
    rows = throughput(points, start, end)
    figure, axis = pyplot.subplots(figsize=(9, 5.5))
    for configuration in sorted({r["configuration"] for r in rows}):
        mine = sorted(
            (r for r in rows if r["configuration"] == configuration),
            key=lambda r: r["concurrency"],
        )
        cs = [r["concurrency"] for r in mine]
        colour = CONFIGURATION_COLOURS.get(configuration, INK)
        axis.plot(
            cs,
            [r["steps per minute, standard-size steps"] for r in mine],
            marker="s",
            color=colour,
            linewidth=2.5,
            label=f"{configuration}: standard-size steps",
        )
        axis.plot(
            cs,
            [r["steps per minute"] for r in mine],
            marker="o",
            color=colour,
            linewidth=1,
            linestyle="--",
            alpha=0.7,
            label=f"{configuration}: as measured",
        )
    concurrencies = sorted(points)
    axis.set_xscale("log", base=2)
    axis.set_xticks(concurrencies, [str(c) for c in concurrencies])
    axis.set_ylim(0, None)
    style(axis, "steps per minute")
    axis.set_xlabel("concurrent programs (closed loop)", color=INK)
    axis.set_title(
        f"window {start:.0f}-{end:.0f} s; solid = every step counted "
        "at the workload's mean step size",
        color=INK,
    )
    axis.legend(frameon=False, fontsize=8, loc="lower center")
    figure.tight_layout()
    path = out / "figure1a_standard_steps.png"
    figure.savefig(path, dpi=120)
    pyplot.close(figure)
    return path


def step_size_bias(pyplot, points, configuration, start, end, out: Path) -> Path:
    runs = {c: r[configuration] for c, r in points.items() if configuration in r}
    # up to four concurrencies spread from the lowest to the highest
    ordered = sorted(runs)
    picks = sorted({round(i * (len(ordered) - 1) / 3) for i in range(4)})
    shown = [ordered[i] for i in picks] if len(ordered) > 4 else ordered
    figure, axes = pyplot.subplots(1, 3, figsize=(20, 5.2))
    by_round = defaultdict(list)
    for run in runs.values():
        for r in run.in_window(start, end):
            by_round[r["round_idx"]].append(r["engine_prompt_tokens"])
    rounds = sorted(k for k in by_round if len(by_round[k]) >= 20 and k <= 60)
    axes[0].plot(
        rounds, [statistics.median(by_round[k]) for k in rounds], color=INK, linewidth=2
    )
    for colour, concurrency in zip(LINE_COLOURS, shown, strict=False):
        run = runs[concurrency]
        inside = run.in_window(start, end)
        counts = Counter(min(r["round_idx"], 60) for r in inside)
        xs = sorted(counts)
        axes[1].plot(
            xs,
            [counts[x] / ((end - start) / 60) for x in xs],
            color=colour,
            linewidth=2,
            label=f"concurrency {concurrency}: "
            f"{len(inside) / ((end - start) / 60):.0f} steps/min",
        )
        finished = sorted(
            items[-1]["finished_seconds"]
            for items in run.programs().values()
            if items[-1]["tool_wait_after_ms"] == 0
        )
        axes[2].step(
            finished,
            range(1, len(finished) + 1),
            where="post",
            color=colour,
            linewidth=2,
            label=f"concurrency {concurrency}: {len(finished)} finished",
        )
    style(axes[0], "prompt tokens of a step (median)")
    axes[0].set_xlabel("round of the program (0 = first)", color=INK)
    axes[0].set_title("later rounds carry more context", color=INK)
    style(axes[1], "steps completed per minute, by round")
    axes[1].set_xlabel("round of the program (capped at 60)", color=INK)
    axes[1].set_title(
        f"{configuration}: which rounds the window's steps come from", color=INK
    )
    axes[1].legend(frameon=False, fontsize=8)
    style(axes[2], "programs finished so far")
    axes[2].set_xlabel("seconds since the run started", color=INK)
    axes[2].set_title(f"{configuration}: whole programs finished", color=INK)
    axes[2].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    path = out / f"step_size_bias_{configuration}.png"
    figure.savefig(path, dpi=110)
    pyplot.close(figure)
    return path
