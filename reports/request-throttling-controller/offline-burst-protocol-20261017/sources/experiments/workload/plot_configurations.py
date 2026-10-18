#!/usr/bin/env python3
"""Compare the configurations of one experiment on one time axis. Every axis is
an absolute quantity with its unit; nothing is a percentage.

    python experiments/workload/plot_configurations.py \\
        data/results/workload/<experiment>

Reads <experiment>/<configuration>/{rounds.jsonl, metrics.jsonl, summary.json}
(written by replay_sessions.py through program_configurations.sh) and writes
<experiment>/figures/configurations_over_time.png, five panels:

    KV cache in use, all engines (tokens), with the engines' total capacity
    rounds completed so far (rounds)
    decode throughput, all engines (output tokens per second, 10 s average)
    programs waiting for their first token (programs, 10 s average): sent,
        nothing back yet -- queued in the engine, or held back by the router
    time to first token (seconds, log scale): rounds grouped by 60 s of send
        time; solid line the median, dashed the maximum
    programs paused by ThunderAgent (programs), from its own log
        (<configuration>/thunderagent.log: `Paused program X` until
        `Resumed program X`); the log's clock is aligned to the replay's by the
        first `Assigned new program` line, which is the first program's first
        round (arrival time 0; checked to within 1 ms on 2026-09-29)

and configurations_table.md: the summary numbers side by side.
"""

from __future__ import annotations

import datetime
import json
import re
import sys
from pathlib import Path

from plot_run import (
    BLUE,
    AQUA,
    INK,
    ORANGE,
    WINDOW_SECONDS,
    binned_percentiles,
    by_engine,
    capacities,
    gauge,
    log_seconds_axis,
    program_phases,
    rates,
    read_jsonl,
    style,
    summed_over_engines,
)

CONFIGURATION_COLOURS = (BLUE, ORANGE)
LOG_LINE = re.compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) .*?"
    r"(Assigned new program|Paused program|Resumed program) (\S+)"
)


def paused_over_time(
    log: Path, last_round_end: dict[str, float], run_end: float
) -> tuple[list[float], list[int], int, int]:
    """(times, programs paused at that time, pauses, pauses after the last round)
    from a ThunderAgent log, in seconds since the replay started.

    ThunderAgent cannot stop a round being generated: it marks the program and
    the pause takes effect when the round ends. A program marked during its
    last round is paused when it has nothing left to run (seen on 2026-09-29:
    4 of 25 pauses); such a pause frees nothing and is counted apart, not as a
    program held back until the end of the run."""
    if not log.exists():
        return [], [], 0, 0
    start = None
    events = []
    for line in log.read_text(errors="replace").splitlines():
        match = LOG_LINE.search(line)
        if not match:
            continue
        stamp = datetime.datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S,%f")
        if start is None and match[2] == "Assigned new program":
            start = stamp
        if start is not None and match[2] != "Assigned new program":
            events.append(((stamp - start).total_seconds(), match[2], match[3]))
    paused: set[str] = set()
    times, counts = [0.0], [0]
    pauses = after_last_round = 0
    for moment, kind, program in events:
        if kind == "Paused program":
            pauses += 1
            if moment >= last_round_end.get(program, float("inf")) - 1.0:
                after_last_round += 1
                continue
            paused.add(program)
        else:
            paused.discard(program)
        times.append(moment)
        counts.append(len(paused))
    times.append(run_end)
    counts.append(len(paused))
    return times, counts, pauses, after_last_round


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    experiment = Path(sys.argv[1])
    runs = sorted(path.parent for path in experiment.glob("*/summary.json"))
    figure, axes = pyplot.subplots(6, 1, figsize=(13, 18), sharex=True)
    total_capacity = None
    table = [
        "| configuration | programs completed | rounds completed | rounds failed "
        "| rounds per minute | output tokens per second "
        "| time to first token, median (s) "
        "| time to first token, 99th percentile (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for index, run in enumerate(runs):
        colour = {'vllm-direct': BLUE, 'official-thunderagent': ORANGE,
                  'request-throttling': AQUA}.get(
                      run.name, CONFIGURATION_COLOURS[index % len(CONFIGURATION_COLOURS)])
        name = {'vllm-direct': 'vLLM', 'official-thunderagent': 'ThunderAgent',
                'request-throttling': 'Request Throttling'}.get(run.name, run.name)
        samples = read_jsonl(run / "metrics.jsonl")
        rounds = read_jsonl(run / "rounds.jsonl")
        capacity_tokens = capacities(run, None)
        total_capacity = sum(capacity_tokens.values())
        engines = by_engine(samples)

        times, values = summed_over_engines(
            engines,
            lambda s, c=capacity_tokens: gauge(
                "vllm:kv_cache_usage_perc", c.get(s[0].get("engine"), 0)
            )(s),
        )
        axes[0].plot(times, values, color=colour, linewidth=2, label=name)

        finished = sorted(r["finished_seconds"] for r in rounds if r["status"] == "ok")
        axes[1].step(finished, range(1, len(finished) + 1), where="post",
                     color=colour, linewidth=2, label=name)  # fmt: skip

        times, values = summed_over_engines(
            engines, lambda s: rates(s, "vllm:generation_tokens_total")
        )
        axes[2].plot(times, values, color=colour, linewidth=2, label=name)

        phases = program_phases(rounds)
        end = max((b[2] for blocks in phases.values() for b in blocks), default=0)
        grid = [index * 1.0 for index in range(int(end) + 2)]
        waiting = [0] * len(grid)
        for blocks in phases.values():
            for phase, begins, ends in blocks:
                if phase != "reasoning, waiting":
                    continue
                for index in range(int(begins), min(len(grid), int(ends) + 1)):
                    if begins <= grid[index] < ends:
                        waiting[index] += 1
        window = int(WINDOW_SECONDS)
        averaged = [
            sum(waiting[max(0, i - window + 1) : i + 1]) / min(window, i + 1)
            for i in range(len(waiting))
        ]
        axes[3].plot(grid, averaged, color=colour, linewidth=2, label=name)

        ok = [r for r in rounds if r["status"] == "ok" and r["first_token_seconds"]]
        lines = binned_percentiles(
            ok, lambda r: r["first_token_seconds"] - r["sent_seconds"]
        )
        for percentile_name, style_name in (("median", "-"), ("maximum", "--")):
            times, values = lines[percentile_name]
            axes[4].plot(times, values, color=colour, linestyle=style_name, linewidth=2,
                         label=f"{name}, {percentile_name}")  # fmt: skip

        last_round_end: dict[str, float] = {}
        for r in rounds:
            last_round_end[r["session_id"]] = max(
                last_round_end.get(r["session_id"], 0.0), r["finished_seconds"]
            )
        run_end = max(last_round_end.values(), default=0.0)
        # Prefix hit rate is a counter delta, not KV occupancy or lifetime ratio.
        hit_times, hit_rates = [], []
        for start in range(0, int(run_end), 60):
            selected = [sample for sample in samples if start <= sample['t'] < start + 60]
            if len(selected) < 2:
                continue
            queries = selected[-1].get('vllm:prefix_cache_queries_total', 0) - selected[0].get('vllm:prefix_cache_queries_total', 0)
            hits = selected[-1].get('vllm:prefix_cache_hits_total', 0) - selected[0].get('vllm:prefix_cache_hits_total', 0)
            hit_times.append(start + 30)
            hit_rates.append(100 * hits / queries if queries > 0 and hits >= 0 else float('nan'))
        axes[5].plot(hit_times, hit_rates, color=colour, linewidth=2, label=name)

        result = json.loads((run / "summary.json").read_text())["result"]
        ttft = result["time_to_first_token_seconds"]
        table.append(
            f"| {name} | {result['sessions_completed']} / {result['sessions']} "
            f"| {result['rounds_succeeded']} | {result['rounds_failed']} "
            f"| {result['rounds_completed_per_minute']} "
            f"| {result['output_tokens_per_second']} "
            f"| {ttft.get('p50')} | {ttft.get('p99')} |"
        )

    if total_capacity:
        axes[0].axhline(total_capacity, color=INK, linewidth=1.2, linestyle="--")
        axes[0].text(0, total_capacity, f" capacity {total_capacity:,} tokens",
                     color=INK, va="bottom")  # fmt: skip
    style(axes[0], "KV cache in use\n(tokens, all engines)")
    style(axes[1], "rounds completed\nso far (rounds)")
    style(axes[2], f"decode\n(output tokens/s,\n{WINDOW_SECONDS:.0f} s average)")
    style(
        axes[3],
        f"programs waiting\nfor first token\n(programs, "
        f"{WINDOW_SECONDS:.0f} s average)",
    )
    axes[3].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=6))
    axes[3].yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    )
    log_seconds_axis(axes[4], "time to first token\n(seconds, log scale,\n60 s groups)")
    style(axes[5], 'Prefix cache hit rate\n(%, counter deltas over 60 s)')
    axes[5].set_ylim(0, 100)
    for axis in axes:
        if axis.get_legend_handles_labels()[0]:
            axis.legend(frameon=False, loc="upper right", fontsize=9)
    if not axes[5].lines:
        axes[5].remove()
        for index, axis in enumerate(axes[:5]):
            axis.set_position([.12, .80 - index * .17, .82, .13])
        axes[4].tick_params(labelbottom=True)
        axes[4].set_xlabel("seconds since the run started", color=INK)
        figure.text(.12, .01, 'Pause-count panel omitted: Router pause logs not supplied.', color=INK)
    else:
        axes[5].set_xlabel("seconds since the run started", color=INK)
    figure.suptitle(experiment.name, color=INK)
    if axes[5].lines:
        figure.tight_layout()
    (experiment / "figures").mkdir(exist_ok=True)
    out = experiment / "figures" / "configurations_over_time.png"
    figure.savefig(out, dpi=130)
    (experiment / "figures" / "configurations_table.md").write_text(
        "\n".join(table) + "\n"
    )
    print("\n".join(table))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
