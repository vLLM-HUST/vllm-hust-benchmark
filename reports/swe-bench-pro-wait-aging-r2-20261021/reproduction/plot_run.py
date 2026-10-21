#!/usr/bin/env python3
"""Figures for one replay run: engine state, latency, and (for programs) what
every program was doing at every moment. Every axis is an absolute quantity
with its unit -- tokens, tokens per second, requests, programs, seconds --
never a percentage: a percentage hides what it is a percentage of.

    python experiments/workload/plot_run.py data/results/workload/<experiment>/<run>
    python experiments/workload/plot_run.py <run> --engine-capacity <machine>.json

Reads from the run directory (written by replay_requests.py / replay_sessions.py):
    metrics.jsonl    engine counters sampled once a second (engine_metrics.py),
                     one line per engine per sample
    summary.json     each engine's KV-cache capacity (`engines`); runs made before
                     that was recorded take it from --engine-capacity instead
    requests.jsonl   independent requests, or
    rounds.jsonl     agent programs, one line per LLM round
Writes <run>/figures/*.png. Needs matplotlib (the repository's `viz` extra).

Figures
    engine_over_time.png    one time axis, five panels:
      KV cache in use (tokens), per engine, with each engine's capacity dashed:
        vllm:kv_cache_usage_perc x that engine's KV capacity in tokens
      requests running and waiting (requests), summed over engines
      prefill (input tokens per second), summed over engines
      decode (output tokens per second), summed over engines
      prefix cache (tokens per second): tokens looked up, and of those found
    latency_over_time.png   time to first token and end-to-end latency (seconds):
                            requests grouped by 60 s of send time, one line each
                            for the median, 90th percentile and maximum
    programs_over_time.png  programs only: one row per program, coloured blocks
                            for its phases; below it, how many programs are in
                            each phase (programs); below that, KV cache in use
                            summed over engines (tokens) against total capacity

Program phases (the words are ThunderAgent's: a program is REASONING while its
LLM call is being served and ACTING while its tool runs):
    reasoning, waiting   the round was sent, no token has come back yet
                         (queueing, a router holding it back, prefill)
    reasoning, decoding  tokens are streaming back
    action               between the end of a round and the next round being
                         sent: the recorded tool wait
Counters become rates over a 10 s window: (value now - value 10 s ago) / 10 s.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# The reference palette (dataviz skill, references/palette.md), light mode.
# Slots 1-4 in fixed order; they validate as adjacent pairs for lines.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
INK = "#52514e"
GRID = "#e4e3df"
ENGINE_COLOURS = (BLUE, ORANGE, AQUA, YELLOW)

PHASE_COLOURS = {
    "reasoning, waiting": AQUA,
    "reasoning, decoding": BLUE,
    "action": ORANGE,
}
WINDOW_SECONDS = 10.0  # rates are averaged over the last 10 s: 1 s is too noisy


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def by_engine(samples: list[dict]) -> dict[str, list[dict]]:
    engines: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        engines[sample.get("engine", "engine")].append(sample)
    return dict(engines)


def window_pairs(samples: list[dict], window_seconds: float):
    """(earlier, later) sample pairs about `window_seconds` apart, one per sample."""
    earliest = 0
    for later in samples:
        while samples[earliest + 1]["t"] <= later["t"] - window_seconds:
            earliest += 1
        earlier = samples[earliest]
        if later["t"] > earlier["t"]:
            yield earlier, later


def rates(
    samples: list[dict], name: str, window_seconds: float = WINDOW_SECONDS
) -> tuple[list[float], list[float]]:
    """Rate of one engine's counter over the last window, at the window's end."""
    times, values = [], []
    for earlier, later in window_pairs(samples, window_seconds):
        if name in later and name in earlier:
            times.append(later["t"])
            values.append((later[name] - earlier[name]) / (later["t"] - earlier["t"]))
    return times, values


def summed_over_engines(
    engines: dict[str, list[dict]], series, step: float = 1.0
) -> tuple[list[float], list[float]]:
    """Add one per-engine series across engines on a common 1 s grid (each
    engine's value is its latest sample at or before the grid point)."""
    per_engine = [series(samples) for samples in engines.values()]
    end = max((times[-1] for times, _ in per_engine if times), default=0.0)
    grid = [index * step for index in range(int(end / step) + 1)]
    totals = [0.0] * len(grid)
    for times, values in per_engine:
        position = -1
        for index, moment in enumerate(grid):
            while position + 1 < len(times) and times[position + 1] <= moment:
                position += 1
            if position >= 0:
                totals[index] += values[position]
    return grid, totals


def gauge(name: str, scale: float = 1.0):
    return lambda samples: (
        [s["t"] for s in samples],
        [scale * s.get(name, 0.0) for s in samples],
    )


def capacities(run: Path, engine_capacity: Path | None) -> dict[str, int]:
    """engine URL -> KV-cache capacity in tokens (device plus host memory)."""
    summary_path = run / "summary.json"
    recorded = []
    if summary_path.exists():
        recorded = json.loads(summary_path.read_text()).get("engines") or []
    if not recorded and engine_capacity:
        recorded = [json.loads(engine_capacity.read_text()) | {"endpoint": "engine"}]
    if not recorded:
        raise SystemExit(
            "no KV capacity: the run's summary.json has no `engines`; "
            "pass --engine-capacity <machine>.json (engine_capacity.py)"
        )
    return {
        entry["endpoint"]: entry["kv_capacity_tokens_device"]
        + entry["kv_capacity_tokens_host"]
        for entry in recorded
    }


def style(axis, ylabel: str) -> None:
    axis.set_ylabel(ylabel, color=INK)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.tick_params(colors=INK)
    ticker = __import__("matplotlib").ticker
    axis.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axis.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def engine_label(url: str) -> str:
    return url.replace("http://", "").removesuffix("/v1")


def plot_kv_tokens(axis, engines, capacity_tokens, *, summed: bool) -> None:
    if summed:
        total_capacity = sum(capacity_tokens.get(e, 0) for e in engines)
        times, values = summed_over_engines(
            engines,
            lambda samples: gauge(
                "vllm:kv_cache_usage_perc",
                capacity_tokens.get(samples[0].get("engine", "engine"), 0),
            )(samples),
        )
        axis.plot(times, values, color=BLUE, linewidth=2)
        axis.axhline(total_capacity, color=INK, linewidth=1.2, linestyle="--")
        axis.text(
            0, total_capacity, f" capacity {total_capacity:,} tokens",
            color=INK, va="bottom",
        )  # fmt: skip
        style(axis, "KV cache in use\n(tokens, all engines)")
        return
    for colour, (engine, samples) in zip(ENGINE_COLOURS, engines.items(), strict=False):
        capacity = capacity_tokens.get(engine, 0)
        times, values = gauge("vllm:kv_cache_usage_perc", capacity)(samples)
        axis.plot(times, values, color=colour, linewidth=2, label=engine_label(engine))
    distinct = sorted(set(capacity_tokens.values()))
    for capacity in distinct:
        axis.axhline(capacity, color=INK, linewidth=1.2, linestyle="--")
        axis.text(0, capacity, f" capacity {capacity:,} tokens per engine",
                  color=INK, va="bottom")  # fmt: skip
    if len(engines) > 1:
        axis.legend(frameon=False, loc="upper right", ncol=len(engines))
    style(axis, "KV cache in use\n(tokens)")


def plot_engine(pyplot, samples, capacity_tokens, out: Path, title: str) -> None:
    engines = by_engine(samples)
    figure, axes = pyplot.subplots(5, 1, figsize=(12, 13), sharex=True)
    plot_kv_tokens(axes[0], engines, capacity_tokens, summed=False)

    for name, colour, label in (
        ("vllm:num_requests_running", BLUE, "running"),
        ("vllm:num_requests_waiting", ORANGE, "waiting"),
    ):
        times, values = summed_over_engines(engines, gauge(name))
        axes[1].plot(times, values, color=colour, linewidth=2, label=label)
    axes[1].legend(frameon=False, loc="upper right")
    style(axes[1], "requests\n(all engines)")

    for axis, name, label in (
        (axes[2], "vllm:prompt_tokens_total", "prefill\n(input tokens/s)"),
        (axes[3], "vllm:generation_tokens_total", "decode\n(output tokens/s)"),
    ):
        times, values = summed_over_engines(
            engines, lambda samples, name=name: rates(samples, name)
        )
        axis.plot(times, values, color=BLUE, linewidth=2)
        style(axis, label + f"\n{WINDOW_SECONDS:.0f} s average")

    for name, colour, label in (
        ("vllm:prefix_cache_queries_total", INK, "looked up"),
        ("vllm:prefix_cache_hits_total", AQUA, "found (not recomputed)"),
    ):
        times, values = summed_over_engines(
            engines, lambda samples, name=name: rates(samples, name)
        )
        axes[4].plot(times, values, color=colour, linewidth=2, label=label)
    preempted = []
    for samples in engines.values():
        times, values = rates(samples, "vllm:num_preemptions_total", 1.0)
        preempted += [t for t, v in zip(times, values, strict=True) if v]
    for moment in preempted:
        axes[4].axvline(moment, color=ORANGE, linewidth=1)
    axes[4].legend(frameon=False, loc="upper right")
    style(
        axes[4],
        f"prefix cache\n(tokens/s, {WINDOW_SECONDS:.0f} s average)"
        + (f"\norange line: preemption ({len(preempted)})" if preempted else ""),
    )
    axes[4].set_xlabel("seconds since the run started", color=INK)
    figure.suptitle(title, color=INK)
    figure.tight_layout()
    figure.savefig(out, dpi=130)
    pyplot.close(figure)


LATENCY_BIN_SECONDS = 60.0  # one point per 60 s of sends


def binned_percentiles(
    records: list[dict], latency, bin_seconds: float = LATENCY_BIN_SECONDS
) -> dict[str, tuple[list[float], list[float]]]:
    """Group requests by when they were sent (60 s bins); per bin, the median,
    the 90th percentile and the maximum of `latency(record)`, at the bin's
    middle. Not a 99th percentile: late in a run a bin holds a dozen requests,
    and the 99th percentile of a dozen values is just the maximum, so it is
    called that. Bins with fewer than 5 requests are left out."""
    bins: dict[int, list[float]] = defaultdict(list)
    for record in records:
        bins[int(record["sent_seconds"] // bin_seconds)].append(latency(record))
    lines = {
        "median": ([], []),
        "90th percentile": ([], []),
        "maximum": ([], []),
    }
    for index in sorted(bins):
        values = sorted(bins[index])
        if len(values) < 5:
            continue
        middle = (index + 0.5) * bin_seconds
        for name, fraction in (("median", 0.5), ("90th percentile", 0.9),
                               ("maximum", 1.0)):  # fmt: skip
            lines[name][0].append(middle)
            lines[name][1].append(
                values[min(len(values) - 1, int(fraction * len(values)))]
            )
    return lines


def log_seconds_axis(axis, label: str) -> None:
    """A log-scale seconds axis labelled 0.1, 0.2, 0.5, 1, 2, 5, 10 ... in plain
    numbers (matplotlib's default writes 6x10^-1 and leaves decades unlabelled)."""
    ticker = __import__("matplotlib").ticker
    axis.set_yscale("log")
    style(axis, label)
    axis.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.yaxis.set_minor_formatter(ticker.NullFormatter())


def plot_latency(pyplot, records: list[dict], out: Path, title: str) -> None:
    ok = [r for r in records if r["status"] == "ok" and r["first_token_seconds"]]
    figure, axes = pyplot.subplots(2, 1, figsize=(12, 7), sharex=True)
    for axis, latency, label in (
        (axes[0], lambda r: r["first_token_seconds"] - r["sent_seconds"],
         "time to first token\n(seconds, log scale)"),
        (axes[1], lambda r: r["finished_seconds"] - r["sent_seconds"],
         "end-to-end latency\n(seconds, log scale)"),
    ):  # fmt: skip
        for (name, (times, values)), colour in zip(
            binned_percentiles(ok, latency).items(), (AQUA, BLUE, ORANGE), strict=True
        ):
            axis.plot(times, values, color=colour, linewidth=2, marker="o",
                      markersize=4, label=name)  # fmt: skip
        log_seconds_axis(axis, label)
        axis.legend(frameon=False, loc="upper right")
    axes[1].set_xlabel("when the requests were sent (seconds since start)", color=INK)
    figure.suptitle(
        f"{title}  (requests grouped by {LATENCY_BIN_SECONDS:.0f} s of send time)",
        color=INK,
    )
    figure.tight_layout()
    figure.savefig(out, dpi=130)
    pyplot.close(figure)


def program_phases(rounds: list[dict]) -> dict[str, list[tuple[str, float, float]]]:
    """program -> [(phase, begins, ends)] in time order."""
    by_program: dict[str, list[dict]] = defaultdict(list)
    for round_ in rounds:
        by_program[round_["session_id"]].append(round_)
    phases: dict[str, list[tuple[str, float, float]]] = {}
    for program, program_rounds in by_program.items():
        program_rounds.sort(key=lambda r: r["round_idx"])
        blocks = []
        for index, r in enumerate(program_rounds):
            first = r["first_token_seconds"] or r["finished_seconds"]
            blocks.append(("reasoning, waiting", r["sent_seconds"], first))
            blocks.append(("reasoning, decoding", first, r["finished_seconds"]))
            if index + 1 < len(program_rounds):
                next_sent = program_rounds[index + 1]["sent_seconds"]
                blocks.append(("action", r["finished_seconds"], next_sent))
        phases[program] = blocks
    return phases


def plot_programs(
    pyplot, rounds, samples, capacity_tokens, out: Path, title: str
) -> None:
    phases = program_phases(rounds)
    order = sorted(phases, key=lambda program: phases[program][0][1])
    end = max(block[2] for blocks in phases.values() for block in blocks)
    row_inches = 0.3 if len(order) <= 30 else 0.12  # many programs: thinner rows
    gantt_inches = max(3.0, row_inches * len(order))
    figure, axes = pyplot.subplots(
        3,
        1,
        figsize=(13, gantt_inches + 5.5),
        sharex=True,
        gridspec_kw={"height_ratios": [gantt_inches, 2.4, 1.9]},
    )

    for row, program in enumerate(order):
        for phase, begins, ends in phases[program]:
            if ends > begins:
                axes[0].broken_barh(
                    [(begins, ends - begins)],
                    (row - 0.38, 0.76),
                    facecolors=PHASE_COLOURS[phase],
                    edgecolor="white",
                    linewidth=0.5,
                )
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order, fontsize=7 if len(order) <= 30 else 4, color=INK)
    axes[0].set_ylim(len(order) - 0.5, -0.5)  # no empty band below the last row
    axes[0].set_ylabel("program (TraceLab session id)", color=INK)
    axes[0].grid(True, axis="x", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        axes[0].spines[side].set_visible(False)
    handles = [
        pyplot.Rectangle((0, 0), 1, 1, color=colour)
        for colour in PHASE_COLOURS.values()
    ]
    axes[0].legend(
        handles, list(PHASE_COLOURS), frameon=False, ncol=3, loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
    )  # fmt: skip

    step = 0.5
    grid = [index * step for index in range(int(end / step) + 2)]
    counts = {phase: [0] * len(grid) for phase in PHASE_COLOURS}
    for blocks in phases.values():
        for phase, begins, ends in blocks:
            for index in range(
                int(begins / step), min(len(grid), int(ends / step) + 1)
            ):
                if begins <= grid[index] < ends:
                    counts[phase][index] += 1
    axes[1].stackplot(
        grid, *counts.values(), colors=list(PHASE_COLOURS.values()),
        edgecolor="white", linewidth=0.3,
    )  # fmt: skip
    style(axes[1], "programs in\neach phase\n(programs)")

    plot_kv_tokens(axes[2], by_engine(samples), capacity_tokens, summed=True)
    axes[2].set_xlabel("seconds since the run started", color=INK)
    figure.suptitle(title, color=INK, y=0.995)
    figure.tight_layout()
    figure.savefig(out, dpi=130)
    pyplot.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--engine-capacity",
        type=Path,
        help="engine_capacity.py JSON, for runs whose summary.json has no `engines`",
    )
    arguments = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    run = arguments.run
    figures = run / "figures"
    figures.mkdir(exist_ok=True)
    samples = read_jsonl(run / "metrics.jsonl")
    rounds = read_jsonl(run / "rounds.jsonl")
    requests = rounds or read_jsonl(run / "requests.jsonl")
    title = run.name
    written = []
    capacity_tokens = capacities(run, arguments.engine_capacity) if samples else {}
    if samples and "engine" not in samples[0]:  # recorded before multi-engine runs
        only = next(iter(capacity_tokens))
        samples = [s | {"engine": only} for s in samples]
    if samples:
        plot_engine(
            pyplot, samples, capacity_tokens, figures / "engine_over_time.png", title
        )
        written.append("engine_over_time.png")
    plot_latency(pyplot, requests, figures / "latency_over_time.png", title)
    written.append("latency_over_time.png")
    if rounds:
        plot_programs(
            pyplot,
            rounds,
            samples,
            capacity_tokens,
            figures / "programs_over_time.png",
            title,
        )
        written.append("programs_over_time.png")
    print(f"wrote {', '.join(written)} to {figures}")


if __name__ == "__main__":
    main()
