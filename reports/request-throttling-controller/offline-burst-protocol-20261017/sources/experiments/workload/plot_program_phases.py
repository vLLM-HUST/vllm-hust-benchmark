#!/usr/bin/env python3
"""Where every program is, second by second, for each configuration of one
concurrency point -- the figure that shows whether programs arrive once and the
run is then only waiting for them, or programs keep flowing through.

    python experiments/workload/plot_program_phases.py \\
        data/results/workload/<experiment>/concurrency_<C>

Reads <point>/<configuration>/{rounds.jsonl, summary.json, thunderagent.log} and
writes <point>/figures/F1_program_phases.png. One column per configuration, the
same y scale in every column of a row.

Top row: programs in each phase at each second (stacked):
    held by ThunderAgent      inside a ThunderAgent pause or new-program queue
                              (from its log: "Paused ... Resumed", "Queued ...
                              Assigned"), while the client waits for a token
    waiting in the engine     the rest of send -> first token (engine queue and
                              prefill)
    decoding                  first token -> last token
    tool (action)             last token -> the program's next send (the client
                              sleeps the recorded tool time)
Bottom row: programs started and programs finished so far.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from plot_observability import log_events, program_intervals, read_jsonl
from plot_run import AQUA, BLUE, INK, ORANGE, YELLOW, style

PHASES = (
    ("waiting before first send", YELLOW),
    ("held by ThunderAgent", "#d62728"),
    ("waiting in the engine", AQUA),
    ("decoding", BLUE),
    ("tool (action)", ORANGE),
)


def subtract(interval, holes):
    """Parts of `interval` (start, end) outside every (start, end) in `holes`."""
    pieces = [interval]
    for hole_start, hole_end in holes:
        next_pieces = []
        for start, end in pieces:
            if hole_end <= start or hole_start >= end:
                next_pieces.append((start, end))
                continue
            if start < hole_start:
                next_pieces.append((start, hole_start))
            if hole_end < end:
                next_pieces.append((hole_end, end))
        pieces = next_pieces
    return pieces


def intersect(interval, spans):
    start, end = interval
    return [
        (max(start, s), min(end, e)) for s, e in spans if min(end, e) > max(start, s)
    ]


def phase_counts(run: Path):
    """seconds, {phase: programs at each second}, started, finished."""
    rounds = [r for r in read_jsonl(run / "rounds.jsonl") if r["status"] == "ok"]
    summary = json.loads((run / "summary.json").read_text())
    events = log_events(run / "thunderagent.log", summary["started_at_wall"])
    held = {
        program: [(s, e) for s, e, _kind in spans]
        for program, spans in program_intervals(events["decisions"]).items()
    }
    end = int(max(r["finished_seconds"] for r in rounds)) + 1
    counts = {name: [0.0] * (end + 1) for name, _ in PHASES}

    def add(phase, start, stop):
        # time-weighted: a second partly covered counts that fraction
        for second in range(int(start), min(int(stop) + 1, end + 1)):
            overlap = min(stop, second + 1) - max(start, second)
            if overlap > 0:
                counts[phase][second] += overlap

    by_program = defaultdict(list)
    for r in rounds:
        by_program[r["session_id"]].append(r)
    started, finished = [], []
    for program, items in by_program.items():
        items.sort(key=lambda r: r["round_idx"])
        arrival = items[0].get("program_arrival_seconds")
        arrival = items[0]["sent_seconds"] if arrival is None else arrival
        started.append(arrival)
        add("waiting before first send", arrival, items[0]["sent_seconds"])
        if items[-1]["tool_wait_after_ms"] == 0:  # the last round of the program
            finished.append(items[-1]["finished_seconds"])
        spans = held.get(program, [])
        for index, r in enumerate(items):
            waiting = (r["sent_seconds"], r["first_token_seconds"])
            for piece in intersect(waiting, spans):
                add("held by ThunderAgent", *piece)
            for piece in subtract(waiting, spans):
                add("waiting in the engine", *piece)
            add("decoding", r["first_token_seconds"], r["finished_seconds"])
            if index + 1 < len(items):
                tool_end = items[index + 1]["sent_seconds"]
            else:
                tool_end = min(
                    r["finished_seconds"] + r["tool_wait_after_ms"] / 1000, end
                )
            add("tool (action)", r["finished_seconds"], tool_end)
    return list(range(end + 1)), counts, sorted(started), sorted(finished)


def main() -> None:
    point = Path(sys.argv[1])
    # vllm-direct first, then ThunderAgent, as in every other figure
    runs = sorted(
        (p.parent for p in point.glob("*/rounds.jsonl")),
        key=lambda run: (run.name != "vllm-direct", run.name),
    )
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    data = {run.name: phase_counts(run) for run in runs}
    figure, axes = pyplot.subplots(
        2, len(runs), figsize=(8 * len(runs), 8), squeeze=False, sharey="row"
    )
    for column, (name, (seconds, counts, started, finished)) in enumerate(data.items()):
        top, bottom = axes[0][column], axes[1][column]
        top.stackplot(
            seconds,
            *[counts[phase] for phase, _ in PHASES],
            labels=[phase for phase, _ in PHASES],
            colors=[colour for _, colour in PHASES],
            linewidth=0,
        )
        style(top, "programs in each phase (count, 1 s)")
        top.set_title(name, color=INK)
        top.legend(
            fontsize=8,
            loc="upper center",
            ncol=4,
            framealpha=0.9,
            bbox_to_anchor=(0.5, -0.04),
        )
        bottom.step(started, range(1, len(started) + 1), where="post",
                    color=YELLOW, linewidth=2,
                    label=f"started ({len(started)})")  # fmt: skip
        bottom.step(finished, range(1, len(finished) + 1), where="post",
                    color=BLUE, linewidth=2,
                    label=f"finished ({len(finished)})")  # fmt: skip
        style(bottom, "programs so far (count)")
        bottom.set_xlabel("seconds since the run started", color=INK)
        bottom.legend(frameon=False, fontsize=8, loc="upper left")
        for axis in (top, bottom):
            axis.set_xlim(0, max(len(s) for s, *_ in data.values()))
    figure.suptitle(f"{point.parent.name} / {point.name}: where the programs are",
                    color=INK)  # fmt: skip
    figure.tight_layout()
    out = point / "figures"
    out.mkdir(exist_ok=True)
    figure.savefig(out / "F1_program_phases.png", dpi=110)
    print("wrote", out / "F1_program_phases.png")


if __name__ == "__main__":
    main()
