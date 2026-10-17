#!/usr/bin/env python3
"""How programs arrive and what happens to each one after it arrives, for each
configuration of one concurrency point.

    python experiments/workload/plot_program_arrivals.py \\
        data/results/workload/<experiment>/concurrency_<C>

Reads <point>/<configuration>/{rounds.jsonl, summary.json, thunderagent.log} and
writes <point>/figures/G1_program_arrivals.png, one column per configuration:

    arrivals        a tick at every external program arrival, and
                    arrivals per 60 s
    so far          programs arrived, programs finished, programs in the system
    each program    one row per program in arrival order, from its arrival to
                    its last token: held by ThunderAgent / waiting in the engine /
                    decoding / tool (action), as in F1_program_phases.png

In open-loop mode, a program "arrives" at the timestamp recorded by the workload,
before any client, router, or engine waiting. In the legacy closed-loop mode,
there is no external arrival timestamp, so the first send remains the fallback.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from plot_observability import log_events, program_intervals, read_jsonl
from plot_program_phases import PHASES, intersect, subtract
from plot_run import BLUE, INK, YELLOW, style

COLOURS = dict(PHASES)


def programs_of(run: Path):
    rounds = [r for r in read_jsonl(run / "rounds.jsonl") if r["status"] == "ok"]
    summary = json.loads((run / "summary.json").read_text())
    events = log_events(run / "thunderagent.log", summary["started_at_wall"])
    held = {
        program: [(s, e) for s, e, _kind in spans]
        for program, spans in program_intervals(events["decisions"]).items()
    }
    by_program = defaultdict(list)
    for r in rounds:
        by_program[r["session_id"]].append(r)
    end = max(r["finished_seconds"] for r in rounds)
    programs = []
    for program, items in by_program.items():
        items.sort(key=lambda r: r["round_idx"])
        spans = held.get(program, [])
        blocks = []
        arrival = items[0].get("program_arrival_seconds")
        arrival = items[0]["sent_seconds"] if arrival is None else arrival
        if items[0]["sent_seconds"] > arrival:
            blocks.append(
                (arrival, items[0]["sent_seconds"], "waiting before first send")
            )
        for index, r in enumerate(items):
            waiting = (r["sent_seconds"], r["first_token_seconds"])
            blocks += [
                (*piece, "held by ThunderAgent") for piece in intersect(waiting, spans)
            ]
            blocks += [
                (*piece, "waiting in the engine") for piece in subtract(waiting, spans)
            ]
            blocks.append((r["first_token_seconds"], r["finished_seconds"], "decoding"))
            tool_end = (
                items[index + 1]["sent_seconds"]
                if index + 1 < len(items)
                else min(r["finished_seconds"] + r["tool_wait_after_ms"] / 1000, end)
            )
            if tool_end > r["finished_seconds"]:
                blocks.append((r["finished_seconds"], tool_end, "tool (action)"))
        programs.append(
            {
                "arrival": arrival,
                "finished": items[-1]["finished_seconds"]
                if items[-1]["tool_wait_after_ms"] == 0
                else None,
                "blocks": blocks,
            }
        )
    programs.sort(key=lambda p: p["arrival"])
    return programs, end


def main() -> None:
    point = Path(sys.argv[1])
    runs = sorted(
        (p.parent for p in point.glob("*/rounds.jsonl")),
        key=lambda run: (run.name != "vllm-direct", run.name),
    )
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    data = {run.name: programs_of(run) for run in runs}
    most = max(len(programs) for programs, _ in data.values())
    figure, axes = pyplot.subplots(
        3,
        len(runs),
        figsize=(9 * len(runs), 8 + 0.06 * most),
        squeeze=False,
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.4, max(4, 0.06 * most)]},
    )
    x_end = max(end for _, end in data.values()) + 20
    for column, (name, (programs, end)) in enumerate(data.items()):
        top, middle, bottom = (axes[row][column] for row in range(3))
        arrivals = [p["arrival"] for p in programs]
        finished = sorted(p["finished"] for p in programs if p["finished"] is not None)
        top.vlines(
            arrivals,
            0,
            1,
            color=INK,
            linewidth=0.6,
            transform=top.get_xaxis_transform(),
        )
        bins = list(range(0, int(x_end) + 60, 60))
        per_bin = [sum(1 for a in arrivals if b <= a < b + 60) for b in bins[:-1]]
        twin = top.twinx()
        twin.bar(bins[:-1], per_bin, width=55, align="edge", color=YELLOW, alpha=0.5)
        twin.set_ylabel("arrivals per 60 s", color=INK)
        style(top, "arrival (one tick each)")
        top.set_yticks([])
        top.set_title(
            f"{name}: {len(programs)} programs arrived, {len(finished)} finished",
            color=INK,
        )
        middle.step(
            arrivals,
            range(1, len(arrivals) + 1),
            where="post",
            color=YELLOW,
            linewidth=2,
            label="arrived so far",
        )
        middle.step(
            finished,
            range(1, len(finished) + 1),
            where="post",
            color=BLUE,
            linewidth=2,
            label="finished so far",
        )
        in_system = [
            (t, sum(1 for a in arrivals if a <= t) - sum(1 for f in finished if f <= t))
            for t in range(0, int(end) + 1, 5)
        ]
        middle.plot(
            [t for t, _ in in_system],
            [n for _, n in in_system],
            color=INK,
            linewidth=1,
            linestyle="--",
            label="in the system (arrived, not finished)",
        )
        style(middle, "programs (count)")
        middle.legend(frameon=False, fontsize=8, loc="upper left")
        for row, program in enumerate(programs):
            for start, stop, phase in program["blocks"]:
                bottom.barh(
                    row,
                    stop - start,
                    left=start,
                    height=0.9,
                    color=COLOURS[phase],
                    linewidth=0,
                )
            if program["finished"] is not None:
                bottom.plot(
                    program["finished"], row, marker="|", color=INK, markersize=4
                )
        bottom.set_ylim(len(programs), -1)
        style(bottom, "programs in arrival order (one row each)")
        bottom.set_xlabel(
            "seconds since the run started (| = program finished)", color=INK
        )
        bottom.set_xlim(0, x_end)
    handles = [pyplot.Rectangle((0, 0), 1, 1, color=colour) for _, colour in PHASES]
    figure.legend(
        handles,
        [phase for phase, _ in PHASES],
        loc="lower center",
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        f"{point.parent.name} / {point.name}: "
        "how programs arrive and what happens to each",
        color=INK,
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.98))
    out = point / "figures"
    out.mkdir(exist_ok=True)
    figure.savefig(out / "G1_program_arrivals.png", dpi=100)
    print("wrote", out / "G1_program_arrivals.png")


if __name__ == "__main__":
    main()
