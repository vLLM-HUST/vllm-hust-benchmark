"""What every program experienced under open-loop arrivals, by priority class:
the figure and table the streaming experiment is judged on.

    python experiments/workload/plot_open_loop_programs.py \
        data/results/workload/<experiment>

Reads <experiment>/<configuration>/{rounds.jsonl, summary.json, thunderagent.log}
as program_configurations.sh writes it with DRAIN_AFTER_ARRIVALS=1, and writes
<experiment>/figures/program_end_to_end_latency_and_completion_comparison.png plus
<experiment>/report/tables/
program_end_to_end_latency_and_completion_statistics.csv, also printed as Markdown.

Why this exists next to plot_program_arrivals.py (G1) and
plot_observability_compare.py (E1/E2): G1 draws one row per program against
time, E2 sums the wait for a first token per program. Neither answers the
question this experiment asks -- for each priority class, how long did a
program take from its external arrival, and where did that time go -- so the
phase decomposition is reused here and aggregated per program instead.

Definitions (the same ones G1 draws, so a program's bar and its row agree):
    arrival         the timestamp the workload fixed before the run; no client,
                    router, or engine waiting is inside it
    end-to-end      last token of the program's last round - arrival
    held by ThunderAgent    inside one of its pauses or its new-program queue
                    (from its own log), while the client waits for a token
    time to first token outside router holds   the rest of send -> first token
    decoding        first token -> last token of a round
    tool (action)    last token -> that program's next send
    waiting         router holds + remaining time to first token + the gap
                    between arrival and the first send
    work            decoding + tool (action): what the program would spend if
                    nothing ever made it wait

Panels:
    1  per-program end-to-end latency, sorted; one series per configuration,
       urgent filled and ordinary hollow, with each class's latency objective
    2  where the time went, per program, stacked, sorted by end-to-end latency
    3  the program's largest context against its waiting: the direct test of
       whether a scheduler that admits the smallest context first leaves the
       large, long-lived programs waiting
    4  programs arrived, completed, and in the system, against time

Every axis is an absolute quantity with its unit; nothing is a percentage.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from plot_observability import log_events, program_intervals, read_jsonl
from plot_program_phases import intersect, subtract
from plot_run import AQUA, BLUE, GRID, INK, ORANGE, YELLOW, style

HELD = "#d62728"
WAITING_PHASES = ("waiting before first send", "held by ThunderAgent",
                  "time to first token outside router holds")  # fmt: skip
DECOMPOSITION = (
    ("waiting before first send", YELLOW),
    ("held by ThunderAgent", HELD),
    ("time to first token outside router holds", AQUA),
    ("decoding", BLUE),
    ("tool (action)", ORANGE),
)


def programs_of(run: Path) -> list[dict]:
    """One record per program that arrived, with its phase seconds summed.

    The phase decomposition is plot_program_arrivals.programs_of()'s, kept
    identical on purpose; what is added is the program's identity, priority
    class, latency objective, and largest context, so the records can be
    aggregated per class and compared across configurations.
    """
    rounds = [r for r in read_jsonl(run / "rounds.jsonl") if r["status"] == "ok"]
    summary = json.loads((run / "summary.json").read_text())
    events = log_events(run / "thunderagent.log", summary["started_at_wall"])
    held = {
        program: [(start, end) for start, end, _kind in spans]
        for program, spans in program_intervals(events["decisions"]).items()
    }
    expected = {
        program: len(items)
        for program, items in _rounds_of_workload(summary, run).items()
    }  # fmt: skip
    by_program: dict[str, list[dict]] = {}
    for round_ in rounds:
        by_program.setdefault(round_["session_id"], []).append(round_)
    end = summary.get("observed_seconds") or max(
        (r["finished_seconds"] for r in rounds), default=0.0
    )

    programs = []
    for program, items in by_program.items():
        items.sort(key=lambda r: r["round_idx"])
        if items[0].get("program_arrival_seconds") is None:
            continue  # closed loop: there is no external arrival to measure from
        arrival = float(items[0]["program_arrival_seconds"])
        spans = held.get(program, [])
        seconds = dict.fromkeys((phase for phase, _colour in DECOMPOSITION), 0.0)
        seconds["waiting before first send"] = max(
            0.0, items[0]["sent_seconds"] - arrival
        )
        for index, round_ in enumerate(items):
            first_token = round_["first_token_seconds"]
            if first_token is None:
                first_token = round_["finished_seconds"]
            waiting = (round_["sent_seconds"], first_token)
            seconds["held by ThunderAgent"] += _length(intersect(waiting, spans))
            seconds["time to first token outside router holds"] += _length(
                subtract(waiting, spans)
            )
            seconds["decoding"] += round_["finished_seconds"] - first_token
            tool_end = (
                items[index + 1]["sent_seconds"]
                if index + 1 < len(items)
                else round_["finished_seconds"]
            )
            seconds["tool (action)"] += max(0.0, tool_end - round_["finished_seconds"])
        complete = len(items) == expected.get(program, len(items))
        programs.append(
            {
                "program": program,
                "priority_class": items[0]["priority_class"],
                "latency_objective_seconds": items[0]["latency_objective_seconds"],
                "arrival_seconds": arrival,
                "finished_seconds": items[-1]["finished_seconds"],
                "end_to_end_seconds": items[-1]["finished_seconds"] - arrival,
                "complete": complete,
                "rounds": len(items),
                "largest_context_tokens": max(
                    (r["engine_prompt_tokens"] or r["prefix_len"] + r["input_len"])
                    for r in items
                ),
                "seconds": seconds,
                "waiting_seconds": sum(seconds[p] for p in WAITING_PHASES),
                "work_seconds": seconds["decoding"] + seconds["tool (action)"],
            }
        )
    # Arrived programs with no successful round must remain in the denominator.
    seen = {p["program"] for p in programs}
    for program, declared in _rounds_of_workload(summary, run).items():
        arrival = int(declared[0]["arrival_time"]) / 1000
        if program in seen or arrival > end:
            continue
        programs.append(
            {
                "program": program,
                "priority_class": declared[0].get("priority_class") or "ordinary",
                "latency_objective_seconds": declared[0].get(
                    "latency_objective_seconds"
                )
                or None,
                "arrival_seconds": arrival,
                "finished_seconds": None,
                "end_to_end_seconds": None,
                "complete": False,
                "rounds": 0,
                "largest_context_tokens": max(
                    int(r["prefix_len"]) + int(r["input_len"]) for r in declared
                ),
                "seconds": dict.fromkeys((phase for phase, _ in DECOMPOSITION), 0.0),
                "waiting_seconds": None,
                "work_seconds": None,
            }
        )
    programs.sort(key=lambda p: p["arrival_seconds"])
    return programs


def _length(pieces) -> float:
    return sum(end - start for start, end in pieces)


def _rounds_of_workload(summary: dict, run: Path | None = None) -> dict[str, list]:
    """program -> its rounds as the workload declared them (to spot truncation)."""
    manifest = summary.get("workload_manifest") or {}
    sessions_csv = Path(summary["parameters"]["workload"]) / "sessions.csv"
    if not sessions_csv.exists():  # the host path is not this machine's path
        local = Path(__file__).resolve().parents[2] / "data" / "workloads"
        name = Path(summary["parameters"]["workload"]).name
        sessions_csv = local / name / "sessions.csv"
    if run is not None and (run / "sessions.csv").exists():
        sessions_csv = run / "sessions.csv"
    rounds: dict[str, list] = {}
    if not sessions_csv.exists():
        del manifest
        return rounds
    with sessions_csv.open() as source:
        for row in csv.DictReader(source):
            rounds.setdefault(row["session_id"], []).append(row)
    return rounds


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def table(by_configuration: dict[str, list[dict]]) -> list[dict]:
    """Per configuration and priority class; programs that did not finish are
    counted but kept out of the latency statistics (they have no latency yet)."""
    rows = []
    for configuration, programs in by_configuration.items():
        classes = sorted({p["priority_class"] for p in programs})
        for priority_class in [*classes, "all"]:
            group = [
                p
                for p in programs
                if priority_class in ("all", p["priority_class"])
            ]  # fmt: skip
            done = [p for p in group if p["complete"]]
            latency = [p["end_to_end_seconds"] for p in done]
            waiting = [p["waiting_seconds"] for p in done]
            attained = [
                p
                for p in done
                if p["latency_objective_seconds"] is not None
                and p["end_to_end_seconds"] <= float(p["latency_objective_seconds"])
            ]
            rows.append(
                {
                    "configuration": configuration,
                    "priority_class": priority_class,
                    "programs_arrived": len(group),
                    "programs_completed": len(done),
                    "programs_unfinished": len(group) - len(done),
                    "latency_population": "completed programs only",
                    "end_to_end_median_s": _round(percentile(latency, 0.5)),
                    "end_to_end_90th_s": _round(percentile(latency, 0.9)),
                    "end_to_end_maximum_s": _round(max(latency, default=None)),
                    "waiting_median_s": _round(percentile(waiting, 0.5)),
                    "waiting_maximum_s": _round(max(waiting, default=None)),
                    "waiting_total_s": _round(sum(waiting)),
                    "programs_attaining_objective": len(attained),
                    "latency_objective_s": _objective(group),
                }
            )
    return rows


def _objective(group: list[dict]) -> float | None:
    objectives = {
        float(p["latency_objective_seconds"])
        for p in group
        if p["latency_objective_seconds"] is not None
    }
    return objectives.pop() if len(objectives) == 1 else None


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


def identical_set(by_configuration: dict[str, list[dict]]) -> list[dict]:
    """The comparison that cannot flatter a scheduler: the same programs, both
    completed, every program's own total, and how many got better or worse."""
    configurations = list(by_configuration)
    if len(configurations) != 2:
        return []
    left, right = configurations
    index = {
        configuration: {p["program"]: p for p in programs if p["complete"]}
        for configuration, programs in by_configuration.items()
    }
    shared = sorted(set(index[left]) & set(index[right]))
    rows = []
    classes = sorted({index[left][p]["priority_class"] for p in shared})
    for priority_class in ["all", *classes]:
        members = [
            p
            for p in shared
            if priority_class in ("all", index[left][p]["priority_class"])
        ]  # fmt: skip
        if not members:
            continue
        a = [index[left][p]["end_to_end_seconds"] for p in members]
        b = [index[right][p]["end_to_end_seconds"] for p in members]
        rows.append(
            {
                "priority_class": priority_class,
                "programs_compared": len(members),
                f"{left}_total_s": _round(sum(a)),
                f"{right}_total_s": _round(sum(b)),
                f"{left}_median_s": _round(percentile(a, 0.5)),
                f"{right}_median_s": _round(percentile(b, 0.5)),
                f"{left}_90th_s": _round(percentile(a, 0.9)),
                f"{right}_90th_s": _round(percentile(b, 0.9)),
                f"{left}_maximum_s": _round(max(a)),
                f"{right}_maximum_s": _round(max(b)),
                f"better_under_{right}": sum(
                    1 for x, y in zip(a, b, strict=True) if y < x
                ),
                f"worse_under_{right}": sum(
                    1 for x, y in zip(a, b, strict=True) if y > x
                ),
                "largest_single_increase_s": _round(
                    max((y - x for x, y in zip(a, b, strict=True)), default=0.0)
                ),
            }
        )
    return rows


def draw(by_configuration: dict[str, list[dict]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    figure, axes = pyplot.subplots(4, 1, figsize=(13, 20))
    colours = dict(zip(by_configuration, (BLUE, ORANGE, AQUA), strict=False))

    axis = axes[0]
    for configuration, programs in by_configuration.items():
        done = sorted(
            (p for p in programs if p["complete"]),
            key=lambda p: p["end_to_end_seconds"],
        )
        if not done:
            continue
        axis.plot(
            range(1, len(done) + 1),
            [p["end_to_end_seconds"] for p in done],
            color=colours[configuration], linewidth=1.6, label=configuration,
        )  # fmt: skip
        for marker, urgent in (("o", True), ("o", False)):
            points = [
                (index + 1, p["end_to_end_seconds"])
                for index, p in enumerate(done)
                if (p["priority_class"] == "urgent") is urgent
            ]
            if points:
                axis.scatter(
                    *zip(*points, strict=True), marker=marker, s=34,
                    facecolors=colours[configuration] if urgent else "none",
                    edgecolors=colours[configuration], zorder=3,
                    label=f"{configuration}, {'urgent' if urgent else 'ordinary'}",
                )  # fmt: skip
    for programs in by_configuration.values():
        for objective in sorted(
            {
                float(p["latency_objective_seconds"])
                for p in programs
                if p["latency_objective_seconds"] is not None
            }
        ):
            axis.axhline(objective, color=INK, linestyle=":", linewidth=1)
            axis.annotate(
                f"objective {objective:,.0f} s", (0.5, objective), color=INK,
                fontsize=8, va="bottom",
            )  # fmt: skip
        break
    style(axis, "program end-to-end latency (s)")
    axis.set_xlabel("programs that completed, sorted by their own latency", color=INK)
    axis.set_title(
        "Program end-to-end latency: external arrival to final token",
        color=INK, loc="left",
    )  # fmt: skip
    axis.legend(fontsize=8, frameon=False)

    axis = axes[1]
    width = 0.4
    for number, programs in enumerate(by_configuration.values()):
        done = sorted(
            (p for p in programs if p["complete"]),
            key=lambda p: p["end_to_end_seconds"],
        )
        positions = [index + number * width for index in range(len(done))]
        bottom = [0.0] * len(done)
        for phase, colour in DECOMPOSITION:
            values = [p["seconds"][phase] for p in done]
            axis.bar(
                positions, values, bottom=bottom, width=width, color=colour,
                edgecolor="none",
                label=f"{phase}" if number == 0 else None,
            )  # fmt: skip
            bottom = [b + v for b, v in zip(bottom, values, strict=True)]
    style(axis, "seconds per program")
    axis.set_xlabel(
        "one group per program, sorted by latency; bars left to right: "
        + ", ".join(by_configuration),
        color=INK,
    )
    axis.set_title("Program time breakdown", color=INK, loc="left")
    axis.legend(fontsize=8, frameon=False, ncol=3)

    axis = axes[2]
    for configuration, programs in by_configuration.items():
        done = [p for p in programs if p["complete"]]
        for urgent in (True, False):
            points = [
                (p["largest_context_tokens"], p["waiting_seconds"])
                for p in done
                if (p["priority_class"] == "urgent") is urgent
            ]
            if points:
                axis.scatter(
                    *zip(*points, strict=True), s=44,
                    facecolors=colours[configuration] if urgent else "none",
                    edgecolors=colours[configuration],
                    label=f"{configuration}, {'urgent' if urgent else 'ordinary'}",
                )  # fmt: skip
    style(axis, "waiting per program (s)")
    axis.set_xlabel("the program's largest context (prompt tokens)", color=INK)
    axis.set_title(
        "Context size and time to first token, including router holds",
        color=INK, loc="left",
    )  # fmt: skip
    axis.legend(fontsize=8, frameon=False)

    axis = axes[3]
    for programs in by_configuration.values():
        if not programs:
            continue
        arrivals = sorted(p["arrival_seconds"] for p in programs)
        axis.step(
            [0.0, *arrivals], range(0, len(arrivals) + 1), where="post", color=INK,
            linewidth=1.2, linestyle="--",
            label="arrived (the same for every configuration)",
        )  # fmt: skip
        break
    for configuration, programs in by_configuration.items():
        completions = sorted(p["finished_seconds"] for p in programs if p["complete"])
        if completions:
            axis.step(
                [0.0, *completions], range(0, len(completions) + 1), where="post",
                color=colours[configuration], linewidth=1.8,
                label=f"completed, {configuration}",
            )  # fmt: skip
        times = sorted({0.0, *(p["arrival_seconds"] for p in programs), *completions})
        backlog = [
            sum(p["arrival_seconds"] <= t for p in programs)
            - sum(finished <= t for finished in completions)
            for t in times
        ]
        axis.step(
            times,
            backlog,
            where="post",
            color=colours[configuration],
            linestyle=":",
            label=f"unfinished, {configuration}",
        )
    style(axis, "programs")
    axis.set_xlabel("seconds since the replay started", color=INK)
    axis.set_title("Programs arrived, completed, and unfinished", color=INK, loc="left")
    axis.legend(fontsize=8, frameon=False)
    axis.grid(True, color=GRID, linewidth=0.8)

    # Tiny local smoke timings must not all be rounded to zero by shared style.
    from matplotlib.ticker import StrMethodFormatter

    for time_axis in axes[:3]:
        time_axis.yaxis.set_major_formatter(StrMethodFormatter("{x:.3g}"))
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=140)
    pyplot.close(figure)


def markdown(rows: list[dict]) -> str:
    if not rows:
        return "(no rows)\n"
    columns = list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "|" + "-|" * len(columns)]
    for row in rows:
        cells = " | ".join("" if row[c] is None else str(row[c]) for c in columns)
        lines.append(f"| {cells} |")
    return "\n".join(lines) + "\n"


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    experiment = Path(sys.argv[1])
    by_configuration = {}
    for run in sorted(experiment.iterdir()):
        if not (run / "rounds.jsonl").exists() or not (run / "summary.json").exists():
            continue
        programs = programs_of(run)
        if programs:
            by_configuration[run.name] = programs
            print(
                f"[open-loop] {run.name}: {len(programs)} programs arrived, "
                f"{sum(1 for p in programs if p['complete'])} completed"
            )
    if not by_configuration:
        print(f"[open-loop] no open-loop run with rounds.jsonl under {experiment}")
        raise SystemExit(1)

    rows = table(by_configuration)
    write_csv(
        rows,
        experiment
        / "report"
        / "tables"
        / "program_end_to_end_latency_and_completion_statistics.csv",
    )
    print(markdown(rows))
    shared = identical_set(by_configuration)
    if shared:
        write_csv(
            shared,
            experiment
            / "report"
            / "tables"
            / "programs_completed_by_all_configurations_comparison.csv",
        )
        print("the same programs, both configurations, both completed:")
        print(markdown(shared))
    figure = (
        experiment
        / "figures"
        / "program_end_to_end_latency_and_completion_comparison.png"
    )
    try:
        draw(by_configuration, figure)
        print(f"[open-loop] wrote {figure}")
    except ImportError:
        print("[open-loop] matplotlib is not installed; tables only")
    tables = experiment / "report" / "tables"
    table_path = tables / "program_end_to_end_latency_and_completion_statistics.csv"
    print(f"[open-loop] wrote {table_path}")


if __name__ == "__main__":
    main()
