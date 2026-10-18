#!/usr/bin/env python3
"""Pick programs out of TraceLab's public coding-agent trace, size their context
to the machine they will run on, and write them as a session dataset. It only
writes data; it sends nothing.

    python experiments/workload/select_tracelab_sessions.py \\
        --name tracelab-context_scale_0.1-min_rounds_4-programs_16 \\
        --engine-capacity data/workloads/engine_capacity/<machine>.json \\
        --context-scale 0.1 --minimum-rounds 4 --sessions 16 \\
        --arrival-process poisson --session-rate 0.2

Source: TraceLab (uw-syfi/TraceLab, release v0.0.2, CC BY 4.0), about 8,000
Claude Code / Codex sessions. Its own exporter
(`artifacts/trace_facts/csv_export/convert.py`) turns the released DuckDB into
`data/workloads/tracelab/coding_trace.csv`, one row per LLM round:

    id,input_len,output_len,arrival_time,round_idx,tool_wait_after_ms,prefix_len

- prefix_len          tokens of earlier context the round starts with
- input_len           new tokens appended this round (user message, tool results)
- output_len          tokens the model generated
- tool_wait_after_ms  summed tool latency after this round, before the next
                      (0 on a session's last round)

Output: data/workloads/<name>/sessions.csv in the same columns (the format
TraceLab's own `session_runner` reads) and manifest.json.

What this program decides, all recorded in the manifest:
1. how long the context is, from the machine (--engine-capacity, written by
   engine_capacity.py from the running engine): every round's prefix_len and
   input_len are multiplied by --context-scale (rounded; input_len at least 1).
   output_len is NOT scaled. Tool waits are multiplied by --tool-wait-scale
   (default 1.0, i.e. as recorded; the recorded waits have a long tail, one
   program in the trace waits 47 minutes in total). The real trace's median context is
   ~132k tokens; one Qwen2.5-14B on one 910B2 holds 16,384 per request, so
   without scaling only the short, early sessions fit.
   ⚠️ What survives scaling: the number of rounds, how context grows round to
   round, tool waits, output lengths. What does not: the absolute context, and
   so the prefill-to-decode work ratio (prefill shrinks, decode does not).
2. which programs: after scaling, every round's prefix_len + input_len +
   output_len must fit the engine's max_model_len, the program needs at least
   --minimum-rounds rounds, and (if given) its total tool wait must be at most
   --maximum-tool-wait-seconds; then --sessions of them, drawn with --seed.
   ⚠️ The tool-wait bound exists because the recorded waits have a very long
   tail (at context scale 0.1: median 9 s per program, 99th percentile 2,687 s,
   largest 80,273 s); one such program would keep a run going for a day. At
   1,800 s it drops 82 of 4,907 programs (1.7%), the ones that wait longest.
3. when programs arrive: replaced by our arrival process (arrival_process.py)
   at --session-rate programs per second (TraceLab's exporter draws its own
   synthetic arrival times, so the trace has none worth keeping).
The manifest also states how the chosen programs compare with the machine's
KV-cache capacity: the sum of every program's largest context, i.e. the KV
demand if all of them were at their largest at the same moment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from arrival_process import ARRIVAL_PROCESSES, arrival_times_seconds
from generate_workload import WORKLOADS_DIRECTORY, git_state

SOURCE_CSV = WORKLOADS_DIRECTORY / "tracelab" / "coding_trace.csv"
COLUMNS = (
    "session_id",
    "arrival_time",
    "round_idx",
    "prefix_len",
    "input_len",
    "output_len",
    "tool_wait_after_ms",
)


def load_sessions(path: Path) -> dict[str, list[dict]]:
    sessions: dict[str, list[dict]] = defaultdict(list)
    with path.open() as source:
        for row in csv.DictReader(source):
            sessions[row["id"]].append(
                {
                    "round_idx": int(row["round_idx"]),
                    "prefix_len": int(row["prefix_len"]),
                    "input_len": int(row["input_len"]),
                    "output_len": int(row["output_len"]),
                    "tool_wait_after_ms": round(float(row["tool_wait_after_ms"])),
                }
            )
    for rounds in sessions.values():
        rounds.sort(key=lambda round_: round_["round_idx"])
    return sessions


def scale_context(rounds: list[dict], context_scale: float) -> list[dict]:
    return [
        r
        | {
            "prefix_len": round(r["prefix_len"] * context_scale),
            "input_len": max(1, round(r["input_len"] * context_scale)),
        }
        for r in rounds
    ]


def scale_tool_wait(rounds: list[dict], tool_wait_scale: float) -> list[dict]:
    return [
        r | {"tool_wait_after_ms": round(r["tool_wait_after_ms"] * tool_wait_scale)}
        for r in rounds
    ]


def largest_context(rounds: list[dict]) -> int:
    """The most KV-cache tokens the program needs at once: its longest round."""
    return max(r["prefix_len"] + r["input_len"] + r["output_len"] for r in rounds)


def describe(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "mean": round(statistics.fmean(ordered), 1),
        "median": ordered[len(ordered) // 2],
        "maximum": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--engine-capacity",
        type=Path,
        action="append",
        required=True,
        help="JSON written by engine_capacity.py for an engine this will run on; "
        "repeat once per engine (KV capacities add up, the smallest "
        "max_model_len applies)",
    )
    parser.add_argument("--context-scale", type=float, default=1.0)
    parser.add_argument(
        "--tool-wait-scale",
        type=float,
        default=1.0,
        help="multiplies every recorded tool wait; 1.0 keeps them as recorded",
    )
    parser.add_argument("--sessions", type=int, default=16)
    parser.add_argument("--minimum-rounds", type=int, default=1)
    parser.add_argument(
        "--maximum-tool-wait-seconds",
        type=float,
        help="drop programs whose summed tool wait is longer (default: keep all)",
    )
    parser.add_argument(
        "--arrival-process", choices=ARRIVAL_PROCESSES, default="poisson"
    )
    parser.add_argument(
        "--session-rate", type=float, default=0.2, help="programs per second"
    )
    parser.add_argument("--coefficient-of-variation", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=11)
    arguments = parser.parse_args()

    machine = [json.loads(path.read_text()) for path in arguments.engine_capacity]
    maximum_context_tokens = min(engine["max_model_len"] for engine in machine)
    kv_capacity_tokens = sum(
        engine["kv_capacity_tokens_device"] + engine["kv_capacity_tokens_host"]
        for engine in machine
    )

    all_sessions = {
        session_id: scale_tool_wait(
            scale_context(rounds, arguments.context_scale), arguments.tool_wait_scale
        )
        for session_id, rounds in load_sessions(SOURCE_CSV).items()
    }
    eligible = sorted(
        session_id
        for session_id, rounds in all_sessions.items()
        if largest_context(rounds) <= maximum_context_tokens
        and len(rounds) >= arguments.minimum_rounds
        and (
            arguments.maximum_tool_wait_seconds is None
            or sum(r["tool_wait_after_ms"] for r in rounds) / 1000
            <= arguments.maximum_tool_wait_seconds
        )
    )
    chosen = random.Random(f"{arguments.seed}/sessions").sample(
        eligible, min(arguments.sessions, len(eligible))
    )
    arrivals = arrival_times_seconds(
        random.Random(f"{arguments.seed}/arrival"),
        len(chosen),
        arguments.arrival_process,
        arguments.session_rate,
        arguments.coefficient_of_variation,
    )

    rows = []
    for session_id, arrival in zip(chosen, arrivals, strict=True):
        for round_ in all_sessions[session_id]:
            rows.append(
                {"session_id": session_id, "arrival_time": round(arrival * 1000)}
                | round_
            )
    directory = WORKLOADS_DIRECTORY / arguments.name
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "sessions.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    chosen_rounds = [all_sessions[session_id] for session_id in chosen]
    largest = [largest_context(rounds) for rounds in chosen_rounds]
    manifest = {
        "what_this_is": (
            "real coding-agent programs (TraceLab v0.0.2) with their context scaled "
            "to a machine, selected to fit it, with synthetic arrival times; token "
            "counts and tool waits otherwise as recorded, no text"
        ),
        "source": "https://github.com/uw-syfi/TraceLab releases v0.0.2 (CC BY 4.0), "
        "exported by its artifacts/trace_facts/csv_export/convert.py",
        "source_sha256": hashlib.sha256(SOURCE_CSV.read_bytes()).hexdigest(),
        "format": "TraceLab session_runner CSV",
        "generator": "experiments/workload/select_tracelab_sessions.py",
        **git_state(),
        "machine": machine,
        "parameters": {
            key: [str(v) for v in value] if isinstance(value, list) else value
            for key, value in vars(arguments).items()
            if key != "name"
        },
        "selection": {
            "programs_in_source": len(all_sessions),
            "programs_that_fit_after_scaling": len(eligible),
            "programs_chosen": len(chosen),
            "bias": "only programs whose every scaled round fits max_model_len"
            + (
                ""
                if arguments.maximum_tool_wait_seconds is None
                else " and whose total tool wait is at most "
                f"{arguments.maximum_tool_wait_seconds:g} s"
            ),
        },
        "against_the_machine": {
            "max_model_len": maximum_context_tokens,
            "kv_capacity_tokens": kv_capacity_tokens,
            "largest_context_per_program": describe(largest),
            "sum_of_largest_contexts": sum(largest),
            "sum_of_largest_contexts_over_kv_capacity": round(
                sum(largest) / kv_capacity_tokens, 2
            ),
        },
        "summary": {
            "rounds_per_program": describe([len(r) for r in chosen_rounds]),
            "context_tokens_per_round": describe(
                [
                    r["prefix_len"] + r["input_len"]
                    for rounds in chosen_rounds
                    for r in rounds
                ]
            ),
            "output_tokens_per_round": describe(
                [r["output_len"] for rounds in chosen_rounds for r in rounds]
            ),
            "tool_wait_seconds_per_program": describe(
                [
                    sum(r["tool_wait_after_ms"] for r in rounds) / 1000
                    for rounds in chosen_rounds
                ]
            ),
            "last_program_arrives_seconds": round(arrivals[-1], 1),
        },
        "output": "sessions.csv",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {len(chosen)} programs, {len(rows)} rounds to {directory}")
    print(json.dumps(manifest["against_the_machine"], indent=1))


if __name__ == "__main__":
    main()
