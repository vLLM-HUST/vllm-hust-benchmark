#!/usr/bin/env python3
"""Build an agent-program workload with every knob under our control. It only
writes data; it sends nothing.

    python experiments/workload/build_program_workload.py \\
        --name swe_smith-context_0.5-output_0.5-tool_lognormal_3s \\
        --context-scale 0.5 --tool-time lognormal:mean=3,cv=1 \\
        --maximum-context-tokens 32768 --programs 600

What a real trajectory contributes is only the SHAPE of each program: how many
rounds it has and how its context grows round by round. The source is the
SWE-smith trajectories (SWE-agent on SWE-bench-style tasks, MIT;
data/workloads/swe_smith_trajectories/programs.jsonl, prepared by
scripts/prepare_agent_trajectories.py). Each program there is
`system, turns = [{user, assistant}, ...]`; round k sends the system prompt,
every earlier turn (user and assistant) and turn k's user message, and the
model answers with turn k's assistant message. Mapped to the session format
replay_sessions.py reads (TraceLab's session_runner columns):

    prefix_len  = tokens of everything before turn k's user message
                  (round 0: 0; the system prompt counts as round 0's new input)
    input_len   = tokens of turn k's user message (+ the system prompt in round 0)
    output_len  = tokens of turn k's assistant message

Everything else is a knob, recorded in the manifest:

    --context-scale     multiplies prefix_len and input_len
    --output-scale      multiplies output_len (default: the same as
                        --context-scale, which keeps the ratio of input work to
                        output work of the real trajectory)
    --tool-time         seconds between two rounds of a program:
                          none                      0 s
                          constant:3                every tool call 3 s
                          lognormal:mean=3,cv=1     log-normal, mean 3 s, coefficient
                                                    of variation 1 (independent draw
                                                    per round, seeded per program)
    --maximum-context-tokens   the engine's max_model_len: programs whose scaled
                        round (prefix + input + output) would not fit are dropped,
                        and how many is recorded
    --minimum-rounds, --programs, --seed
    --program-rate      mean number of newly arriving programs per second
    --arrival-process   constant, Poisson, or Gamma inter-arrival times
    --arrival-coefficient-of-variation
                        burstiness of Gamma arrivals; 1 is Poisson-like
    --arrival-duration-seconds
                        stop admitting programs after this time; the replay can
                        then drain every admitted program to completion
    --urgent-fraction   fraction labelled urgent, with a separate latency goal

⚠️ Token counts come from characters / 3.6 (code-heavy English; the same
constant as the earlier agent benchmark). The replay sends token ids, so the
engine receives exactly these counts; what the constant decides is only how
long a round of the ORIGINAL trajectory is taken to be.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics

from arrival_process import ARRIVAL_PROCESSES, arrival_times_seconds
from generate_workload import WORKLOADS_DIRECTORY, git_state

SOURCE = WORKLOADS_DIRECTORY / "swe_smith_trajectories" / "programs.jsonl"
CHARACTERS_PER_TOKEN = 3.6
COLUMNS = (
    "session_id",
    "arrival_time",
    "round_idx",
    "prefix_len",
    "input_len",
    "output_len",
    "tool_wait_after_ms",
    "priority_class",
    "latency_objective_seconds",
)


def tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / CHARACTERS_PER_TOKEN))


def program_rounds(program: dict) -> list[dict]:
    rounds, context = [], 0
    for index, turn in enumerate(program["turns"]):
        new = tokens(turn["user"]) + (tokens(program["system"]) if index == 0 else 0)
        output = tokens(turn["assistant"])
        rounds.append(
            {
                "round_idx": index,
                "prefix_len": context,
                "input_len": new,
                "output_len": output,
            }
        )
        context += new + output
    return rounds


def tool_time_sampler(specification: str):
    """'none' | 'constant:<s>' | 'lognormal:mean=<s>,cv=<c>' -> f(random) -> s."""
    kind, _, rest = specification.partition(":")
    if kind == "none":
        return lambda generator: 0.0
    if kind == "constant":
        seconds = float(rest)
        return lambda generator: seconds
    if kind == "lognormal":
        fields = dict(item.split("=") for item in rest.split(","))
        mean, cv = float(fields["mean"]), float(fields["cv"])
        sigma = math.sqrt(math.log(1 + cv * cv))
        mu = math.log(mean) - sigma * sigma / 2
        return lambda generator: generator.lognormvariate(mu, sigma)
    raise SystemExit(f"unknown --tool-time {specification!r}")


def describe(values) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {}
    n = len(ordered)
    return {
        "mean": round(statistics.fmean(ordered), 1),
        "median": ordered[n // 2],
        "p90": ordered[int(0.9 * (n - 1))],
        "maximum": ordered[-1],
    }


def program_schedule(
    count: int,
    seed: int,
    program_rate: float | None,
    arrival_process: str,
    arrival_coefficient_of_variation: float,
    arrival_duration_seconds: float | None,
) -> list[float]:
    """Return external program arrival times in seconds.

    The legacy mode (no program rate) preserves the old one-millisecond spacing
    for existing experiment commands. A streaming experiment must set a rate.
    """
    if program_rate is None:
        if arrival_duration_seconds is not None:
            raise ValueError("--arrival-duration-seconds requires --program-rate")
        return [index / 1000 for index in range(count)]
    if program_rate <= 0:
        raise ValueError("--program-rate must be positive")
    arrivals = arrival_times_seconds(
        random.Random(f"{seed}/arrival"),
        count,
        arrival_process,
        program_rate,
        arrival_coefficient_of_variation,
    )
    if arrival_duration_seconds is not None:
        arrivals = [time for time in arrivals if time <= arrival_duration_seconds]
    return arrivals


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--context-scale", type=float, default=1.0)
    parser.add_argument("--output-scale", type=float)
    parser.add_argument("--tool-time", default="none")
    parser.add_argument("--maximum-context-tokens", type=int, default=32768)
    parser.add_argument("--minimum-rounds", type=int, default=4)
    parser.add_argument("--programs", type=int, default=600)
    parser.add_argument(
        "--program-rate",
        type=float,
        help="mean number of external program arrivals per second",
    )
    parser.add_argument(
        "--arrival-process", choices=ARRIVAL_PROCESSES, default="poisson"
    )
    parser.add_argument(
        "--arrival-coefficient-of-variation", type=float, default=1.0
    )
    parser.add_argument(
        "--arrival-duration-seconds",
        type=float,
        help="keep only arrivals at or before this time, then drain them",
    )
    parser.add_argument("--urgent-fraction", type=float, default=0.0)
    parser.add_argument("--urgent-latency-objective-seconds", type=float, default=120)
    parser.add_argument(
        "--ordinary-latency-objective-seconds", type=float, default=600
    )
    parser.add_argument("--seed", type=int, default=11)
    arguments = parser.parse_args()
    if not 0 <= arguments.urgent_fraction <= 1:
        parser.error("--urgent-fraction must be between 0 and 1")
    output_scale = (
        arguments.context_scale
        if arguments.output_scale is None
        else arguments.output_scale
    )
    sample_tool_time = tool_time_sampler(arguments.tool_time)

    source_programs = [json.loads(line) for line in SOURCE.open()]
    kept, dropped_long, dropped_short = [], 0, 0
    for program in source_programs:
        rounds = program_rounds(program)
        scaled = [
            r
            | {
                "prefix_len": round(r["prefix_len"] * arguments.context_scale),
                "input_len": max(1, round(r["input_len"] * arguments.context_scale)),
                "output_len": max(1, round(r["output_len"] * output_scale)),
            }
            for r in rounds
        ]
        if len(scaled) < arguments.minimum_rounds:
            dropped_short += 1
        elif (
            max(r["prefix_len"] + r["input_len"] + r["output_len"] for r in scaled)
            > arguments.maximum_context_tokens
        ):
            dropped_long += 1
        else:
            kept.append((program["program_id"], scaled))
    chosen = random.Random(f"{arguments.seed}/programs").sample(
        kept, min(arguments.programs, len(kept))
    )
    try:
        arrivals = program_schedule(
            len(chosen),
            arguments.seed,
            arguments.program_rate,
            arguments.arrival_process,
            arguments.arrival_coefficient_of_variation,
            arguments.arrival_duration_seconds,
        )
    except ValueError as error:
        parser.error(str(error))
    chosen = chosen[: len(arrivals)]
    urgent_count = round(len(chosen) * arguments.urgent_fraction)
    urgent_programs = set(
        random.Random(f"{arguments.seed}/priority").sample(
            [program_id for program_id, _ in chosen], urgent_count
        )
    )
    rows = []
    for arrival_seconds, (program_id, rounds) in zip(arrivals, chosen, strict=True):
        generator = random.Random(f"{arguments.seed}/tool/{program_id}")
        priority_class = "urgent" if program_id in urgent_programs else "ordinary"
        latency_objective_seconds = (
            arguments.urgent_latency_objective_seconds
            if priority_class == "urgent"
            else arguments.ordinary_latency_objective_seconds
        )
        for index, r in enumerate(rounds):
            last = index == len(rounds) - 1
            wait = 0.0 if last else sample_tool_time(generator)
            rows.append(
                {
                    "session_id": program_id,
                    "arrival_time": round(arrival_seconds * 1000),
                    **r,
                    "tool_wait_after_ms": round(wait * 1000),
                    "priority_class": priority_class,
                    "latency_objective_seconds": latency_objective_seconds,
                }
            )
    directory = WORKLOADS_DIRECTORY / arguments.name
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "sessions.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    largest = [
        max(r["prefix_len"] + r["input_len"] + r["output_len"] for r in rs)
        for _, rs in chosen
    ]
    kv_bytes_per_token = 196608  # Qwen2.5-14B bf16: 2 x 48 layers x 8 heads x 128 x 2
    manifest = {
        "what_this_is": "agent programs with the round structure of SWE-smith "
        "trajectories (SWE-agent), token counts scaled and tool time set by knobs",
        "source": "https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories "
        "(MIT), data/workloads/swe_smith_trajectories/programs.jsonl",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "generator": "experiments/workload/build_program_workload.py",
        **git_state(),
        "knobs": {
            "context_scale": arguments.context_scale,
            "output_scale": output_scale,
            "tool_time": arguments.tool_time,
            "maximum_context_tokens": arguments.maximum_context_tokens,
            "minimum_rounds": arguments.minimum_rounds,
            "programs": arguments.programs,
            "program_rate": arguments.program_rate,
            "arrival_process": arguments.arrival_process,
            "arrival_coefficient_of_variation": (
                arguments.arrival_coefficient_of_variation
            ),
            "arrival_duration_seconds": arguments.arrival_duration_seconds,
            "urgent_fraction": arguments.urgent_fraction,
            "urgent_latency_objective_seconds": (
                arguments.urgent_latency_objective_seconds
            ),
            "ordinary_latency_objective_seconds": (
                arguments.ordinary_latency_objective_seconds
            ),
            "seed": arguments.seed,
            "characters_per_token": CHARACTERS_PER_TOKEN,
        },
        "selection": {
            "programs_in_source": len(source_programs),
            "dropped_fewer_rounds": dropped_short,
            "dropped_too_long_after_scaling": dropped_long,
            "eligible": len(kept),
            "chosen": len(chosen),
            "urgent": len(urgent_programs),
            "ordinary": len(chosen) - len(urgent_programs),
        },
        "summary": {
            "last_program_arrival_seconds": (
                round(arrivals[-1], 3) if arrivals else None
            ),
            "rounds_per_program": describe([len(rs) for _, rs in chosen]),
            "largest_context_per_program_tokens": describe(largest),
            "largest_context_per_program_gib_qwen2.5_14b": {
                k: round(v * kv_bytes_per_token / 2**30, 2)
                for k, v in describe(largest).items()
            },
            "output_tokens_per_round": describe([r["output_len"] for r in rows]),
            "input_tokens_per_round_new": describe([r["input_len"] for r in rows]),
            "tool_seconds_per_round": describe(
                [
                    r["tool_wait_after_ms"] / 1000
                    for r in rows
                    if r["tool_wait_after_ms"]
                ]
            ),
        },
        "output": "sessions.csv",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(
        json.dumps(
            {k: manifest[k] for k in ("knobs", "selection", "summary")}, indent=1
        )
    )


if __name__ == "__main__":
    main()
