"""The numbers behind a closed-loop experiment, one function per question, each
returning rows (a list of dicts) that experiment_report.py writes as Markdown
and CSV. Nothing here draws.

An experiment directory holds <experiment>/concurrency_<C>/<configuration>/ with
rounds.jsonl, summary.json, engine_metrics_raw.jsonl.gz and (through
ThunderAgent) thunderagent.log, as written by concurrency_sweep.sh.

Definitions used throughout (the report repeats them):
    step            one round of a program: one request to the model and the tool
                    time after it (ThunderAgent paper §5.1: "one step includes a
                    reasoning and acting period")
    prompt of a step    the engine's usage.prompt_tokens for that request: the
                    whole context sent (history + new input)
    window          [start, end] seconds since the run started; a step belongs
                    to it when its last token arrived inside it
    standard step   the mean prompt of a step over every round of every program
                    of the workload (sessions.csv), i.e. what a step averages
                    when every program runs to its end
"""

from __future__ import annotations

import csv
import gzip
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
KV_BYTES_PER_TOKEN = 196_608  # Qwen2.5-14B bf16: 2 x 48 layers x 8 KV heads x 128 x 2 B
# measured 2026-09-30 on ascend86, idle single-card engine, 12,000-token context,
# nothing cached (see the Figure 1a reproduction report); an estimate, not a law
PREFILL_TOKENS_PER_SECOND = 6714
LONG_WAIT_SECONDS = 300


# ---------------------------------------------------------------- loading


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def value(samples, name: str) -> float:
    return sum(v for metric, _labels, v in samples if metric == name)


class Run:
    """One configuration at one concurrency."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.configuration = directory.name
        self.concurrency = int(re.search(r"concurrency_(\d+)", str(directory))[1])
        self.summary = json.loads((directory / "summary.json").read_text())
        self.rounds = [
            r for r in read_jsonl(directory / "rounds.jsonl") if r["status"] == "ok"
        ]
        self._engine = None

    @property
    def engine(self) -> list[dict]:
        if self._engine is None:
            self._engine = read_jsonl(self.directory / "engine_metrics_raw.jsonl.gz")
        return self._engine

    def in_window(self, start: float, end: float) -> list[dict]:
        return [r for r in self.rounds if start <= r["finished_seconds"] <= end]

    def engine_window(self, start: float, end: float) -> list[dict]:
        return [row for row in self.engine if start <= row["t"] <= end]

    def programs(self) -> dict[str, list[dict]]:
        by_program = defaultdict(list)
        for r in self.rounds:
            by_program[r["session_id"]].append(r)
        for items in by_program.values():
            items.sort(key=lambda r: r["round_idx"])
        return by_program

    def kv_capacity_tokens(self) -> int:
        return sum(e["kv_capacity_tokens_device"] for e in self.summary["engines"])

    def thunderagent_accounting(self, start: float, end: float) -> list[dict]:
        """ThunderAgent's own view every 5 s: programs it admitted, how many are
        reasoning / acting, how many it holds, tokens it counts as occupied."""
        log = self.directory / "thunderagent.log"
        if not log.exists():
            return []
        rows = []
        for line in log.read_text(errors="replace").splitlines():
            if not line.startswith("[thunderagent] scheduler before"):
                continue
            head, _, tail = line.partition("| engine:")
            fields = dict(re.findall(r"(\w+)=([\d.]+)", head))
            held = re.search(r"\| waiting=(\d+)", tail)
            if not held or "programs" not in fields:
                continue
            t = float(fields["t"]) - self.summary["started_at_wall"]
            if start <= t <= end:
                rows.append(
                    {
                        "admitted": float(fields["programs"]),
                        "reasoning": float(fields.get("reasoning", 0)),
                        "acting": float(fields.get("acting", 0)),
                        "held": float(held[1]),
                        "active_tokens": float(fields.get("active_tokens", 0)),
                    }
                )
        return rows


def load(experiment: Path) -> dict[int, dict[str, Run]]:
    """{concurrency: {configuration: Run}}, complete runs only."""
    points: dict[int, dict[str, Run]] = defaultdict(dict)
    for summary in sorted(experiment.glob("concurrency_*/*/summary.json")):
        run = Run(summary.parent)
        points[run.concurrency][run.configuration] = run
    return dict(sorted(points.items()))


def workload_rows(run: Run) -> list[dict]:
    name = Path(run.summary["parameters"]["workload"]).name
    with (REPOSITORY / "data" / "workloads" / name / "sessions.csv").open() as handle:
        return list(csv.DictReader(handle))


def standard_step_tokens(run: Run) -> float:
    template = run.summary.get("chat_template_tokens", 0)
    return statistics.fmean(
        int(r["prefix_len"]) + int(r["input_len"]) + template
        for r in workload_rows(run)
    )


def default_window(run: Run) -> tuple[float, float]:
    """The last two thirds of the run: the first third is the start-up burst."""
    duration = float(run.summary["parameters"]["duration_seconds"])
    return duration / 3, duration


def mean(values) -> float:
    values = list(values)
    return statistics.fmean(values) if values else float("nan")


# ---------------------------------------------------------------- tables


def throughput(points, start, end) -> list[dict]:
    """Every throughput measure side by side. Steps per minute counts every step
    as one; the token measures weight a step by its size; the standard-step
    measure converts prompt tokens back into steps of the workload's mean size."""
    rows = []
    for concurrency, runs in points.items():
        for name, run in runs.items():
            inside = run.in_window(start, end)
            minutes = (end - start) / 60
            prompt = sum(r["engine_prompt_tokens"] for r in inside)
            output = sum(r["engine_completion_tokens"] for r in inside)
            rows.append(
                {
                    "concurrency": concurrency,
                    "configuration": name,
                    "steps per minute": len(inside) / minutes,
                    "prompt tokens of those steps per minute": prompt / minutes,
                    "output tokens per second": output / (end - start),
                    "mean prompt per step": prompt / len(inside) if inside else 0,
                    "mean round index of those steps": mean(
                        r["round_idx"] for r in inside
                    ),
                    "steps per minute, standard-size steps": prompt
                    / minutes
                    / standard_step_tokens(run),
                }
            )
    return rows


def steps_over_time(points, bin_seconds: float = 200) -> list[dict]:
    """Steps per minute in consecutive bins: shows whether the run reached a
    steady state or was still eating the cheap early rounds."""
    rows = []
    for concurrency, runs in points.items():
        for name, run in runs.items():
            end = max(r["finished_seconds"] for r in run.rounds)
            start = 0.0
            while start < end:
                inside = [
                    r
                    for r in run.rounds
                    if start <= r["finished_seconds"] < start + bin_seconds
                ]
                if inside:
                    rows.append(
                        {
                            "concurrency": concurrency,
                            "configuration": name,
                            "seconds": f"{start:.0f}-{start + bin_seconds:.0f}",
                            "steps per minute": len(inside) / (bin_seconds / 60),
                            "mean round index": mean(r["round_idx"] for r in inside),
                            "mean prompt per step": mean(
                                r["engine_prompt_tokens"] for r in inside
                            ),
                            "prompt tokens per minute": sum(
                                r["engine_prompt_tokens"] for r in inside
                            )
                            / (bin_seconds / 60),
                        }
                    )
                start += bin_seconds
    return rows


def kv_fit_and_recompute(points, start, end, prefill=PREFILL_TOKENS_PER_SECOND):
    """Does the programs' context fit in the KV cache, and what does not fitting
    cost. Avoidable recompute = prompt tokens not served from cache minus the new
    input every round must prefill anyway."""
    rows = []
    for concurrency, runs in points.items():
        for name, run in runs.items():
            window = run.engine_window(start, end)
            seconds = window[-1]["t"] - window[0]["t"]
            prompt = value(window[-1]["samples"], "vllm:prompt_tokens_total") - value(
                window[0]["samples"], "vllm:prompt_tokens_total"
            )
            cached = value(
                window[-1]["samples"], "vllm:prompt_tokens_cached_total"
            ) - value(window[0]["samples"], "vllm:prompt_tokens_cached_total")
            preemptions = value(
                window[-1]["samples"], "vllm:num_preemptions_total"
            ) - value(window[0]["samples"], "vllm:num_preemptions_total")
            template = run.summary.get("chat_template_tokens", 0)
            new = sum(
                r["input_len"] + template
                for r in run.rounds
                if start <= r["sent_seconds"] <= end
            )
            avoidable = max(0.0, prompt - cached - new) / seconds
            inside = run.in_window(start, end)
            context = mean(
                r["engine_prompt_tokens"] + r["engine_completion_tokens"]
                for r in inside
            )
            gib = KV_BYTES_PER_TOKEN / 2**30
            rows.append(
                {
                    "concurrency": concurrency,
                    "configuration": name,
                    "mean context per step": context,
                    "concurrency x context / KV capacity": concurrency
                    * context
                    / run.kv_capacity_tokens(),
                    "from cache GiB per minute": cached * gib / seconds * 60,
                    "not from cache GiB per minute": (prompt - cached)
                    * gib
                    / seconds
                    * 60,
                    "avoidable recompute tokens per second": avoidable,
                    "avoidable recompute card seconds per minute (estimate)": avoidable
                    / prefill
                    * 60,
                    "preemptions": preemptions,
                    "engine requests running (mean)": mean(
                        value(row["samples"], "vllm:num_requests_running")
                        for row in window
                    ),
                    "engine requests waiting (mean)": mean(
                        value(row["samples"], "vllm:num_requests_waiting")
                        for row in window
                    ),
                    "engine seconds with nothing running": sum(
                        1
                        for row in window
                        if value(row["samples"], "vllm:num_requests_running") == 0
                    ),
                }
            )
    return rows


def littles_law(points, start, end) -> list[dict]:
    """Closed loop: steps per minute = programs / time per step, where a step's
    time is its tool time plus its time in the engine (including any hold)."""
    rows = []
    for concurrency, runs in points.items():
        for name, run in runs.items():
            sent = [
                r
                for r in run.rounds
                if start <= r["sent_seconds"] <= end and r["tool_wait_after_ms"] > 0
            ]
            tool = mean(r["tool_wait_after_ms"] / 1000 for r in sent)
            engine = mean(r["finished_seconds"] - r["sent_seconds"] for r in sent)
            measured = len(run.in_window(start, end)) / ((end - start) / 60)
            rows.append(
                {
                    "concurrency": concurrency,
                    "configuration": name,
                    "tool seconds per step": tool,
                    "engine and held seconds per step": engine,
                    "predicted steps per minute": concurrency * 60 / (tool + engine),
                    "measured steps per minute": measured,
                }
            )
    return rows


def arrivals_and_fairness(points) -> list[dict]:
    """How programs arrived (burst at t=0, then one per completion) and how the
    first batch fared against later arrivals; long waits per program."""
    rows = []
    for concurrency, runs in points.items():
        for name, run in runs.items():
            programs = []
            for items in run.programs().values():
                waits = [r["first_token_seconds"] - r["sent_seconds"] for r in items]
                programs.append(
                    {
                        "arrival": items[0]["sent_seconds"],
                        "finished": items[-1]["tool_wait_after_ms"] == 0,
                        "waiting": sum(waits),
                        "longest": max(waits),
                    }
                )
            first = [p for p in programs if p["arrival"] < 5]
            later = [p for p in programs if p["arrival"] >= 5]

            def share(group):
                return sum(p["finished"] for p in group) / len(group) if group else 0

            rows.append(
                {
                    "concurrency": concurrency,
                    "configuration": name,
                    "arrived at t=0": len(first),
                    "arrived later": len(later),
                    "finished": sum(p["finished"] for p in programs),
                    "first batch finished share": share(first),
                    "first batch median waiting s": mean([])
                    if not first
                    else statistics.median(p["waiting"] for p in first),
                    "later arrivals finished share": share(later),
                    "later arrivals median waiting s": mean([])
                    if not later
                    else statistics.median(p["waiting"] for p in later),
                    f"programs with a wait > {LONG_WAIT_SECONDS} s": sum(
                        p["longest"] > LONG_WAIT_SECONDS for p in programs
                    ),
                    "longest single wait s": max(p["longest"] for p in programs),
                }
            )
    return rows


def thunderagent_admission(points, start, end) -> list[dict]:
    """What ThunderAgent lets in versus what the engine holds, and which program
    sizes the completed steps come from (it places small programs first)."""
    rows = []
    for concurrency, runs in points.items():
        for name, run in runs.items():
            accounting = run.thunderagent_accounting(start, end)
            if not accounting:
                continue
            largest = defaultdict(int)
            for r in workload_rows(run):
                size = int(r["prefix_len"]) + int(r["input_len"]) + int(r["output_len"])
                largest[r["session_id"]] = max(largest[r["session_id"]], size)
            inside = run.in_window(start, end)
            sizes = Counter(
                "small"
                if largest[r["session_id"]] < 8000
                else "medium"
                if largest[r["session_id"]] < 16000
                else "large"
                for r in inside
            )
            window = run.engine_window(start, end)
            capacity = run.kv_capacity_tokens()
            rows.append(
                {
                    "concurrency": concurrency,
                    "configuration": name,
                    "admitted programs": mean(a["admitted"] for a in accounting),
                    "reasoning": mean(a["reasoning"] for a in accounting),
                    "acting": mean(a["acting"] for a in accounting),
                    "held": mean(a["held"] for a in accounting),
                    "ThunderAgent counts as occupied (tokens)": mean(
                        a["active_tokens"] for a in accounting
                    ),
                    "engine KV held by running requests (tokens)": mean(
                        value(row["samples"], "vllm:kv_cache_usage_perc") * capacity
                        for row in window
                    ),
                    "engine requests running": mean(
                        value(row["samples"], "vllm:num_requests_running")
                        for row in window
                    ),
                    "steps from small programs (<8k)": sizes["small"] / len(inside),
                    "steps from large programs (>16k)": sizes["large"] / len(inside),
                }
            )
    return rows


# ---------------------------------------------------------------- writing


def to_markdown(rows: list[dict]) -> str:
    if not rows:
        return "(none)\n"
    columns = list(rows[0])

    def cell(v):
        if isinstance(v, float):
            if v != v:
                return "-"
            if 0 < abs(v) < 1 and not float(v).is_integer():
                return f"{v:.2f}"
            return f"{v:,.1f}" if abs(v) < 1000 else f"{v:,.0f}"
        return str(v)

    lines = ["| " + " | ".join(columns) + " |", "|" + " --- |" * len(columns)]
    lines += ["| " + " | ".join(cell(row[c]) for c in columns) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def to_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
