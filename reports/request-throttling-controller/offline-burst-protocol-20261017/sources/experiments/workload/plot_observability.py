#!/usr/bin/env python3
"""Ground truth over time, every source, every engine: the figures for one
configuration run of replay_sessions.py (closed loop, through ThunderAgent).

    python experiments/workload/plot_observability.py <run directory>

Nothing here is a summary: every panel is a recorded quantity against time
(seconds since the replay started; logs are aligned by wall clock using
`started_at_wall` in summary.json). Engine-level figures have one column per
engine. Units are on every axis. Where a panel shows a statistic of many events
in a time window (a median, a maximum), the window is on the axis label.

Sources (all in the run directory):
    engine_metrics_raw.jsonl.gz   every vllm:* sample of every engine, 1 s
    device.jsonl.gz               npu-smi per card, 5 s
    rounds.jsonl                  client: each LLM round (send, first token, end)
    tokens.jsonl                  client: arrival time of every streamed chunk
    thunderagent.log              every ThunderAgent decision; its per-backend
                                  accounting before and after every 5 s tick
    thunderagent-profile/step_profiles.csv   ThunderAgent's own per-step timing

Figures written to <run>/figures/:
    A1_engine_state.png        KV in use (tokens) vs capacity and vs ThunderAgent's
                               own accounting; requests running / waiting by
                               reason; preemptions; batch tokens per engine step
    A2_engine_throughput.png   prefill, cached-prompt and decode tokens/s;
                               vLLM's estimated FLOP/s and HBM read/write bytes/s
    A3_engine_latency.png      from vLLM's histograms, per 10 s: queue, prefill,
                               decode, time to first token, inter-token latency,
                               end-to-end -- mean and 90th percentile
    A4_device.png              npu-smi: AI Core busy, HBM bandwidth, NPU
                               utilisation (as npu-smi reports them, percent of
                               the card), engine process memory (MB)
    B1_client_latency.png      client: time to first token, gap between streamed
                               tokens, per-token latency (end-to-end / tokens,
                               Llumnix's), time held by ThunderAgent
    C1_slots_timeline.png      one row per closed-loop slot: every program it ran,
                               its phases, and when ThunderAgent paused or queued it
    C2_program_states.png      programs per ThunderAgent state per backend
    C3_context_growth.png      prompt tokens of every round against time
    D1_decisions.png           every ThunderAgent decision as a tick, per backend;
                               decisions per 10 s by type
    D2_pause_cost.png          ThunderAgent's own pause_s per step; hold time
"""

from __future__ import annotations

import csv
import datetime
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from plot_run import AQUA, BLUE, INK, ORANGE, YELLOW, style

GREY = "#9a9994"
VIOLET = "#4a3aa7"
RED = "#e34948"
WINDOW = 10.0

# ---------------------------------------------------------------- reading


def read_jsonl(path: Path, compressed: bool = False) -> list[dict]:
    if not path.exists():
        return []
    opener = gzip.open if compressed else open
    with opener(path, "rt") as source:
        return [json.loads(line) for line in source if line.strip()]


def engine_series(raw: list[dict]) -> dict[str, list[dict]]:
    """engine -> time-ordered samples: {"t", scalars{name: value}, labelled,
    histograms{name: {"buckets": {le: count}, "sum", "count"}}}."""
    engines: dict[str, list[dict]] = defaultdict(list)
    for line in raw:
        scalars: dict[str, float] = defaultdict(float)
        labelled: dict[tuple[str, str], float] = {}
        histograms: dict[str, dict] = defaultdict(
            lambda: {"buckets": defaultdict(float), "sum": 0.0, "count": 0.0}
        )
        for name, labels, value in line["samples"]:
            if name.endswith("_bucket"):
                histograms[name[: -len("_bucket")]]["buckets"][labels.get("le")] += (
                    value
                )
            if name.endswith("_sum"):
                histograms[name[:-4]]["sum"] += value
            elif name.endswith("_count"):
                histograms[name[:-6]]["count"] += value
            elif not name.endswith("_bucket"):
                scalars[name] += value
                for key in ("reason", "source", "finished_reason"):
                    if key in labels:
                        labelled[(name, labels[key])] = (
                            labelled.get((name, labels[key]), 0.0) + value
                        )
        engines[line["engine"]].append(
            {
                "t": line["t"],
                "scalars": scalars,
                "labelled": labelled,
                "histograms": histograms,
            }
        )
    return dict(engines)


def windows(samples: list[dict], seconds: float = WINDOW):
    earliest = 0
    for later in samples:
        while samples[earliest + 1]["t"] <= later["t"] - seconds:
            earliest += 1
        if later["t"] > samples[earliest]["t"]:
            yield samples[earliest], later


def counter_rate(samples, name, seconds=WINDOW):
    times, values = [], []
    for a, b in windows(samples, seconds):
        times.append(b["t"])
        values.append((b["scalars"][name] - a["scalars"][name]) / (b["t"] - a["t"]))
    return times, values


def histogram_window(samples, name, seconds=WINDOW):
    """Per window: (time, mean, 90th percentile, events) of a vLLM histogram,
    from bucket increments. The 90th percentile is the upper edge of the bucket
    where the cumulative count crosses 90% (histograms keep no finer detail)."""
    out = []
    for a, b in windows(samples, seconds):
        ha, hb = a["histograms"].get(name), b["histograms"].get(name)
        if not ha or not hb:
            continue
        count = hb["count"] - ha["count"]
        if count <= 0:
            continue
        mean = (hb["sum"] - ha["sum"]) / count
        edges = sorted(
            (float("inf") if le == "+Inf" else float(le), le) for le in hb["buckets"]
        )
        p90 = None
        for edge, le in edges:
            if hb["buckets"][le] - ha["buckets"].get(le, 0.0) >= 0.9 * count:
                p90 = edge
                break
        out.append((b["t"], mean, p90, count))
    return out


def log_events(log: Path, started_at_wall: float) -> dict:
    """Decisions and per-backend accounting from a ThunderAgent log."""
    decisions: list[tuple[float, str, str, str]] = []  # t, kind, program, backend
    state: dict[str, list[dict]] = defaultdict(list)
    holds: list[tuple[float, float, str]] = []  # t, waited_s, outcome
    if not log.exists():
        return {"decisions": decisions, "state": state, "holds": holds}
    stamp = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})")
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith("[thunderagent] scheduler before"):
            fields = dict(re.findall(r"(\w+)=([\w.]+)", line))
            state[fields["backend"]].append(
                {
                    "t": float(fields["t"]) - started_at_wall,
                    **{
                        k: float(fields[k])
                        for k in (
                            "capacity",
                            "active_tokens",
                            "remaining",
                            "programs",
                            "reasoning",
                            "acting",
                            "waiting",
                        )
                        if k in fields
                    },
                }
            )
            continue
        if line.startswith("[thunderagent] request"):
            t = float(re.search(r"t=([\d.]+)", line)[1]) - started_at_wall
            waited = float(re.search(r"waited_s=([\d.]+)", line)[1])
            outcome = (
                "paused"
                if "was paused" in line
                else "queued"
                if "queued" in line
                else "new"
                if "new, assigned" in line
                else "proceeded"
            )
            holds.append((t, waited, outcome))
            continue
        match = stamp.match(line)
        if not match:
            continue
        wall = datetime.datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S,%f").timestamp()
        t = wall - started_at_wall
        for pattern, kind in (
            (r"Paused program (\S+) from \S+:(\d+)", "pause"),
            (r"Marked program (\S+) for pause.*decision_backend=\S+:(\d+)", "mark"),
            (r"Resumed program (\S+) to \S+:(\d+)", "resume"),
            (r"Queued new program (\S+)()", "queue"),
            (r"Assigned new program (\S+) to \S+:(\d+)", "assign"),
        ):
            found = re.search(pattern, line)
            if found:
                decisions.append((t, kind, found[1], found[2] or ""))
                break
    return {"decisions": decisions, "state": state, "holds": holds}


# ---------------------------------------------------------------- drawing

FLOOR_SECONDS = 1e-3  # log axes start at 1 ms; a 0 s hold is drawn at the floor


def with_gaps(times, values, max_gap):
    """Break a line where two points are further apart than `max_gap`, so a
    stretch with no events is not drawn as a line between its ends."""
    out_t, out_v = [], []
    for index, (t, v) in enumerate(zip(times, values, strict=True)):
        if index and t - times[index - 1] > max_gap:
            out_t.append(None)
            out_v.append(None)
        out_t.append(t)
        out_v.append(v)
    return out_t, out_v


def seconds_axis(axis, label: str) -> None:
    """Log-scale seconds, floor 1 ms, plain-number ticks; decades only when the
    data spans more than three of them (else 1-2-5 steps)."""
    ticker = __import__("matplotlib").ticker
    axis.set_yscale("log")
    style(axis, label)
    low, high = axis.get_ylim()
    low = max(low, FLOOR_SECONDS)
    axis.set_ylim(low, max(high, low * 10))
    subs = (1.0,) if high / low > 1000 else (1.0, 2.0, 5.0)
    axis.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=subs))
    axis.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.yaxis.set_minor_formatter(ticker.NullFormatter())


def engine_grid(pyplot, rows: int, engines: list[str], height: float = 2.3):
    figure, axes = pyplot.subplots(
        rows,
        len(engines),
        figsize=(4.4 * len(engines), height * rows),
        sharex=True,
        squeeze=False,
    )
    for column, engine in enumerate(engines):
        axes[0][column].set_title(
            engine.replace("http://", "").removesuffix("/v1"), color=INK
        )
    return figure, axes


def finish(figure, axes, out: Path, title: str) -> None:
    for axis in (
        axes[-1]
        if isinstance(axes[-1], (list, tuple)) or hasattr(axes[-1], "__len__")
        else [axes[-1]]
    ):
        axis.set_xlabel("seconds since the run started", color=INK)
    figure.suptitle(title, color=INK)
    figure.tight_layout()
    figure.savefig(out, dpi=110)
    __import__("matplotlib.pyplot").pyplot.close(figure)


def port(engine: str) -> str:
    return engine.rstrip("/").removesuffix("/v1").rsplit(":", 1)[-1]


def plot_engine_state(pyplot, series, capacity, accounting, out, title):
    engines = sorted(series)
    figure, axes = engine_grid(pyplot, 5, engines)
    for c, engine in enumerate(engines):
        s = series[engine]
        t = [x["t"] for x in s]
        cap = capacity.get(engine, 0)
        axes[0][c].plot(
            t,
            [cap * x["scalars"]["vllm:kv_cache_usage_perc"] for x in s],
            color=BLUE,
            linewidth=1.5,
            label="vLLM: KV in use",
        )
        own = accounting.get(port(engine), [])
        if own:
            axes[0][c].plot(
                [x["t"] for x in own],
                [x["active_tokens"] for x in own],
                color=ORANGE,
                linewidth=1.5,
                label="ThunderAgent: active program tokens",
            )
        axes[0][c].axhline(cap, color=INK, linestyle="--", linewidth=1)
        axes[0][c].text(
            0, cap, f" capacity {cap:,}", color=INK, va="bottom", fontsize=8
        )
        style(axes[0][c], "tokens" if c == 0 else "")
        axes[1][c].plot(
            t,
            [x["scalars"]["vllm:num_requests_running"] for x in s],
            color=BLUE,
            label="running",
        )
        for reason, colour in (("capacity", ORANGE), ("deferred", VIOLET)):
            axes[1][c].plot(
                t,
                [
                    x["labelled"].get(
                        ("vllm:num_requests_waiting_by_reason", reason), 0
                    )
                    for x in s
                ],
                color=colour,
                label=f"waiting ({reason})",
            )
        style(axes[1][c], "requests" if c == 0 else "")
        axes[2][c].step(
            t,
            [
                x["scalars"]["vllm:num_preemptions_total"]
                - s[0]["scalars"]["vllm:num_preemptions_total"]
                for x in s
            ],
            color=RED,
            where="post",
        )
        style(axes[2][c], "preemptions\nso far" if c == 0 else "")
        window = histogram_window(s, "vllm:iteration_tokens_total")
        times = [w[0] for w in window]
        axes[3][c].plot(
            *with_gaps(times, [w[1] for w in window], 2 * WINDOW), color=BLUE
        )
        style(
            axes[3][c],
            f"tokens per engine\nstep, mean ({WINDOW:.0f} s,\nlog)" if c == 0 else "",
        )
        axes[3][c].set_yscale("log")  # after style(), which sets a linear locator
        ticker = __import__("matplotlib").ticker
        axes[3][c].yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0,)))
        axes[3][c].yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
        )
        axes[3][c].yaxis.set_minor_formatter(ticker.NullFormatter())
        axes[4][c].plot(
            *with_gaps(times, [w[3] / WINDOW for w in window], 2 * WINDOW), color=AQUA
        )
        style(
            axes[4][c],
            f"engine steps\nper second\n({WINDOW:.0f} s window)" if c == 0 else "",
        )
    for row in axes[:2]:
        row[-1].legend(frameon=False, fontsize=7, loc="lower right")
    finish(figure, axes, out, title + "  --  A1 engine state")


def plot_engine_throughput(pyplot, series, out, title):
    engines = sorted(series)
    rows = (
        ("vllm:prompt_tokens_total", "prefill: prompt\ntokens/s", 1.0, BLUE),
        ("vllm:prompt_tokens_cached_total", "prompt tokens/s\nfrom cache", 1.0, AQUA),
        ("vllm:generation_tokens_total", "decode: output\ntokens/s", 1.0, BLUE),
        ("vllm:estimated_flops_per_gpu_total", "estimated\nTFLOP/s", 1e-12, ORANGE),
        (
            "vllm:estimated_read_bytes_per_gpu_total",
            "estimated HBM\nread GB/s",
            1e-9,
            VIOLET,
        ),
        (
            "vllm:estimated_write_bytes_per_gpu_total",
            "estimated HBM\nwrite GB/s",
            1e-9,
            VIOLET,
        ),
    )
    figure, axes = engine_grid(pyplot, len(rows), engines)
    for c, engine in enumerate(engines):
        for r, (name, label, scale, colour) in enumerate(rows):
            times, values = counter_rate(series[engine], name)
            axes[r][c].plot(times, [v * scale for v in values], color=colour)
            style(axes[r][c], f"{label}\n({WINDOW:.0f} s average)" if c == 0 else "")
    finish(figure, axes, out, title + "  --  A2 engine throughput")


def plot_engine_latency(pyplot, series, out, title):
    engines = sorted(series)
    rows = (
        ("vllm:request_queue_time_seconds", "queue time"),
        ("vllm:request_prefill_time_seconds", "prefill time"),
        ("vllm:request_decode_time_seconds", "decode time"),
        ("vllm:time_to_first_token_seconds", "time to\nfirst token"),
        ("vllm:inter_token_latency_seconds", "inter-token\nlatency"),
        ("vllm:e2e_request_latency_seconds", "end-to-end"),
    )
    figure, axes = engine_grid(pyplot, len(rows), engines)
    for c, engine in enumerate(engines):
        for r, (name, label) in enumerate(rows):
            window = histogram_window(series[engine], name)
            times = [w[0] for w in window]
            axes[r][c].plot(
                *with_gaps(
                    times, [max(w[1], FLOOR_SECONDS) for w in window], 2 * WINDOW
                ),
                color=BLUE,
                label="mean",
            )
            axes[r][c].plot(
                *with_gaps(
                    times,
                    [w[2] if w[2] != float("inf") else None for w in window],
                    2 * WINDOW,
                ),
                color=ORANGE,
                label="90th percentile (bucket edge)",
            )
            seconds_axis(
                axes[r][c], f"{label}\n(s, {WINDOW:.0f} s window)" if c == 0 else ""
            )
    axes[0][-1].legend(frameon=False, fontsize=7, loc="upper right")
    finish(figure, axes, out, title + "  --  A3 engine-side latency (vLLM histograms)")


def plot_device(pyplot, device, out, title):
    cards = sorted({x["card"] for x in device})
    figure, axes = pyplot.subplots(
        4, len(cards), figsize=(4.4 * len(cards), 9), sharex=True, squeeze=False
    )
    rows = (
        ("aicore_usage_rate_percent", "AI Core busy\n(% of the card)"),
        ("hbm_bandwidth_usage_rate_percent", "HBM bandwidth\n(% of the card)"),
        ("npu_utilization_percent", "NPU utilisation\n(% of the card)"),
    )
    for c, card in enumerate(cards):
        lines = [x for x in device if x["card"] == card]
        t = [x["t"] for x in lines]
        axes[0][c].set_title(f"card {card}", color=INK)
        for r, (key, label) in enumerate(rows):
            axes[r][c].plot(t, [x.get(key) for x in lines], color=BLUE)
            axes[r][c].set_ylim(0, 105)
            style(axes[r][c], label if c == 0 else "")
        axes[3][c].plot(
            t, [sum(x["process_memory_mb"].values()) for x in lines], color=ORANGE
        )
        style(axes[3][c], "process memory\non the card (MB)" if c == 0 else "")
    finish(figure, axes, out, title + "  --  A4 device (npu-smi, every 5 s)")


def grouped(points, seconds):
    groups = defaultdict(list)
    for t, value in points:
        groups[int(t // seconds)].append(value)
    rows = []
    for index in sorted(groups):
        values = sorted(groups[index])
        if len(values) >= 5:
            rows.append(
                (
                    (index + 0.5) * seconds,
                    values[len(values) // 2],
                    values[int(0.9 * (len(values) - 1))],
                    values[-1],
                )
            )
    return rows


def plot_client_latency(pyplot, rounds, tokens, holds, out, title):
    ok = [r for r in rounds if r["status"] == "ok" and r["first_token_seconds"]]
    gaps = []
    for line in tokens:
        previous = line["chunks"][0][0] if line["chunks"] else None
        for arrival, count in line["chunks"][1:]:
            gaps.append((arrival, (arrival - previous) / max(1, count)))
            previous = arrival
    series = (
        (
            "time to first token\n(s, 60 s groups)",
            [
                (r["sent_seconds"], r["first_token_seconds"] - r["sent_seconds"])
                for r in ok
            ],
            60,
        ),
        ("gap between streamed\ntokens (s, 10 s groups)", gaps, 10),
        (
            "per-token latency: end-to-end\n/ output tokens (s, 60 s groups)",
            [
                (
                    r["sent_seconds"],
                    (r["finished_seconds"] - r["sent_seconds"])
                    / max(1, r["engine_completion_tokens"] or r["output_len"]),
                )
                for r in ok
            ],
            60,
        ),
        (
            "held by ThunderAgent before\nforwarding (s, 60 s groups;"
            "\n0 s drawn at 1 ms)",
            [(t, waited) for t, waited, _ in holds],
            60,
        ),
    )
    figure, axes = pyplot.subplots(
        len(series), 1, figsize=(13, 3 * len(series)), sharex=True
    )
    for axis, (label, points, seconds) in zip(axes, series, strict=True):
        rows = grouped(points, seconds)
        for index, (name, colour) in enumerate(
            (("median", AQUA), ("90th percentile", BLUE), ("maximum", ORANGE)), start=1
        ):
            axis.plot(
                *with_gaps(
                    [x[0] for x in rows],
                    [max(x[index], FLOOR_SECONDS) for x in rows],
                    2 * seconds,
                ),
                color=colour,
                marker="o",
                markersize=3,
                label=name,
            )
        seconds_axis(axis, label)
        axis.legend(frameon=False, fontsize=8, loc="upper right")
    finish(figure, axes, out, title + "  --  B1 client-side latency")


PHASE_COLOURS = {
    "queued by ThunderAgent": YELLOW,
    "waiting for first token": AQUA,
    "decoding": BLUE,
    "tool (action)": ORANGE,
    "paused by ThunderAgent": RED,
}


def program_intervals(decisions):
    """program -> [(start, end, kind)] for ThunderAgent pauses and queueing."""
    open_pause, open_queue, spans = {}, {}, defaultdict(list)
    for t, kind, program, _backend in sorted(decisions):
        if kind == "pause":
            open_pause[program] = t
        elif kind == "resume" and program in open_pause:
            spans[program].append(
                (open_pause.pop(program), t, "paused by ThunderAgent")
            )
        elif kind == "queue":
            open_queue[program] = t
        elif kind == "assign" and program in open_queue:
            spans[program].append(
                (open_queue.pop(program), t, "queued by ThunderAgent")
            )
    return spans


def plot_slots(pyplot, rounds, decisions, out, title):
    by_slot = defaultdict(list)
    for r in rounds:
        by_slot[r["slot"]].append(r)
    spans = program_intervals(decisions)
    slots = sorted(s for s in by_slot if s is not None)
    run_end = max((r["finished_seconds"] for r in rounds), default=0.0)
    figure, axis = pyplot.subplots(figsize=(15, 0.32 * len(slots) + 2))
    for row, slot in enumerate(slots):
        items = sorted(by_slot[slot], key=lambda r: r["sent_seconds"])
        for index, r in enumerate(items):
            first = r["first_token_seconds"] or r["finished_seconds"]
            blocks = [
                (r["sent_seconds"], first, "waiting for first token"),
                (first, r["finished_seconds"], "decoding"),
            ]
            following = items[index + 1] if index + 1 < len(items) else None
            if following and following["session_id"] == r["session_id"]:
                blocks.append(
                    (r["finished_seconds"], following["sent_seconds"], "tool (action)")
                )
            elif r.get("tool_wait_after_ms"):
                # the program's tool is still running when the run ends: draw it
                # up to its recorded end or the end of the run, whichever first
                blocks.append(
                    (
                        r["finished_seconds"],
                        min(
                            r["finished_seconds"] + r["tool_wait_after_ms"] / 1000,
                            run_end,
                        ),
                        "tool (action)",
                    )
                )
            for begin, end, phase in blocks:
                if end > begin:
                    axis.broken_barh(
                        [(begin, end - begin)],
                        (row - 0.4, 0.8),
                        facecolors=PHASE_COLOURS[phase],
                    )
            if index == 0 or items[index - 1]["session_id"] != r["session_id"]:
                axis.plot(
                    [r["sent_seconds"]] * 2,
                    [row - 0.45, row + 0.45],
                    color=INK,
                    linewidth=1,
                )  # a new program starts here
        for program in {r["session_id"] for r in items}:
            for begin, end, phase in spans.get(program, []):
                axis.broken_barh(
                    [(begin, end - begin)],
                    (row - 0.15, 0.3),
                    facecolors=PHASE_COLOURS[phase],
                )
    axis.set_yticks(range(len(slots)), [f"slot {s}" for s in slots], fontsize=7)
    axis.set_ylim(len(slots) - 0.5, -0.5)
    axis.grid(True, axis="x", color="#e4e3df")
    handles = [pyplot.Rectangle((0, 0), 1, 1, color=c) for c in PHASE_COLOURS.values()]
    axis.legend(
        handles,
        list(PHASE_COLOURS),
        ncol=5,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        fontsize=8,
    )
    axis.set_xlabel(
        "seconds since the run started  (a thin black line: a new "
        "program starts in that slot; thin bars inside: ThunderAgent "
        "paused / queued that program)",
        color=INK,
    )
    figure.suptitle(title + "  --  C1 closed-loop slots", color=INK, y=1.0)
    figure.tight_layout()
    figure.savefig(out, dpi=110)
    pyplot.close(figure)


def plot_program_states(pyplot, accounting, out, title):
    backends = sorted(accounting)
    figure, axes = pyplot.subplots(
        len(backends) + 1, 1, figsize=(13, 2.4 * (len(backends) + 1)), sharex=True
    )
    for axis, backend in zip(axes, backends, strict=False):
        rows = accounting[backend]
        t = [x["t"] for x in rows]
        axis.stackplot(
            t,
            [x["reasoning"] for x in rows],
            [x["acting"] for x in rows],
            colors=[BLUE, ORANGE],
            labels=["reasoning", "acting"],
        )
        style(axis, f"backend {backend}\nprograms")
        axis.legend(frameon=False, fontsize=8, loc="upper right")
    first = accounting[backends[0]] if backends else []
    axes[-1].plot(
        [x["t"] for x in first],
        [x["waiting"] for x in first],
        color=YELLOW,
        label="ThunderAgent's waiting queue (all backends)",
    )
    style(axes[-1], "programs")
    axes[-1].legend(frameon=False, fontsize=8, loc="upper right")
    finish(
        figure,
        axes,
        out,
        title + "  --  C2 programs per ThunderAgent state (its 5 s tick)",
    )


def plot_context_growth(pyplot, rounds, out, title):
    figure, axis = pyplot.subplots(figsize=(13, 5))
    by_program = defaultdict(list)
    for r in rounds:
        if r["engine_prompt_tokens"]:
            by_program[r["session_id"]].append(
                (r["sent_seconds"], r["engine_prompt_tokens"])
            )
    for points in by_program.values():
        points.sort()
        axis.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            color=BLUE,
            linewidth=0.6,
            alpha=0.5,
            marker=".",
            markersize=2,
        )
    style(axis, "prompt tokens of the round\n(one line per program)")
    axis.set_xlabel("when the round was sent (seconds since start)", color=INK)
    figure.suptitle(title + "  --  C3 context growth", color=INK)
    figure.tight_layout()
    figure.savefig(out, dpi=110)
    pyplot.close(figure)


DECISION_COLOURS = {
    "assign": GREY,
    "queue": YELLOW,
    "mark": VIOLET,
    "pause": RED,
    "resume": AQUA,
}


def plot_decisions(pyplot, decisions, out, title):
    backends = sorted({b for _, _, _, b in decisions if b})
    figure, axes = pyplot.subplots(
        2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )
    rows = [(b, k) for b in backends for k in DECISION_COLOURS] + [("-", "queue")]
    for index, (backend, kind) in enumerate(rows):
        times = [
            t
            for t, k, _, b in decisions
            if k == kind and (b == backend or (backend == "-" and kind == "queue"))
        ]
        axes[0].eventplot(
            [times], lineoffsets=index, linelengths=0.8, colors=DECISION_COLOURS[kind]
        )
    axes[0].set_yticks(
        range(len(rows)),
        [f"{b} {k}" if b != "-" else "queue (new)" for b, k in rows],
        fontsize=7,
    )
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    axes[0].set_ylabel("backend, decision", color=INK)
    migrations = 0
    last_backend = {}
    for t, kind, program, backend in sorted(decisions):
        if kind in ("assign", "resume"):
            if kind == "resume" and last_backend.get(program) not in (None, backend):
                migrations += 1
                axes[1].axvline(t, color=INK, linewidth=0.5, alpha=0.4)
            last_backend[program] = backend
    for kind, colour in DECISION_COLOURS.items():
        counts = defaultdict(int)
        for t, k, _, _ in decisions:
            if k == kind:
                counts[int(t // WINDOW)] += 1
        end = int(max((t for t, *_ in decisions), default=0) // WINDOW) + 1
        axes[1].plot(
            [(i + 0.5) * WINDOW for i in range(end)],
            [counts.get(i, 0) for i in range(end)],
            color=colour,
            label=f"{kind} ({sum(counts.values())})",
        )
    axes[1].plot(
        [],
        [],
        color=INK,
        linewidth=0.5,
        label=f"grey line: resumed on another backend ({migrations})",
    )
    style(axes[1], f"decisions per {WINDOW:.0f} s")
    axes[1].legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
    finish(figure, [axes[1]], out, title + "  --  D1 ThunderAgent decisions")


def plot_pause_cost(pyplot, profile, holds, started_at_wall, out, title):
    figure, axes = pyplot.subplots(2, 1, figsize=(13, 6), sharex=True)
    points = [
        (float(r["completed_at"]) - started_at_wall, float(r["pause_s"]))
        for r in profile
        if r.get("pause_s")
    ]
    per_minute = defaultdict(float)
    for t, pause in points:
        per_minute[int(t // 60)] += pause
    end = int(max((t for t, _ in points), default=0) // 60) + 1
    axes[0].bar(
        [(i + 0.5) * 60 for i in range(end)],
        [per_minute.get(i, 0.0) for i in range(end)],
        width=55,
        color=RED,
    )
    style(axes[0], "program-seconds paused\nper minute (ThunderAgent\nstep_profiles)")
    for outcome, colour in (("paused", RED), ("queued", YELLOW)):
        rows = grouped([(t, w) for t, w, o in holds if o == outcome], 60)
        axes[1].plot(
            [x[0] for x in rows],
            [x[3] for x in rows],
            color=colour,
            marker="o",
            markersize=3,
            label=f"{outcome}: longest hold",
        )
        axes[1].plot(
            [x[0] for x in rows],
            [x[1] for x in rows],
            color=colour,
            linestyle="--",
            label=f"{outcome}: median hold",
        )
    style(axes[1], "hold before a request\nis forwarded (s,\n60 s groups)")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    finish(figure, [axes[1]], out, title + "  --  D2 the cost of ThunderAgent's holds")


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    run = Path(sys.argv[1])
    figures = run / "figures"
    figures.mkdir(exist_ok=True)
    summary = json.loads((run / "summary.json").read_text())
    started = summary["started_at_wall"]
    capacity = {
        e["endpoint"]: e["kv_capacity_tokens_device"] + e["kv_capacity_tokens_host"]
        for e in summary.get("engines", [])
    }
    series = engine_series(read_jsonl(run / "engine_metrics_raw.jsonl.gz", True))
    device = read_jsonl(run / "device.jsonl.gz", True)
    rounds = read_jsonl(run / "rounds.jsonl")
    tokens = read_jsonl(run / "tokens.jsonl")
    events = log_events(run / "thunderagent.log", started)
    profile_path = run / "thunderagent-profile" / "step_profiles.csv"
    profile = list(csv.DictReader(profile_path.open())) if profile_path.exists() else []
    title = run.name
    plot_engine_state(
        pyplot,
        series,
        capacity,
        events["state"],
        figures / "A1_engine_state.png",
        title,
    )
    plot_engine_throughput(pyplot, series, figures / "A2_engine_throughput.png", title)
    plot_engine_latency(pyplot, series, figures / "A3_engine_latency.png", title)
    if device:
        plot_device(pyplot, device, figures / "A4_device.png", title)
    plot_client_latency(
        pyplot,
        rounds,
        tokens,
        events["holds"],
        figures / "B1_client_latency.png",
        title,
    )
    plot_slots(
        pyplot, rounds, events["decisions"], figures / "C1_slots_timeline.png", title
    )
    if events["state"]:
        plot_program_states(
            pyplot, events["state"], figures / "C2_program_states.png", title
        )
    plot_context_growth(pyplot, rounds, figures / "C3_context_growth.png", title)
    if events["decisions"]:
        plot_decisions(pyplot, events["decisions"], figures / "D1_decisions.png", title)
        plot_pause_cost(
            pyplot,
            profile,
            events["holds"],
            started,
            figures / "D2_pause_cost.png",
            title,
        )
    print("wrote", ", ".join(sorted(p.name for p in figures.glob("[A-D]*.png"))))


if __name__ == "__main__":
    main()
