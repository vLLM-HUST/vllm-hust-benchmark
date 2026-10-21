"""Generate a layered system-performance report from existing Agent evidence.

No inference, deployment, external publication or serving dependencies.
See docs/reference/SYSTEM_RESEARCH_REPORTING.md for the input/output contract.
"""

from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from agent_report_data import EvidenceError, load_run, counter_delta, clipped_duration
from plot_run import BLUE, ORANGE, AQUA, INK, GRID, style


def format_number(value, digits=2):
    return (
        f"{value:.{digits}f}"
        if value is not None and math.isfinite(value)
        else "not recorded"
    )


def number(value):
    return np.nan if value is None else value


def safe_product(a, b):
    return number(a) * number(b)


def program_states(row):
    if row.get("programs") is None:
        return {"active": np.nan, "paused": np.nan}
    return collections.Counter(
        (p.get("state", "unknown") for p in row["programs"].values())
    )


def queue_length(row):
    scheduler = row.get("scheduler")
    return (
        len(scheduler["waiting"])
        if scheduler is not None and "waiting" in scheduler
        else np.nan
    )


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and (not math.isfinite(value)):
        return None
    return value


def render(data, output, window, boundaries, rate_window):
    DATA, OUT = (data, output)
    COL, LS = (
        {"native": BLUE, "controller": ORANGE},
        {"native": "-", "controller": "--"},
    )
    agent_count = data["native"]["agent_count"]
    cache_gib = number(data["native"]["cache_gib_per_rank"])
    full_key = f"0-{window:g}"
    phase_boundaries = boundaries
    edges = [0, *boundaries, window]
    phases = list(zip(edges, edges[1:])) + [(0, window)]
    cadence = "/".join(
        (
            f"{np.median(np.diff([r['t'] for r in d['rows']])):.2f}"
            for d in DATA.values()
        )
    )
    action_times = [
        r["snapshot_finished_epoch"] for r in DATA["controller"]["actions"] or []
    ]
    action_cadence = (
        f"{np.median(np.diff(action_times)):.2f} s"
        if len(action_times) > 1
        else "not recorded"
    )

    def diff(first, last, key):
        return number(counter_delta(first, last, key))

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "savefig.facecolor": "white",
            "svg.hashsalt": "agent-comparison-report-v1",
        }
    )

    def line(ax, t, y, arm, label=None):
        ax.plot(t, y, color=COL[arm], ls=LS[arm], lw=1.7, label=label or arm.title())

    def timeaxis(ax, title, ylabel, limit=None):
        style(ax, ylabel)
        ax.set_title(title, loc="left")
        ax.set_xlim(0, window)
        ax.set_xlabel("Seconds since each arm started")
        ax.set_ylim(bottom=0)
        if limit:
            ax.set_ylim(*limit)
        for t in phase_boundaries:
            ax.axvline(t, color=GRID, ls=":", lw=0.9)
        ax.legend(frameon=False)

    def figure(rows, cols, title, footer, size=None):
        fig, axes = plt.subplots(rows, cols, figsize=size or (12.4, 3.1 * rows + 0.7))
        fig.subplots_adjust(
            left=0.08,
            right=0.97,
            top=0.82 if rows == 1 else 0.9,
            bottom=0.22 if rows == 1 else 0.12,
            hspace=0.55,
            wspace=0.3,
        )
        fig.suptitle(title, fontsize=14, color=INK, y=0.98)
        fig.text(0.5, 0.025, footer, ha="center", fontsize=9, color=INK)
        return (fig, np.asarray(axes).reshape(rows, cols))

    def save(fig, name):
        for ax in fig.axes:
            arm_lines = [
                line
                for line in ax.lines
                if line.get_label() in ("Native", "Controller")
            ]
            missing_lines = arm_lines and (
                not any((np.isfinite(line.get_ydata()).any() for line in arm_lines))
            )
            missing_bars = ax.patches and (
                not any((np.isfinite(p.get_height()) for p in ax.patches))
            )
            if missing_lines or missing_bars:
                ax.text(
                    0.5,
                    0.5,
                    "Not recorded",
                    transform=ax.transAxes,
                    ha="center",
                    color=INK,
                )
        for ext in ("png", "svg"):
            fig.savefig(
                OUT / f"{name}.{ext}",
                dpi=160,
                metadata={"Date": None} if ext == "svg" else {},
            )
        plt.close(fig)

    def rolling(rows, key, seconds=rate_window, per_count=None):
        outt = []
        outv = []
        i = 0
        for j, b in enumerate(rows):
            while i + 1 < j and rows[i + 1]["t"] <= b["t"] - seconds:
                i += 1
            a = rows[i]
            dt = b["t"] - a["t"]
            if dt < seconds * 0.9:
                continue
            n = diff(a, b, per_count) if per_count else dt
            outt.append(b["t"])
            outv.append(diff(a, b, key) / n if n > 0 else np.nan)
        return (outt, outv)

    f, axs = figure(
        3,
        2,
        f"ENGINE 1 | Backend queue, cache pool and work processed | {str(agent_count)} Agents per arm",
        f"Own-arm 0–{f'{window:g}'} s; {f'{cadence} s median'} samples. Rates: trailing {f'{rate_window:g} s'} counter deltas / measured seconds. No device-HBM measurement.",
        (12.4, 10.3),
    )
    for arm, d in DATA.items():
        rows = d["rows"]
        t = [x["t"] for x in rows]
        for ax, key in [
            (axs[0, 0], "num_requests_running"),
            (axs[0, 1], "num_requests_waiting"),
        ]:
            line(ax, t, [x["m"].get("vllm:" + key, np.nan) for x in rows], arm)
        line(
            axs[1, 0],
            t,
            [cache_gib * x["m"].get("vllm:kv_cache_usage_perc", np.nan) for x in rows],
            arm,
        )
        tx, y = rolling(
            rows, "prefix_cache_hits_total", per_count="prefix_cache_queries_total"
        )
        line(axs[1, 1], tx, np.asarray(y) * 100, arm)
        for ax, key in [
            (axs[2, 0], "prompt_tokens_by_source_total:local_compute"),
            (axs[2, 1], "generation_tokens_total"),
        ]:
            tx, y = rolling(rows, key)
            line(ax, tx, y, arm)
    for ax, title, ylabel, lim in [
        (axs[0, 0], "A  Requests in model execution batches", "Requests", None),
        (axs[0, 1], "B  Requests waiting inside vLLM", "Requests", None),
        (
            axs[1, 0],
            f"C  Cache-pool use per rank (capacity: {(f'{cache_gib:g} GiB' if np.isfinite(cache_gib) else 'unknown GiB')})",
            f"GiB = pool usage fraction × {(f'{cache_gib:g} GiB' if np.isfinite(cache_gib) else 'unknown GiB')}",
            (0, cache_gib * 1.05) if np.isfinite(cache_gib) else None,
        ),
        (
            axs[1, 1],
            "D  Prefix-cache token hit rate",
            "% = hit tokens / queried tokens",
            (0, 100),
        ),
        (
            axs[2, 0],
            "E  Locally computed prompt-token counter",
            "Input tokens / second",
            None,
        ),
        (
            axs[2, 1],
            "F  Engine generation-token counter",
            "Output tokens / second",
            None,
        ),
    ]:
        timeaxis(ax, title, ylabel, lim)
    if np.isfinite(cache_gib):
        axs[1, 0].axhline(cache_gib, color=INK, ls=":", lw=1)
    else:
        axs[1, 0].text(
            0.5,
            0.5,
            "Cache capacity not recorded",
            transform=axs[1, 0].transAxes,
            ha="center",
        )
    save(f, "engine-load")
    f, axs = figure(
        2,
        2,
        "ENGINE 2 | Waiting, execution and batch processing",
        "Engine-global histograms; membership = observations added in each interval. No client-request join; means exclude unfinished observations.",
        (12.4, 7.6),
    )
    names = [f"{lo:g}-{hi:g}" for lo, hi in phases[:-1]]
    xx = np.arange(len(names))
    summary = []
    finite_bounds = [
        k
        for d in DATA.values()
        for k, _ in d["phases"][full_key]["histograms"]["iteration_tokens_total"][
            "buckets"
        ]
        if k != "Inf"
    ]
    max_bound = max(finite_bounds, default=1)
    overflow_at = max_bound * 2
    for a, arm in enumerate(DATA):
        d = DATA[arm]
        for ax, key in [
            (axs[0, 0], "request_queue_time_seconds"),
            (axs[0, 1], "request_prefill_time_seconds"),
            (axs[1, 0], "request_decode_time_seconds"),
        ]:
            ys = [number(d["phases"][p]["histograms"][key]["mean"]) for p in names]
            bs = ax.bar(
                xx + (a - 0.5) * 0.34,
                ys,
                0.34,
                color=COL[arm],
                label=arm.title(),
                edgecolor=INK if a else "white",
                linewidth=0.8,
            )
            ax.bar_label(
                bs, labels=[format_number(v, 2) for v in ys], padding=3, fontsize=9
            )
        h = d["phases"][full_key]["histograms"]["iteration_tokens_total"]
        finite = [(overflow_at if k == "Inf" else k, v) for k, v in h["buckets"]]
        axs[1, 1].step(
            [v[0] for v in finite],
            [100 * v[1] / h["count"] if h["count"] else np.nan for v in finite],
            where="post",
            color=COL[arm],
            ls=LS[arm],
            label=arm.title(),
        )
        for p in names + [full_key]:
            z = d["phases"][p]
            summary.append(
                {
                    "arm": arm,
                    "interval": p,
                    "samples": z["engine_samples"],
                    "measured_seconds": z["engine_sample_duration_s"],
                    "engine_running_mean": z["engine_time_weighted_means"][
                        "num_requests_running"
                    ],
                    "engine_waiting_mean": z["engine_time_weighted_means"][
                        "num_requests_waiting"
                    ],
                    "cache_pool_mean_GiB_per_rank": safe_product(
                        cache_gib,
                        z["engine_time_weighted_means"]["kv_cache_usage_perc"],
                    ),
                    "queue_mean_s": z["histograms"]["request_queue_time_seconds"][
                        "mean"
                    ],
                    "prefill_mean_s": z["histograms"]["request_prefill_time_seconds"][
                        "mean"
                    ],
                    "decode_mean_s": z["histograms"]["request_decode_time_seconds"][
                        "mean"
                    ],
                    "engine_local_prompt_tokens_per_s": z[
                        "engine_counter_rates_per_second"
                    ]["prompt_tokens_by_source_total:local_compute"],
                    "engine_generation_tokens_per_s": z[
                        "engine_counter_rates_per_second"
                    ]["generation_tokens_total"],
                    "iteration_histogram_observations": z["histograms"][
                        "iteration_tokens_total"
                    ]["count"],
                    "tokens_per_iteration_observation_mean": z["histograms"][
                        "iteration_tokens_total"
                    ]["mean"],
                    "client_returns": z["client"]["returns"],
                    "client_output_tokens": z["client"]["output_tokens"],
                    "client_latency_mean_s": z["client"]["mean_latency_s"],
                    "client_latency_p95_s": z["client"]["p95_latency_s"],
                }
            )
    for ax, title in [
        (axs[0, 0], "A  Engine queue time"),
        (axs[0, 1], "B  Engine prefill time"),
        (axs[1, 0], "C  Engine decode time"),
    ]:
        style(ax, "Mean seconds / histogram observation")
        ax.set_title(title, loc="left")
        ax.set_xticks(xx, [f"{lo:g}–{hi:g} s" for lo, hi in phases[:-1]])
        ax.set_xlabel("Interval since each arm started")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        ax.legend(frameon=False)
    ax = axs[1, 1]
    style(ax, "Cumulative % of histogram observations")
    ax.set_title("D  Reported iteration-token distribution", loc="left")
    ax.set_xscale("log", base=2)
    ax.set_xlim(1, overflow_at)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Tokens / reported iteration (bucket boundary)")
    ticks = sorted(
        set(
            [
                1,
                *[
                    float(2 ** round(x))
                    for x in np.linspace(0, math.log2(max_bound), 4)
                ],
                overflow_at,
            ]
        )
    )
    ax.set_xticks(
        ticks, [f">{max_bound:g}" if x == overflow_at else f"{x:g}" for x in ticks]
    )
    ax.legend(frameon=False, fontsize=8)
    save(f, "engine-latency-and-batches")
    f, axs = figure(
        2,
        2,
        "CONTROLLER | Admission, paused programs and repeated actions",
        f"Own-arm 0–{f'{window:g}'} s; state samples {f'{cadence} s median'}. Action totals from {action_cadence} log snapshots; not exact event timestamps. Estimates are not HBM.",
        (12.4, 7.6),
    )
    for arm, d in DATA.items():
        rows = d["rows"]
        t = [x["t"] for x in rows]
        states = [program_states(x) for x in rows]
        for ax, key in [(axs[0, 0], "active"), (axs[0, 1], "paused")]:
            line(ax, t, [s.get(key, 0) for s in states], arm)
        line(axs[1, 0], t, [queue_length(x) for x in rows], arm)
    samples = DATA["controller"]["actions"] or []
    epoch = DATA["controller"]["origin_epoch"]
    ts = [0]
    vals = {k: [0] for k in ["Marked", "Paused", "Resumed"]}
    for x in samples:
        t = x["snapshot_finished_epoch"] - epoch
        if not 0 <= t < window:
            continue
        ts.append(t)
        for k in vals:
            vals[k].append(vals[k][-1] + x["counts"].get(k, 0))
    for k, c, ls in [
        ("Marked", BLUE, "-"),
        ("Paused", ORANGE, "--"),
        ("Resumed", AQUA, ":"),
    ]:
        if samples:
            axs[1, 1].step(ts, vals[k], where="post", color=c, ls=ls, label=k)
    if not samples:
        axs[1, 1].text(
            0.5,
            0.5,
            "Action log not recorded",
            transform=axs[1, 1].transAxes,
            ha="center",
        )
    for ax, title, ylabel, lim in [
        (
            axs[0, 0],
            "A  Active programs (not engine-running requests)",
            f"Programs out of {str(agent_count)}",
            (0, agent_count * 1.04),
        ),
        (
            axs[0, 1],
            "B  Paused programs",
            f"Programs out of {str(agent_count)}",
            (0, agent_count * 1.04),
        ),
        (
            axs[1, 0],
            "C  Controller admission queue",
            "Programs",
            (0, agent_count * 1.04),
        ),
        (
            axs[1, 1],
            "D  Controller repeated action counts",
            "Cumulative events (not unique programs)",
            None,
        ),
    ]:
        timeaxis(ax, title, ylabel, lim)
    save(f, "controller")
    f, axs = figure(
        2,
        2,
        f"PROGRAM | Progress of all {str(agent_count)} Agents, including zero-reply Agents",
        f"Rows = fixed task indices 0–{str(agent_count - 1)}; same color scale. Time fractions divide by {str(agent_count)} × {f'{window:g}'} Agent-seconds; query time includes every wait.",
        (12.4, 8.1),
    )
    maxcount = max(
        1, max((len(p["returns"]) for d in DATA.values() for p in d["programs"]))
    )
    grid = np.r_[np.arange(0, window, rate_window), window]
    fraction = {}
    for idx, (arm, d) in enumerate(DATA.items()):
        mat = np.array(
            [
                np.searchsorted(
                    sorted((v["t"] for v in p["returns"])), grid, side="right"
                )
                for p in d["programs"]
            ]
        )
        im = axs[0, idx].imshow(
            mat,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=[0, window, -0.5, agent_count - 0.5],
            vmin=0,
            vmax=maxcount,
            cmap="viridis",
        )
        axs[0, idx].set_title(
            f"{arm.title()}: cumulative replies per Agent", loc="left"
        )
        axs[0, idx].set_xlabel("Seconds since each arm started")
        axs[0, idx].set_ylabel(f"Agent index (all {str(agent_count)})")
        f.colorbar(
            im, ax=axs[0, idx], label="Returned calls / Agent", fraction=0.046, pad=0.03
        )
        counts = sorted((len(p["returns"]) for p in d["programs"]))
        axs[1, 0].step(
            counts,
            np.arange(1, agent_count + 1) / agent_count * 100,
            where="post",
            color=COL[arm],
            ls=LS[arm],
            label=arm.title(),
        )
        q = sum((clipped_duration(p["query_segments"], window) for p in d["programs"]))
        tool = sum(
            (clipped_duration(p["tool_segments"], window) for p in d["programs"])
        )
        if q + tool > agent_count * window + 0.01:
            raise EvidenceError(
                "Overlapping query/tool events exceed available Agent time"
            )
        fraction[arm] = {
            "query": q / (agent_count * window),
            "tool": tool / (agent_count * window),
            "other": max(0, 1 - (q + tool) / (agent_count * window)),
        }
        base = 0
        for k, c in [("query", BLUE), ("tool", ORANGE), ("other", GRID)]:
            v = fraction[arm][k] * 100
            axs[1, 1].barh(idx, v, left=base, color=c, label=k if idx == 0 else None)
            base += v
        axs[1, 1].text(
            2,
            idx,
            f"query {fraction[arm]['query'] * 100:.3f}%; tool {fraction[arm]['tool'] * 100:.3f}%",
            va="center",
            color="white",
            fontsize=9,
        )
    ax = axs[1, 0]
    style(ax, f"Cumulative % of {str(agent_count)} Agents")
    ax.set_title(f"C  Per-Agent completed calls at {f'{window:g}'} s", loc="left")
    ax.set_xlabel("Normally returned model calls / Agent")
    ax.set_ylim(0, 105)
    ax.set_xlim(left=0)
    ax.legend(frameon=False)
    ax = axs[1, 1]
    style(ax, "")
    ax.set_title(
        f"D  Agent time allocation (all {str(agent_count)} Agents)", loc="left"
    )
    ax.set_yticks([0, 1], ["Native", "Controller"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of total Agent-seconds")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.52), ncol=3)
    save(f, "program")
    f, axs = figure(
        2,
        2,
        "REQUEST | What clients received, and how long normal replies took",
        f"Own-arm 0–{window:g} s. Rates: disjoint {rate_window:g} s bins / bin seconds. Latencies: normal replies only; pending {DATA['native']['pending']}/{DATA['controller']['pending']} excluded.",
        (12.4, 7.6),
    )
    for arm, d in DATA.items():
        rr = d["returns"]
        t = np.array([x["t"] for x in rr])
        edges = np.r_[np.arange(0, window, rate_window), window]
        cnt = np.histogram(t, edges)[0]
        tok = np.histogram(t, edges, weights=[x["output"] for x in rr])[0]
        for ax, y in [
            (axs[0, 0], cnt / np.diff(edges)),
            (axs[0, 1], tok / np.diff(edges)),
        ]:
            line(ax, edges[1:], y, arm)
        lat = np.sort([x["latency"] for x in rr])
        axs[1, 0].plot(
            lat,
            np.arange(1, len(lat) + 1) / len(lat) * 100,
            color=COL[arm],
            ls=LS[arm],
            label=f"{arm.title()} (n={len(lat)})",
        )
        axs[1, 1].scatter(
            [x["input"] for x in rr],
            [x["latency"] for x in rr],
            s=5,
            alpha=0.2,
            c=COL[arm],
            marker="o" if arm == "native" else "^",
            label=arm.title(),
        )
    for ax, title, y in [
        (axs[0, 0], "A  Normally returned calls", "Calls / second"),
        (axs[0, 1], "B  Returned output throughput", "Output tokens / second"),
    ]:
        timeaxis(ax, title, y)
    ax = axs[1, 0]
    style(ax, "Cumulative % of normal replies")
    ax.set_title("C  Client end-to-end latency (completed only)", loc="left")
    ax.set_xlabel("Seconds / model call")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 105)
    ax.legend(frameon=False)
    ax = axs[1, 1]
    style(ax, "Client end-to-end latency (seconds)")
    ax.set_title("D  Input length versus client latency", loc="left")
    ax.set_xlabel("Input tokens / normally returned call")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    save(f, "request")
    f, axs = figure(
        1,
        2,
        "REQUEST DETAIL | Mean and 95th-percentile client latency by interval",
        "Normally returned calls grouped by return time; unfinished calls excluded. Percentiles are empirical; one pair, no confidence intervals.",
        (12.4, 4.9),
    )
    for a, (arm, d) in enumerate(DATA.items()):
        for ax, key in [(axs[0, 0], "mean_latency_s"), (axs[0, 1], "p95_latency_s")]:
            y = [number(d["phases"][p]["client"][key]) for p in names]
            bars = ax.bar(
                xx + (a - 0.5) * 0.34, y, 0.34, color=COL[arm], label=arm.title()
            )
            ax.bar_label(
                bars, labels=[format_number(v, 1) for v in y], padding=3, fontsize=9
            )
    for ax, title in [
        (axs[0, 0], "A  Mean client end-to-end latency"),
        (axs[0, 1], "B  95th-percentile client end-to-end latency"),
    ]:
        style(ax, "Seconds / normally returned model call")
        ax.set_title(title, loc="left")
        ax.set_xticks(xx, [f"{lo:g}–{hi:g} s" for lo, hi in phases[:-1]])
        ax.set_xlabel("Return-time interval since each arm started")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.13)
        ax.legend(frameon=False)
    save(f, "request-latency-by-interval")
    metric_names = [
        "num_requests_running",
        "num_requests_waiting",
        "num_requests_waiting_by_reason:capacity",
        "num_requests_waiting_by_reason:deferred",
        "kv_cache_usage_perc",
        "prompt_tokens_total",
        "prompt_tokens_by_source_total:local_compute",
        "prompt_tokens_by_source_total:local_cache_hit",
        "generation_tokens_total",
        "prefix_cache_queries_total",
        "prefix_cache_hits_total",
        "num_preemptions_total",
    ]
    with (OUT / "engine-controller-samples.csv").open("w") as f:
        fields = (
            ["arm", "seconds_since_arm_start"]
            + metric_names
            + ["active_programs", "paused_programs", "controller_queue_programs"]
        )
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for arm, d in DATA.items():
            for row in d["rows"]:
                state = program_states(row)
                w.writerow(
                    {
                        "arm": arm,
                        "seconds_since_arm_start": row["t"],
                        **{k: row["m"].get("vllm:" + k) for k in metric_names},
                        "active_programs": clean_json(state.get("active", np.nan)),
                        "paused_programs": clean_json(state.get("paused", np.nan)),
                        "controller_queue_programs": clean_json(queue_length(row)),
                    }
                )
    with (OUT / "normal-client-returns.csv").open("w") as f:
        fields = ["arm", "program", "t", "start", "latency", "input", "output"]
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for arm, d in DATA.items():
            for row in sorted(d["returns"], key=lambda r: r["t"]):
                w.writerow({"arm": arm, **row})
    f, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))
    f.subplots_adjust(left=0.08, right=0.97, top=0.8, bottom=0.24, wspace=0.3)
    for i, arm in enumerate(["native", "controller"]):
        for ax, key, scale in [
            (axes[0], "time_to_first_token_seconds", 1),
            (axes[1], "request_time_per_output_token_seconds", 1000),
        ]:
            y = [
                number(DATA[arm]["phases"][p]["histograms"][key]["mean"]) * scale
                for p in names
            ]
            bars = ax.bar(
                xx + (i - 0.5) * 0.34,
                y,
                0.34,
                color=[BLUE, ORANGE][i],
                label=arm.title(),
            )
            ax.bar_label(
                bars, labels=[format_number(v, 1) for v in y], padding=3, fontsize=10
            )
    for ax, title, ylabel in [
        (
            axes[0],
            "A  Engine time to first token",
            "Mean seconds / first-token observation",
        ),
        (
            axes[1],
            "B  Engine time per output token",
            "Mean milliseconds / output token (request-weighted)",
        ),
    ]:
        style(ax, ylabel)
        ax.set_title(title, loc="left")
        ax.set_xticks(xx, [f"{lo:g}–{hi:g} s" for lo, hi in phases[:-1]])
        ax.set_xlabel("Observation interval since each arm started")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        ax.legend(frameon=False)
    f.suptitle(
        "ENGINE 3 | Internal first-token and per-output-token latency",
        fontsize=14,
        color=INK,
    )
    f.text(
        0.5,
        0.055,
        "Global engine histogram deltas; sum delta / count delta. First-token and completed-request cohorts differ.",
        ha="center",
        fontsize=9,
        color=INK,
    )
    f.text(
        0.5,
        0.015,
        "Excludes Controller admission wait. Non-streaming clients do not observe these token arrival times.",
        ha="center",
        fontsize=9,
        color=INK,
    )
    save(f, "engine-token-latency")
    return (summary, fraction)


def write_report(
    data, output, window, boundaries, sources, summary, fractions, rate_window, question
):
    """Write only computed numbers and explicit interpretation boundaries."""
    full = f"0-{window:g}"
    native, controller = (data["native"], data["controller"])
    total_n, total_c = (native["phases"][full], controller["phases"][full])
    missing = [
        "连续的逐卡NPU计算利用率、总HBM已用GiB及HBM带宽：此归档适配器没有这些设备数据，不能判断硬件饱和。",
        "逐请求client→proxy→engine关联：引擎全局统计与客户端集合不能视为已经一一匹配。",
        "实际scheduler step的token数、执行时长和kernel/通信profile：iteration直方图不能直接解释为调度预算利用率。",
        "初始输入一致性需要实验验收或输入哈希另行核对；有两臂数据本身不证明同输入。",
        "精确逐program暂停时长及原子决策理由：状态/动作快照不能替代逐事件追踪。",
        "反向顺序重复及观测开销标定：单一Native→Controller配对不提供稳定因果收益或置信区间。",
    ]
    for arm, dataset in data.items():
        if dataset["cache_gib_per_rank"] is None:
            missing.append(
                f"{arm}：没有记录缓存池字节容量，缓存GiB图不可用；不能猜容量。"
            )
        if dataset["actions"] is None:
            missing.append(f"{arm}：没有动作采样文件，动作次数保持缺失；缺失不是零次。")
        for key, value in dataset["phases"][full]["engine_time_weighted_means"].items():
            if value is None:
                missing.append(f"{arm}：指标vllm:{key}未记录。")
        for key, h in dataset["phases"][full]["histograms"].items():
            if not h["available"]:
                missing.append(f"{arm}：直方图vllm:{key}未记录。")
    record = {
        "schema_version": 1,
        "research_question": question,
        "run_name": native["run_name"],
        "window_seconds": window,
        "phase_boundaries_seconds": boundaries,
        "rate_window_seconds": rate_window,
        "agent_count_per_arm": native["agent_count"],
        "histogram_scope": "Interval endpoint sum/count/bucket deltas; cohorts differ by metric; unfinished observations excluded.",
        "cache_scope": "Per-rank cache pool occupancy equivalent; not total HBM used; not additive TP capacity.",
        "client_scope": "Normal replies grouped by return time; pending and failures separate.",
        "missing_data_and_limits": missing,
        "time_fractions": fractions,
        "arms": {
            arm: {
                k: d[k]
                for k in (
                    "origin_epoch",
                    "agent_count",
                    "cache_gib_per_rank",
                    "settings",
                    "phases",
                    "pending",
                    "errors",
                )
            }
            for arm, d in data.items()
        },
    }
    (output / "data.json").write_text(
        json.dumps(clean_json(record), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    (output / "source-sha256.json").write_text(json.dumps(sources, indent=2) + "\n")
    with (output / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(summary[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(clean_json(summary))
    fmt = format_number

    def row(name, a, b):
        return f"| {name} | {fmt(a)} | {fmt(b)} |"

    report = [
        "# Agent对照：按系统层次生成的性能报告",
        "",
        "## 研究目标",
        "",
        question,
        "",
        "## 结论与证据范围",
        "",
        f"两组各{native['agent_count']}个Agent，分别观察开始后的0–{window:g}秒；本报告只离线读取归档，不启动推理、实验或策略变更。",
        f"正常返回调用{total_n['client']['returns']}/{total_c['client']['returns']}次，返回输出吞吐{fmt(total_n['client']['output_tokens'] / window)}/{fmt(total_c['client']['output_tokens'] / window)} tokens/s。",
        "这些是一次配对的观察值；下面逐层检查行为、代价和数据缺口，不自动将吞吐变化归为唯一调度因果。",
        "",
        "## 模块、层次与链路",
        "",
        "Program按依赖产生模型调用和工具执行 → Controller准入/暂停/恢复 → Engine排队、prefill、decode → Request返回并推动Program继续。Engine内部指标不包含Controller全部等待；设备HBM不是缓存池。",
        "",
        "| 层次 | 要回答的问题 | 图与口径 |",
        "|---|---|---|",
        "| Controller | 工作在哪里被拦住？为什么暂停？ | active/paused、准入队列、动作计数；program/事件粒度 |",
        "| Engine | 等待或执行哪里变慢？是否有资源压力？ | 请求数、缓存GiB、输入/输出计数速率、直方图阶段延迟 |",
        "| Program | 各Agent是否持续推进，是否不公平？ | 每Agent累计回复、分布、总Agent秒时间分配 |",
        "| Request | 客户端拿到多少结果，延迟与长尾怎样？ | 返回calls/s、tokens/s、端到端延迟和输入长度 |",
        "",
        "## 单位、分母和时间",
        "",
        f"- 横轴分别取各组最早实际到达为0，窗口0≤t<{window:g}秒；状态平均使用相邻样本持续时间加权，不外推。",
        f"- 引擎速率为向后{rate_window:g}秒的计数增量/实际秒数；客户端为不重叠{rate_window:g}秒桶的增量/桶宽，最后不足整桶时用真实桶宽。",
        "- 缓存池字节容量从本轮cache_config_info读取，不硬编码16GiB；每rank使用量=池使用率×池容量，不是总HBM。",
        "- prefix命中率分母是同一窗口查询tokens；时间分配分母是实际Agent数×窗口秒数。",
        "- 直方图均值=sum增量/count增量，按观察进入时间归组；分位数桶仅表示范围。客户端p95从正常返回调用原始时长计算，按返回时间分组。",
        f"- 窗口末未返回：Native {native['pending']}、Controller {controller['pending']}；窗口内错误{native['errors']}/{controller['errors']}。未完成调用不纳入正常返回延迟，不能据此保证全部请求延迟。",
        f"- Agent预算取输入设置：{json.dumps(native['settings'], ensure_ascii=False)}。设置缺失保持null。",
        "",
        "## 1. Controller：准入和外部等待",
        "",
        "![Controller准入和动作](controller.png)",
        "",
        "active是Controller登记状态，不是引擎执行请求数。paused/准入队列显示引擎外等待；动作次数可以重复，不是独立Agent数。动作快照时间不是精确事件时刻。",
        "",
        "## 2. Engine：负载、缓存和阶段耗时",
        "",
        "![引擎负载、缓存池和吞吐](engine-load.png)",
        "",
        "先看执行/等待是否积压，再看缓存GiB和输入/输出速率。低缓存使用率不能推出设备空闲；队列积压也不能唯一归因于KV满。速率是引擎全局计数，不是客户端已交付结果。",
        "",
        "![引擎阶段延迟和iteration直方图](engine-latency-and-batches.png)",
        "",
        "![引擎TTFT和TPOT](engine-token-latency.png)",
        "",
        "prefill时间不是纯kernel时间，TTFT不包含Controller全部等待。iteration指标是否等同每个真实scheduler step尚未验证，不能拿它证明调度token预算跑满。",
        "",
        "| 全窗口指标 | Native | Controller |",
        "|---|---:|---:|",
    ]
    for metric, title in [
        ("request_queue_time_seconds", "引擎平均queue time，秒"),
        ("request_prefill_time_seconds", "引擎平均prefill time，秒"),
        ("request_decode_time_seconds", "引擎平均decode time，秒"),
        ("time_to_first_token_seconds", "引擎TTFT均值，秒"),
        ("request_time_per_output_token_seconds", "引擎TPOT均值，秒/token"),
    ]:
        report.append(
            row(
                title,
                total_n["histograms"][metric]["mean"],
                total_c["histograms"][metric]["mean"],
            )
        )
    report += [
        "",
        "分段精确采样边界、直方图sum/count/buckets和全部统计见[data.json](data.json)，各层数字见[summary.csv](summary.csv)。",
        "",
        "## 3. Program：进展与分布",
        "",
        "![全部Agent进展](program.png)",
        "",
        "同一编号Agent使用同一热图色标，保留零回复Agent。模型调用内的时间包含准入/排队/计算/传输，不等于设备利用率。工具时间只统计工具事件覆盖区间。",
        "",
        "## 4. Request：交付、平均与长尾",
        "",
        "![客户端返回和延迟](request.png)",
        "",
        "![分段平均及p95](request-latency-by-interval.png)",
        "",
        "| 全窗口正常返回调用 | Native | Controller |",
        "|---|---:|---:|",
        row(
            "平均端到端延迟，秒",
            total_n["client"]["mean_latency_s"],
            total_c["client"]["mean_latency_s"],
        ),
        row(
            "p95端到端延迟，秒",
            total_n["client"]["p95_latency_s"],
            total_c["client"]["p95_latency_s"],
        ),
        "",
        "各组后续生成轨迹可能不同；更长回复会增加输出tokens，不等于相同工作量加速。平均值更好不能替代长尾和公平性。",
        "",
        "## 跨层核对与缺失",
        "",
    ]
    for arm, d in data.items():
        h = d["phases"][full]["histograms"]["e2e_request_latency_seconds"]
        if h["available"]:
            normal = d["phases"][full]["client"]["returns"]
            report.append(
                f"- {arm}：引擎完成延迟直方图新增{h['count']:g}个观察，客户端正常返回{normal}次，差{h['count'] - normal:g}；不同边界和未关联集合不能直接相减当作丢失请求。"
            )
    report += (
        [""]
        + ["- " + text for text in missing]
        + [
            "",
            "## 总结",
            "",
            "本报告把交付结果、程序进展、调度行为和引擎处理分开呈现，再用同一时间窗口关联。能够观察到的差异保留，负向结果不筛除；硬件极限、未关联请求及机制净因果仍需相应证据。报告生成本身不授权重跑。",
            "",
            "## 复算来源",
            "",
            "[源文件SHA](source-sha256.json)记录读取原件；[数值表](summary.csv)、[引擎及Controller逐样本](engine-controller-samples.csv)、[正常返回调用](normal-client-returns.csv)均由脚本生成，不含题面、工具输出或服务地址。",
            "客户端CSV的t/start/latency单位是秒；input/output单位是tokens；program为归档任务编号。",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(report))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run folder containing terminal-extract and optional arm settings",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New derived report directory, outside terminal-extract",
    )
    parser.add_argument("--window-seconds", type=float, required=True)
    parser.add_argument(
        "--phase-boundaries",
        type=float,
        nargs=2,
        help="Two internal boundaries; defaults to 1/4 and 1/2 of window",
    )
    parser.add_argument("--rate-window-seconds", type=float, default=10.0)
    parser.add_argument(
        "--question",
        default="Controller相较Native如何改变引擎处理、调用交付、长尾和Agent进展？",
        help="Research question printed in the generated report",
    )
    args = parser.parse_args(argv)
    window = args.window_seconds
    boundaries = args.phase_boundaries or [window / 4, window / 2]
    if (
        not math.isfinite(window)
        or window <= 0
        or (not 0 < boundaries[0] < boundaries[1] < window)
    ):
        parser.error("Require 0 < phase boundary 1 < boundary 2 < finite window")
    if not 0 < args.rate_window_seconds <= window:
        parser.error(
            "Rate window must be positive and no longer than observation window"
        )
    run = args.run_dir.resolve()
    output = (
        args.output_dir or run / "derived-reports" / f"layered-{window:g}s"
    ).resolve()
    if (
        output == run
        or output.is_relative_to(run / "terminal-extract")
        or run.is_relative_to(output)
    ):
        parser.error(
            "Output must not overwrite the run root, input archive tree, or its parent"
        )
    manifest_path = output / "report-manifest.json"
    if output.exists() and any(output.iterdir()) and (not manifest_path.exists()):
        parser.error(
            "Output is nonempty and not owned by this report generator; choose a new directory"
        )
    data, sources = load_run(
        run, window, list(zip([0, *boundaries], [*boundaries, window])) + [(0, window)]
    )
    output.mkdir(parents=True, exist_ok=True)
    summary, fractions = render(
        data, output, window, boundaries, args.rate_window_seconds
    )
    write_report(
        data,
        output,
        window,
        boundaries,
        sources,
        summary,
        fractions,
        args.rate_window_seconds,
        args.question,
    )
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir())
        if p.is_file() and p != manifest_path
    }
    manifest = {
        "schema_version": 1,
        "generator": "benchmarks/report_agent_comparison.py",
        "research_question": args.question,
        "window_seconds": window,
        "phase_boundaries_seconds": boundaries,
        "rate_window_seconds": args.rate_window_seconds,
        "generator_sources_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path(__file__).with_name("agent_report_data.py"),
                Path(__file__).with_name("plot_run.py"),
            ]
        },
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "report": str(output / "report.md"),
                "figures": len(list(output.glob("*.png"))),
                "agents_per_arm": data["native"]["agent_count"],
                "source_files": len(sources),
                "new_inference_requests": 0,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EvidenceError, FileNotFoundError) as exc:
        print(f"Evidence error: {exc}", file=sys.stderr)
        raise SystemExit(2)
