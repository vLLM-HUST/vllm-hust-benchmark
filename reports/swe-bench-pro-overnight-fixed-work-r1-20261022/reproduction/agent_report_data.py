"""Read-only adapter for paired Agent archives; no serving/runtime imports."""

from __future__ import annotations

import collections
import hashlib
import json
import math
import re
from pathlib import Path

HISTOGRAMS = (
    "request_queue_time_seconds",
    "request_prefill_time_seconds",
    "request_decode_time_seconds",
    "e2e_request_latency_seconds",
    "time_to_first_token_seconds",
    "inter_token_latency_seconds",
    "request_time_per_output_token_seconds",
    "request_prefill_kv_computed_tokens",
    "iteration_tokens_total",
)
COUNTERS = (
    "prompt_tokens_total",
    "prompt_tokens_cached_total",
    "prompt_tokens_by_source_total:local_compute",
    "prompt_tokens_by_source_total:local_cache_hit",
    "generation_tokens_total",
    "num_preemptions_total",
    "prefix_cache_queries_total",
    "prefix_cache_hits_total",
)
GAUGES = (
    "num_requests_running",
    "num_requests_waiting",
    "num_requests_waiting_by_reason:capacity",
    "num_requests_waiting_by_reason:deferred",
    "kv_cache_usage_perc",
)
LINE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*?)\})?\s+([-+0-9.eE]+|NaN|[+-]?Inf)(?:\s|$)"
)
LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


class EvidenceError(ValueError):
    """Evidence cannot support the requested statistic; do not silently repair it."""


def parse_metrics(text):
    values, buckets, configs = {}, {}, []
    for line in text.splitlines():
        match = LINE.match(line)
        if not match:
            continue
        name, labels, number = match.groups()
        labels, value = dict(LABEL.findall(labels or "")), float(number)
        if not math.isfinite(value):
            continue
        if name.endswith("_bucket"):
            bound = float(labels["le"])
            target = buckets.setdefault(name[:-7], {})
            target[bound] = target.get(bound, 0) + value
        else:
            values[name] = values.get(name, 0) + value
        for tag in ("source", "reason"):
            if tag in labels:
                values[name + ":" + labels[tag]] = value
        if name == "vllm:cache_config_info":
            configs.append(labels)
    if len(configs) > 1:
        raise EvidenceError(
            "This adapter supports one TP backend, not a sum of independent engines"
        )
    return values, buckets, configs[0] if configs else {}


def counter_delta(first, last, name):
    key = "vllm:" + name
    if key not in first["m"] or key not in last["m"]:
        return None
    delta = last["m"][key] - first["m"][key]
    if delta < -1e-6:
        raise EvidenceError(f"Counter reset inside interval: {name}")
    return max(0.0, delta)


def histogram_delta(first, last, name):
    key = "vllm:" + name
    count = counter_delta(first, last, name + "_count")
    total = counter_delta(first, last, name + "_sum")
    if count is None or total is None or key not in first["b"] or key not in last["b"]:
        return {
            "available": False,
            "count": None,
            "sum": None,
            "mean": None,
            "buckets": [],
            "p50_bucket_bounds": None,
            "p95_bucket_bounds": None,
        }
    if set(first["b"][key]) != set(last["b"][key]):
        raise EvidenceError(f"Histogram bucket layout changed: {name}")
    bins = {le: last["b"][key][le] - v for le, v in first["b"][key].items()}
    if any(v < -1e-6 for v in bins.values()):
        raise EvidenceError(f"Histogram reset: {name}")
    ordered = sorted(bins.items())
    if math.inf not in bins or abs(bins[math.inf] - count) > 1e-6:
        raise EvidenceError(f"Histogram +Inf/count mismatch: {name}")
    if any(b[1] < a[1] for a, b in zip(ordered, ordered[1:])):
        raise EvidenceError(f"Histogram buckets not cumulative: {name}")

    def bounds(q):
        low = 0.0
        for high, n in ordered:
            if n >= count * q:
                return [low, high if math.isfinite(high) else None]
            low = high
        return None

    return {
        "available": True,
        "count": count,
        "sum": total,
        "mean": total / count if count else None,
        "p50_bucket_bounds": bounds(0.5) if count else None,
        "p95_bucket_bounds": bounds(0.95) if count else None,
        "buckets": [[le if math.isfinite(le) else "Inf", v] for le, v in ordered],
    }


def time_weighted_mean(rows, key, lo, hi):
    """Left-hold only within observed adjacent samples; never extrapolate."""
    total = duration = 0.0
    for first, last in zip(rows, rows[1:]):
        dt = max(0.0, min(hi, last["t"]) - max(lo, first["t"]))
        value = first["m"].get("vllm:" + key)
        if dt and value is not None:
            total += dt * value
            duration += dt
    return total / duration if duration else None


def clipped_duration(segments, window):
    return sum(max(0.0, min(end, window) - max(start, 0)) for start, end in segments)


def query_records(events, origin, window):
    """Censor at the window boundary; retain normal, failed and pending counts."""
    starts, tool_starts, queries, tools, replies = {}, {}, [], [], []
    pending = errors = started = 0
    for event in events:
        t = event["monotonic"] - origin
        kind = event["event"]
        if kind == "agent_query_start":
            if event["request_id"] in starts:
                raise EvidenceError("Duplicate query start ID")
            starts[event["request_id"]] = t
            started += 0 <= t < window
        elif kind in ("agent_query_return", "agent_query_exception"):
            start = starts.pop(event["request_id"], None)
            if start is None or t < start:
                raise EvidenceError("Query end has missing/invalid start")
            queries.append((start, min(t, window)))
            pending += 0 <= start < window <= t
            if 0 <= t < window:
                if kind == "agent_query_return":
                    usage = event["usage"]
                    replies.append(
                        {
                            "t": t,
                            "start": start,
                            "latency": t - start,
                            "input": usage["prompt_tokens"],
                            "output": usage["completion_tokens"],
                        }
                    )
                else:
                    errors += 1
        elif kind == "tool_start":
            tool_starts[event["call"]] = t
        elif kind in ("tool_return", "tool_exception"):
            start = tool_starts.pop(event["call"], None)
            if start is None or t < start:
                raise EvidenceError("Tool end has missing/invalid start")
            tools.append((start, min(t, window)))
    for start in starts.values():
        queries.append((start, window))
        pending += 0 <= start < window
    tools.extend((start, window) for start in tool_starts.values())
    return {
        "returns": replies,
        "query_segments": queries,
        "tool_segments": tools,
        "started": started,
        "pending_at_window_end": pending,
        "errors_in_window": errors,
    }


def load_run(run_dir, window, phases):
    import numpy as np

    run_dir = Path(run_dir).resolve()
    root = run_dir / "terminal-extract"
    sources, data = {}, {}

    def source(path):
        sources[str(path.relative_to(run_dir))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        return path

    def read_lines(path):
        return [
            json.loads(line) for line in source(path).read_text().splitlines() if line
        ]

    configs = {}
    for arm in ("native", "controller"):
        batch = root / ("pair-1-" + arm)
        paths = sorted(batch.glob("task-*/arrival.json"))
        if not paths:
            raise EvidenceError(f"No actual Agent arrivals: {batch}")
        arrivals = [json.loads(source(p).read_text()) for p in paths]
        origin = min(x["actual_start_monotonic"] for x in arrivals)
        epoch = min(x["epoch"] for x in arrivals)
        observations = read_lines(root / (arm + "-evidence/observations-private.jsonl"))
        rows, cache_sizes = [], set()
        for observation in observations:
            t = observation["monotonic"] - origin
            if not 0 <= t < window or not isinstance(observation.get("engine"), str):
                continue
            values, buckets, config = parse_metrics(observation["engine"])
            if config.get("kv_cache_memory_bytes") is not None:
                cache_sizes.add(int(config["kv_cache_memory_bytes"]) / 1024**3)
            rows.append(
                {
                    "t": t,
                    "m": values,
                    "b": buckets,
                    "scheduler": observation.get("scheduler"),
                    "programs": observation.get("programs"),
                }
            )
        if len(rows) < 2 or any(b["t"] <= a["t"] for a, b in zip(rows, rows[1:])):
            raise EvidenceError(f"{arm}: need >=2 strictly increasing engine samples")
        if len(cache_sizes) > 1:
            raise EvidenceError(f"{arm}: cache allocation changed inside measurement")
        cache_gib = next(iter(cache_sizes), None)
        event_paths = sorted(batch.glob("task-*/agent-events-private.jsonl"))
        if {p.parent for p in paths} != {p.parent for p in event_paths}:
            raise EvidenceError(f"{arm}: some Agent event files are missing")
        programs, returns = [], []
        for path in event_paths:
            program = query_records(read_lines(path), origin, window)
            program["name"] = path.parent.name
            for reply in program["returns"]:
                reply["program"] = program["name"]
                returns.append(reply)
            programs.append(program)
        settings_path = run_dir / (arm + "-settings-private.json")
        settings = (
            json.loads(source(settings_path).read_text())
            if settings_path.exists()
            else {}
        )
        # Only non-payload experimental settings are copied to the report.
        expected = settings.get("batch", {}).get("program_instances")
        if expected is not None and expected != len(programs):
            raise EvidenceError(
                f"{arm}: actual Agent files differ from configured batch size"
            )
        configs[arm] = {
            k: settings.get("agent", {}).get(k)
            for k in (
                "step_limit",
                "wall_budget_seconds",
                "max_tokens",
                "tool_timeout_seconds",
            )
        }
        phase_data = {}
        for lo, hi in phases:
            sample = [x for x in rows if lo <= x["t"] < hi]
            if len(sample) < 2:
                raise EvidenceError(f"{arm}: insufficient samples in {lo}-{hi}")
            first, last = sample[0], sample[-1]
            duration = last["t"] - first["t"]
            normal = [x for x in returns if lo <= x["t"] < hi]
            counters = {key: counter_delta(first, last, key) for key in COUNTERS}
            triple = [
                counters[k]
                for k in (
                    "prompt_tokens_total",
                    "prompt_tokens_by_source_total:local_compute",
                    "prompt_tokens_by_source_total:local_cache_hit",
                )
            ]
            if (
                all(v is not None for v in triple)
                and abs(triple[0] - triple[1] - triple[2]) > 1
            ):
                # A non-local source requires a different breakdown; do not mislabel its tokens.
                raise EvidenceError(
                    "Prompt tokens are not local_compute + local_cache_hit; extend adapter for other sources"
                )
            states = collections.defaultdict(float)
            state_coverage = 0.0
            for a, b in zip(rows, rows[1:]):
                dt = max(0.0, min(hi, b["t"]) - max(lo, a["t"]))
                if a["programs"] is not None:
                    state_coverage += dt
                    for state, n in collections.Counter(
                        p.get("state", "unknown") for p in a["programs"].values()
                    ).items():
                        states[state] += dt * n

            def stat(field, quantile=None):
                values = [x[field] for x in normal]
                return (
                    float(
                        np.quantile(values, quantile)
                        if quantile is not None
                        else np.mean(values)
                    )
                    if values
                    else None
                )

            phase_data[f"{lo:g}-{hi:g}"] = {
                "engine_sample_start_s": first["t"],
                "engine_sample_end_s": last["t"],
                "engine_sample_duration_s": duration,
                "engine_samples": len(sample),
                "engine_time_weighted_means": {
                    k: time_weighted_mean(rows, k, lo, hi) for k in GAUGES
                },
                "engine_counter_deltas": counters,
                "engine_counter_rates_per_second": {
                    k: v / duration if v is not None else None
                    for k, v in counters.items()
                },
                "histograms": {k: histogram_delta(first, last, k) for k in HISTOGRAMS},
                "client": {
                    "returns": len(normal),
                    "output_tokens": sum(x["output"] for x in normal),
                    "mean_latency_s": stat("latency"),
                    "p50_latency_s": stat("latency", 0.5),
                    "p95_latency_s": stat("latency", 0.95),
                    "mean_input_tokens": stat("input"),
                },
                "mean_program_states": {
                    k: v / state_coverage for k, v in states.items()
                }
                if state_coverage
                else None,
                "program_state_coverage_s": state_coverage,
            }
        action_path = root / (arm + "-evidence/action-log-observations-private.jsonl")
        actions = read_lines(action_path) if action_path.exists() else None
        data[arm] = {
            "run_name": run_dir.name,
            "origin_epoch": epoch,
            "rows": rows,
            "returns": returns,
            "programs": programs,
            "phases": phase_data,
            "cache_gib_per_rank": cache_gib,
            "actions": actions,
            "settings": configs[arm],
            "agent_count": len(programs),
            "pending": sum(p["pending_at_window_end"] for p in programs),
            "errors": sum(p["errors_in_window"] for p in programs),
        }
    if data["native"]["agent_count"] != data["controller"]["agent_count"]:
        raise EvidenceError("The two arms have different actual Agent counts")
    if configs["native"] != configs["controller"]:
        raise EvidenceError("Agent budget settings differ between arms")
    if data["native"]["cache_gib_per_rank"] != data["controller"]["cache_gib_per_rank"]:
        raise EvidenceError("Cache-pool capacities differ between arms")
    return data, sources
