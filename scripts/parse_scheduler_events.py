#!/usr/bin/env python3
"""Parse vLLM server logs for scheduler, KV cache, and preemption events.

This module extracts structured data from vLLM server stdout logs to support
issue #134 (KV capacity scan and tiering state machine analysis).

Log patterns parsed:
  - KV cache info at startup (size, blocks, max concurrency)
  - Memory breakdown (weights, activation, non-torch, graph, KV)
  - Periodic engine stats (running/waiting queue, KV usage %, prefix hit rate)
  - Preemption events with timestamps
  - Utility victim selection events (BidKV)
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Regex patterns for vLLM server log parsing
# ---------------------------------------------------------------------------

# Startup KV cache info
_KV_CACHE_MEMORY_RE = re.compile(
    r"Available KV cache memory:\s*([\d.]+)\s*GiB", re.IGNORECASE
)
_KV_CACHE_TOKENS_RE = re.compile(
    r"GPU KV cache size:\s*([\d,]+)\s*tokens", re.IGNORECASE
)
_MAX_CONCURRENCY_RE = re.compile(
    r"Maximum concurrency for\s*[\d,]+\s*tokens per request:\s*([\d.]+)x",
    re.IGNORECASE,
)
_KV_BLOCK_SIZE_RE = re.compile(r"Setting kv cache block size to\s*(\d+)", re.IGNORECASE)
_CURRENT_KV_MEMORY_RE = re.compile(
    r"Current KV cache memory:\s*([\d.]+)\s*GiB", re.IGNORECASE
)

# Memory breakdown
_FREE_MEMORY_RE = re.compile(
    r"Free memory on device \(([\d.]+)/([\d.]+)\s*GiB\)", re.IGNORECASE
)
_DESIRED_UTIL_RE = re.compile(
    r"Desired GPU memory utilization is \(([\d.]+),\s*([\d.]+)\s*GiB\)",
    re.IGNORECASE,
)
_WEIGHTS_MEM_RE = re.compile(r"([\d.]+)\s*GiB for weights", re.IGNORECASE)
_ACTIVATION_MEM_RE = re.compile(r"([\d.]+)\s*GiB for peak activation", re.IGNORECASE)
_NON_TORCH_MEM_RE = re.compile(r"([\d.]+)\s*GiB for non-torch memory", re.IGNORECASE)
_GRAPH_MEM_RE = re.compile(r"([\d.]+)\s*GiB for NPU graph memory", re.IGNORECASE)

# Periodic engine stats
_ENGINE_STATS_RE = re.compile(
    r"Engine \d+:\s*"
    r"Avg prompt throughput:\s*([\d.]+)\s*tokens/s,\s*"
    r"Avg generation throughput:\s*([\d.]+)\s*tokens/s,\s*"
    r"Running:\s*(\d+)\s*reqs,\s*"
    r"Waiting:\s*(\d+)\s*reqs,\s*"
    r"GPU KV cache usage:\s*([\d.]+)%,\s*"
    r"Prefix cache hit rate:\s*([\d.]+)%",
    re.IGNORECASE,
)

# Timestamp from log line prefix: "INFO 07-26 15:50:29 [loggers.py:282]"
_LOG_TS_RE = re.compile(r"(\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})")

# Preemption events
_PREEMPT_RE = re.compile(
    r"(preempt|preempting|preempted|PreemptionCount)\s*[:=]?\s*(\d+)",
    re.IGNORECASE,
)
_PREEMPT_MSG_RE = re.compile(
    r"(Sequence group|seq_group)\s+(\d+)\s+is\s+(preempted|preempting)",
    re.IGNORECASE,
)

# Utility victim selection events
_VICTIM_SELECT_RE = re.compile(
    r"(victim|VictimSelector|utility.*select|pick_victim|UnifiedVictimSelector)"
    r"[:\s].*",
    re.IGNORECASE,
)

# Suggested kv-cache-memory values
_SUGGESTED_KV_RE = re.compile(
    r"--kv-cache-memory=(\d+)\s*\(([\d.]+)\s*GiB\)", re.IGNORECASE
)


def _parse_log_timestamp(line: str, year: int = 2026) -> str | None:
    """Extract ISO-8601 timestamp from a vLLM log line.

    vLLM logs use ``MM-DD HH:MM:SS`` format (no year). We prepend ``year``
    (default 2026, the benchmark year) to produce ``YYYY-MM-DDTHH:MM:SS``.
    """
    m = _LOG_TS_RE.search(line)
    if not m:
        return None
    ts_str = m.group(1)
    try:
        dt = datetime.strptime(f"{year} {ts_str}", "%Y %m-%d %H:%M:%S")
        return dt.isoformat()
    except ValueError:
        return None


def parse_kv_cache_info(log_text: str) -> dict[str, Any]:
    """Extract KV cache configuration from server startup logs.

    Returns a dict with keys:
      - ``kv_cache_memory_gib``: float | None
      - ``kv_cache_tokens``: int | None
      - ``max_concurrency``: float | None
      - ``kv_block_size``: int | None
      - ``current_kv_memory_gib``: float | None
      - ``suggested_kv_memory_bytes``: list[dict] (from vLLM suggestions)
    """
    info: dict[str, Any] = {
        "kv_cache_memory_gib": None,
        "kv_cache_tokens": None,
        "max_concurrency": None,
        "kv_block_size": None,
        "current_kv_memory_gib": None,
        "suggested_kv_memory_bytes": [],
    }

    if m := _KV_CACHE_MEMORY_RE.search(log_text):
        info["kv_cache_memory_gib"] = float(m.group(1))

    if m := _KV_CACHE_TOKENS_RE.search(log_text):
        info["kv_cache_tokens"] = int(m.group(1).replace(",", ""))

    if m := _MAX_CONCURRENCY_RE.search(log_text):
        info["max_concurrency"] = float(m.group(1))

    if m := _KV_BLOCK_SIZE_RE.search(log_text):
        info["kv_block_size"] = int(m.group(1))

    if m := _CURRENT_KV_MEMORY_RE.search(log_text):
        info["current_kv_memory_gib"] = float(m.group(1))

    for m in _SUGGESTED_KV_RE.finditer(log_text):
        info["suggested_kv_memory_bytes"].append(
            {
                "bytes": int(m.group(1)),
                "gib": float(m.group(2)),
            }
        )

    return info


def parse_memory_breakdown(log_text: str) -> dict[str, Any]:
    """Extract memory breakdown from server startup logs.

    Returns a dict with keys:
      - ``free_memory_gib``: float | None
      - ``total_memory_gib``: float | None
      - ``desired_utilization``: float | None
      - ``desired_memory_gib``: float | None
      - ``weights_gib``: float | None
      - ``activation_gib``: float | None
      - ``non_torch_gib``: float | None
      - ``graph_memory_gib``: float | None
    """
    breakdown: dict[str, Any] = {
        "free_memory_gib": None,
        "total_memory_gib": None,
        "desired_utilization": None,
        "desired_memory_gib": None,
        "weights_gib": None,
        "activation_gib": None,
        "non_torch_gib": None,
        "graph_memory_gib": None,
    }

    if m := _FREE_MEMORY_RE.search(log_text):
        breakdown["free_memory_gib"] = float(m.group(1))
        breakdown["total_memory_gib"] = float(m.group(2))

    if m := _DESIRED_UTIL_RE.search(log_text):
        breakdown["desired_utilization"] = float(m.group(1))
        breakdown["desired_memory_gib"] = float(m.group(2))

    if m := _WEIGHTS_MEM_RE.search(log_text):
        breakdown["weights_gib"] = float(m.group(1))

    if m := _ACTIVATION_MEM_RE.search(log_text):
        breakdown["activation_gib"] = float(m.group(1))

    if m := _NON_TORCH_MEM_RE.search(log_text):
        breakdown["non_torch_gib"] = float(m.group(1))

    if m := _GRAPH_MEM_RE.search(log_text):
        breakdown["graph_memory_gib"] = float(m.group(1))

    return breakdown


def parse_engine_stats(log_text: str) -> list[dict[str, Any]]:
    """Extract periodic engine stats from server logs.

    Each entry contains:
      - ``timestamp``: ISO-8601 string
      - ``prompt_throughput``: float (tokens/s)
      - ``generation_throughput``: float (tokens/s)
      - ``running_reqs``: int
      - ``waiting_reqs``: int
      - ``kv_cache_usage_pct``: float
      - ``prefix_cache_hit_rate_pct``: float
    """
    stats: list[dict[str, Any]] = []
    for line in log_text.splitlines():
        if m := _ENGINE_STATS_RE.search(line):
            ts = _parse_log_timestamp(line)
            stats.append(
                {
                    "timestamp": ts,
                    "prompt_throughput": float(m.group(1)),
                    "generation_throughput": float(m.group(2)),
                    "running_reqs": int(m.group(3)),
                    "waiting_reqs": int(m.group(4)),
                    "kv_cache_usage_pct": float(m.group(5)),
                    "prefix_cache_hit_rate_pct": float(m.group(6)),
                }
            )
    return stats


def parse_preemption_events(log_text: str) -> list[dict[str, Any]]:
    """Extract preemption events from server logs.

    Returns a list of dicts with:
      - ``timestamp``: ISO-8601 string | None
      - ``seq_group_id``: int | None
      - ``event_type``: str (e.g. "preempted", "preempting")
    """
    events: list[dict[str, Any]] = []
    for line in log_text.splitlines():
        if m := _PREEMPT_MSG_RE.search(line):
            ts = _parse_log_timestamp(line)
            events.append(
                {
                    "timestamp": ts,
                    "seq_group_id": int(m.group(2)),
                    "event_type": m.group(3).lower(),
                }
            )
    return events


def parse_victim_selection_events(log_text: str) -> list[dict[str, Any]]:
    """Extract utility victim selection (BidKV) events from server logs.

    Returns a list of dicts with:
      - ``timestamp``: ISO-8601 string | None
      - ``raw_line``: str (the matched log line)
    """
    events: list[dict[str, Any]] = []
    for line in log_text.splitlines():
        if _VICTIM_SELECT_RE.search(line):
            ts = _parse_log_timestamp(line)
            events.append({"timestamp": ts, "raw_line": line.strip()})
    return events


def parse_server_log(log_path: str | Path) -> dict[str, Any]:
    """Parse a complete vLLM server log file.

    Returns a dict with all parsed sections:
      - ``kv_cache_info``: dict from parse_kv_cache_info
      - ``memory_breakdown``: dict from parse_memory_breakdown
      - ``engine_stats``: list from parse_engine_stats
      - ``preemption_events``: list from parse_preemption_events
      - ``victim_selection_events``: list from parse_victim_selection_events
      - ``log_file``: str (path to the parsed log)
    """
    log_path = Path(log_path)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")

    return {
        "log_file": str(log_path),
        "kv_cache_info": parse_kv_cache_info(log_text),
        "memory_breakdown": parse_memory_breakdown(log_text),
        "engine_stats": parse_engine_stats(log_text),
        "preemption_events": parse_preemption_events(log_text),
        "victim_selection_events": parse_victim_selection_events(log_text),
    }


def reconstruct_preempt_timeline(
    preemption_events: list[dict[str, Any]],
    engine_stats: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reconstruct a preempt-to-admission timeline from parsed events.

    This approximates the ``preempt -> restore_start -> restore_done ->
    scheduler_wakeup -> admission -> first_prefill/decode`` chain described in
    issue #134 by correlating preemption events with engine stats snapshots.

    Returns a dict with:
      - ``total_preemptions``: int
      - ``pressure_episodes``: list of dicts, each containing:
          - ``preempt_timestamp``: str | None
          - ``preempt_seq_group_id``: int | None
          - ``restore_approx_timestamp``: str | None (first stat with waiting=0
            after preemption)
          - ``restore_to_admission_gap_s``: float | None
          - ``peak_waiting_reqs``: int
          - ``peak_kv_usage_pct``: float
      - ``summary``: dict with aggregate stats
    """
    if not preemption_events:
        return {
            "total_preemptions": 0,
            "pressure_episodes": [],
            "summary": {
                "total_preemptions": 0,
                "total_restore_gaps_s": [],
                "max_waiting_reqs": 0,
                "max_kv_usage_pct": 0.0,
            },
        }

    episodes: list[dict[str, Any]] = []
    stats_sorted = sorted(engine_stats, key=lambda s: s.get("timestamp") or "")

    # Window before preemption to capture pressure buildup (seconds)
    PRESSURE_WINDOW_S = 60

    for pe in preemption_events:
        pe_ts = pe.get("timestamp")
        pe_sgid = pe.get("seq_group_id")

        # Find stats around this preemption (within window before + all after)
        if pe_ts:
            try:
                pe_dt = datetime.fromisoformat(pe_ts)
                window_start = (
                    pe_dt - timedelta(seconds=PRESSURE_WINDOW_S)
                ).isoformat()
                episode_stats = [
                    s
                    for s in stats_sorted
                    if (s.get("timestamp") or "") >= window_start
                ]
            except (ValueError, TypeError):
                episode_stats = list(stats_sorted)
            post_stats = [
                s for s in stats_sorted if (s.get("timestamp") or "") >= pe_ts
            ]
        else:
            episode_stats = list(stats_sorted)
            post_stats = list(stats_sorted)

        # Peak waiting/usage around the pressure episode
        peak_waiting = max((s["waiting_reqs"] for s in episode_stats), default=0)
        peak_kv = max((s["kv_cache_usage_pct"] for s in episode_stats), default=0.0)

        # Approximate restore = first stat after preemption where waiting drops to 0
        restore_ts = None
        restore_to_admission_gap_s = None
        for s in post_stats:
            if s["waiting_reqs"] == 0:
                restore_ts = s.get("timestamp")
                if restore_ts and pe_ts:
                    try:
                        t1 = datetime.fromisoformat(pe_ts)
                        t2 = datetime.fromisoformat(restore_ts)
                        restore_to_admission_gap_s = (t2 - t1).total_seconds()
                    except (ValueError, TypeError):
                        pass
                break

        episodes.append(
            {
                "preempt_timestamp": pe_ts,
                "preempt_seq_group_id": pe_sgid,
                "restore_approx_timestamp": restore_ts,
                "restore_to_admission_gap_s": restore_to_admission_gap_s,
                "peak_waiting_reqs": peak_waiting,
                "peak_kv_usage_pct": peak_kv,
            }
        )

    restore_gaps = [
        e["restore_to_admission_gap_s"]
        for e in episodes
        if e["restore_to_admission_gap_s"] is not None
    ]

    return {
        "total_preemptions": len(preemption_events),
        "pressure_episodes": episodes,
        "summary": {
            "total_preemptions": len(preemption_events),
            "total_restore_gaps_s": restore_gaps,
            "max_waiting_reqs": max(
                (e["peak_waiting_reqs"] for e in episodes), default=0
            ),
            "max_kv_usage_pct": max(
                (e["peak_kv_usage_pct"] for e in episodes), default=0.0
            ),
        },
    }


def main() -> None:
    """CLI entry point: parse a server log and print structured output."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse vLLM server log for scheduler/KV/preemption events."
    )
    parser.add_argument("log_file", help="Path to vLLM server stdout log")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output JSON file (default: stdout)",
    )
    args = parser.parse_args()

    result = parse_server_log(args.log_file)
    timeline = reconstruct_preempt_timeline(
        result["preemption_events"], result["engine_stats"]
    )
    result["preempt_timeline"] = timeline

    output = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
