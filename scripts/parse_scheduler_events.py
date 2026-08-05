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
  - 6-stage preempt timeline events (restore, scheduler wakeup, admission, etc.)
  - CPU offload / tiering connector events

PR #146 review fixes:
  - ``TIMELINE_STAGES`` defines the complete 6-stage event chain.
  - ``parse_stage_events`` extracts per-stage event lists with timestamps and
    seq_group_id correlation.
  - ``reconstruct_preempt_timeline`` now builds a ``stages`` dict per episode,
    sets ``timeline_complete``, and returns ``timeline_status`` instead of
    relying solely on ``total_preemptions > 0``.
  - ``verify_kv_capacity_from_log`` validates actual KV cache memory against
    the target capacity (fail-closed, not just a warning).
  - ``parse_cpu_offload_events`` extracts tiering/offload event patterns.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Timeline stage definitions (6-stage preempt->restore->admission chain)
# ---------------------------------------------------------------------------

TIMELINE_STAGES = [
    "preempt",
    "restore_start",
    "restore_done",
    "scheduler_wakeup",
    "admission",
    "first_prefill_or_decode",
]

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
# Cumulative preemption summary from vLLM UtilityVictim scheduler logs.
# Format: "total_preemptions=N utility_hits=... default_hits=... tokens_freed=..."
# Each increase in the cumulative count represents one new preemption event.
_TOTAL_PREEMPTIONS_RE = re.compile(
    r"total_preemptions\s*[:=]\s*(\d+)",
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

# ---------------------------------------------------------------------------
# 6-stage timeline event patterns
# ---------------------------------------------------------------------------

# Stage 1: preempt (reuses _PREEMPT_MSG_RE above)

# Stage 2: restore_start — KV cache restore begins for a seq_group
# Use "restoring" (ing form) to distinguish from "restored" (ed form).
_RESTORE_START_RE = re.compile(
    r"(restore_start|restoring\s+kv|begin\s+restore|kv\s+restore.*start|"
    r"restoring.*seq_group\s+(\d+)|seq_group\s+(\d+)\s+restore.*start)",
    re.IGNORECASE,
)

# Stage 3: restore_done — KV cache restore completes for a seq_group
# Use "restored" (ed form) to distinguish from "restoring" (ing form).
_RESTORE_DONE_RE = re.compile(
    r"(restore_done|restored\s+kv|finish\s+restore|kv\s+restore.*complete|"
    r"restored.*seq_group\s+(\d+)|restore.*complete|"
    r"seq_group\s+(\d+)\s+restore.*done)",
    re.IGNORECASE,
)

# Stage 4: scheduler_wakeup — scheduler woke up after restore
_SCHEDULER_WAKEUP_RE = re.compile(
    r"(scheduler_wakeup|scheduler.*woke|woke\s+up|scheduler.*wakeup|"
    r"scheduler_loop|scheduler.*resumed)",
    re.IGNORECASE,
)

# Stage 5: admission — seq_group admitted/re-queued after restore
_ADMISSION_RE = re.compile(
    r"(admission|admitted|admitting|seq_group\s+(\d+)\s+(admitted|admitting)|"
    r"admit.*seq_group\s+(\d+))",
    re.IGNORECASE,
)

# Requeue events (related to admission)
_REQUEUE_RE = re.compile(
    r"(requeue|re-queue|requeued|re-queued)",
    re.IGNORECASE,
)

# Stage 6: first_prefill_or_decode — first prefill or decode after admission
_FIRST_PREFILL_RE = re.compile(
    r"(first_prefill|first\s+prefill|prefill.*seq_group\s+(\d+)|"
    r"seq_group\s+(\d+)\s+prefill)",
    re.IGNORECASE,
)
_FIRST_DECODE_RE = re.compile(
    r"(first_decode|first\s+decode|decode.*seq_group\s+(\d+)|"
    r"seq_group\s+(\d+)\s+decode)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# CPU offload / tiering connector patterns
# ---------------------------------------------------------------------------

_CPU_OFFLOAD_RE = re.compile(
    r"(cpu_offload|CPUOffloading|CPUOffloadingConnector|"
    r"offload.*kv|kv.*offload|cpu_bytes_to_use|"
    r"load.*from.*cpu|save.*to.*cpu|connector.*cpu|"
    r"kv_connector|kv_transfer)",
    re.IGNORECASE,
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


def _extract_seq_group_id(line: str) -> int | None:
    """Try to extract a seq_group_id from a log line."""
    # Try the standard "Sequence group N" / "seq_group N" pattern
    m = re.search(r"(?:Sequence group|seq_group)\s+(\d+)", line, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Try generic "seq_group_id=N" pattern
    m = re.search(r"seq_group_id\s*[=:]\s*(\d+)", line, re.IGNORECASE)
    if m:
        return int(m.group(1))
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

    Supports two log formats:
      1. Individual preempt messages: "Sequence group N is preempted"
      2. Cumulative summary lines: "total_preemptions=N ..." (vLLM
         UtilityVictim scheduler). Each increase in the cumulative count
         generates one synthetic preempt event with ``seq_group_id=None``
         and ``event_type="preempted"``.
    """
    events: list[dict[str, Any]] = []
    prev_total = 0
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
        elif m := _TOTAL_PREEMPTIONS_RE.search(line):
            current_total = int(m.group(1))
            if current_total > prev_total:
                ts = _parse_log_timestamp(line)
                # One synthetic event per new preemption
                for _ in range(current_total - prev_total):
                    events.append(
                        {
                            "timestamp": ts,
                            "seq_group_id": None,
                            "event_type": "preempted",
                        }
                    )
                prev_total = current_total
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


def parse_stage_events(log_text: str) -> dict[str, list[dict[str, Any]]]:
    """Extract per-stage events for the 6-stage preempt timeline.

    Returns a dict mapping each stage name in ``TIMELINE_STAGES`` to a list of
    event dicts. Each event dict contains:
      - ``timestamp``: ISO-8601 string | None
      - ``seq_group_id``: int | None
      - ``raw_line``: str (the matched log line)
    """
    stage_events: dict[str, list[dict[str, Any]]] = {
        stage: [] for stage in TIMELINE_STAGES
    }

    for line in log_text.splitlines():
        stripped = line.strip()
        ts = _parse_log_timestamp(line)
        sgid = _extract_seq_group_id(line)

        # Stage 1: preempt
        if _PREEMPT_MSG_RE.search(line):
            if m := _PREEMPT_MSG_RE.search(line):
                sgid = int(m.group(2))
            stage_events["preempt"].append(
                {"timestamp": ts, "seq_group_id": sgid, "raw_line": stripped}
            )
            continue

        # Stage 2: restore_start
        if _RESTORE_START_RE.search(line):
            stage_events["restore_start"].append(
                {"timestamp": ts, "seq_group_id": sgid, "raw_line": stripped}
            )
            continue

        # Stage 3: restore_done
        if _RESTORE_DONE_RE.search(line):
            stage_events["restore_done"].append(
                {"timestamp": ts, "seq_group_id": sgid, "raw_line": stripped}
            )
            continue

        # Stage 4: scheduler_wakeup
        if _SCHEDULER_WAKEUP_RE.search(line):
            stage_events["scheduler_wakeup"].append(
                {"timestamp": ts, "seq_group_id": sgid, "raw_line": stripped}
            )
            continue

        # Stage 5: admission (also matches requeue)
        if _ADMISSION_RE.search(line) or _REQUEUE_RE.search(line):
            stage_events["admission"].append(
                {"timestamp": ts, "seq_group_id": sgid, "raw_line": stripped}
            )
            continue

        # Stage 6: first_prefill_or_decode
        if _FIRST_PREFILL_RE.search(line) or _FIRST_DECODE_RE.search(line):
            stage_events["first_prefill_or_decode"].append(
                {"timestamp": ts, "seq_group_id": sgid, "raw_line": stripped}
            )
            continue

    return stage_events


def parse_cpu_offload_events(log_text: str) -> list[dict[str, Any]]:
    """Extract CPU offload / tiering connector events from server logs.

    Matches lines mentioning CPUOffloadingConnector, kv_connector,
    kv_transfer, cpu offload load/save, etc.

    Returns a list of dicts with:
      - ``timestamp``: ISO-8601 string | None
      - ``raw_line``: str (the matched log line)
    """
    events: list[dict[str, Any]] = []
    for line in log_text.splitlines():
        if _CPU_OFFLOAD_RE.search(line):
            ts = _parse_log_timestamp(line)
            events.append({"timestamp": ts, "raw_line": line.strip()})
    return events


def validate_timeline_complete(episode: dict[str, Any]) -> bool:
    """Check that all 6 stages in an episode have non-None timestamps.

    Per reviewer round 1 issue 2: the timeline must not only have all stages
    present but also be **monotonically ordered** — ``preempt ≤ restore_start
    ≤ restore_done ≤ scheduler_wakeup ≤ admission ≤ first_prefill_or_decode``.
    Without ordering, events from different episodes could be falsely
    correlated, yielding a false-positive ``timeline_complete``.

    Args:
        episode: dict with a ``stages`` key mapping stage names to timestamps.

    Returns:
        True if all stages in ``TIMELINE_STAGES`` have a non-None value **and**
        the timestamps are monotonically non-decreasing.
    """
    stages = episode.get("stages", {})
    if not isinstance(stages, dict):
        return False
    # Stage 1: all stages must be present
    if not all(stages.get(stage) is not None for stage in TIMELINE_STAGES):
        return False
    # Stage 2: timestamps must be monotonically non-decreasing
    ts_values = [stages[stage] for stage in TIMELINE_STAGES]
    for i in range(1, len(ts_values)):
        if ts_values[i] is not None and ts_values[i - 1] is not None:
            if ts_values[i] < ts_values[i - 1]:
                return False
    return True


def _build_stages_for_episode(
    preempt_event: dict[str, Any],
    stage_events: dict[str, list[dict[str, Any]]],
) -> dict[str, str | None]:
    """Build a stages dict for a single preempt episode.

    Correlates the preempt event with subsequent stage events by seq_group_id
    (preferred) or by timestamp ordering (fallback).
    """
    stages: dict[str, str | None] = {s: None for s in TIMELINE_STAGES}
    stages["preempt"] = preempt_event.get("timestamp")
    pe_sgid = preempt_event.get("seq_group_id")
    pe_ts = preempt_event.get("timestamp")

    for stage in [
        "restore_start",
        "restore_done",
        "scheduler_wakeup",
        "admission",
        "first_prefill_or_decode",
    ]:
        events = stage_events.get(stage, [])
        matched: dict[str, Any] | None = None

        # Strategy 1: match by seq_group_id
        if pe_sgid is not None:
            for ev in events:
                if ev.get("seq_group_id") == pe_sgid:
                    matched = ev
                    break

        # Strategy 2: if exactly one event has timestamp >= preempt timestamp,
        # use it.  Per reviewer round 1 issue 2: when multiple candidates exist
        # without seq_group_id correlation, the match is ambiguous — leaving
        # the stage as None is safer (fail-closed) than risking a false positive.
        if matched is None and pe_ts is not None:
            candidates = [
                ev for ev in events if ev.get("timestamp") and ev["timestamp"] >= pe_ts
            ]
            if len(candidates) == 1:
                matched = candidates[0]

        # Strategy 3: first event of this stage (no correlation possible).
        # Per reviewer round 1 issue 2: this fallback is unreliable for stages
        # whose log lines typically lack seq_group_id (e.g. scheduler_wakeup).
        # Only use it when there is exactly ONE event for this stage — if
        # multiple uncorrelated events exist, leaving the stage as None is
        # safer (fail-closed) than risking a false-positive correlation.
        if matched is None and len(events) == 1:
            matched = events[0]

        if matched:
            stages[stage] = matched.get("timestamp")

    return stages


def reconstruct_preempt_timeline(
    preemption_events: list[dict[str, Any]],
    engine_stats: list[dict[str, Any]],
    stage_events: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Reconstruct a preempt-to-admission timeline from parsed events.

    This builds the ``preempt -> restore_start -> restore_done ->
    scheduler_wakeup -> admission -> first_prefill/decode`` chain described in
    issue #134. If ``stage_events`` is provided, each episode gets a ``stages``
    dict with per-stage timestamps and a ``timeline_complete`` bool.

    Args:
        preemption_events: list of preempt event dicts from
            ``parse_preemption_events``.
        engine_stats: list of engine stat dicts from ``parse_engine_stats``.
        stage_events: optional dict from ``parse_stage_events``. When provided,
            episodes are enriched with the 6-stage chain.

    Returns:
        Dict with:
        - ``total_preemptions``: int
        - ``pressure_episodes``: list of dicts, each containing:
            - ``preempt_timestamp``: str | None
            - ``preempt_seq_group_id``: int | None
            - ``restore_approx_timestamp``: str | None
            - ``restore_to_admission_gap_s``: float | None
            - ``peak_waiting_reqs``: int
            - ``peak_kv_usage_pct``: float
            - ``stages``: dict mapping stage names to timestamps (if stage_events
              provided)
            - ``timeline_complete``: bool (if stage_events provided)
        - ``summary``: dict with aggregate stats
        - ``timeline_status``: "complete" | "incomplete" | "no_preemptions"
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
            "timeline_status": "no_preemptions",
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

        episode: dict[str, Any] = {
            "preempt_timestamp": pe_ts,
            "preempt_seq_group_id": pe_sgid,
            "restore_approx_timestamp": restore_ts,
            "restore_to_admission_gap_s": restore_to_admission_gap_s,
            "peak_waiting_reqs": peak_waiting,
            "peak_kv_usage_pct": peak_kv,
        }

        # Enrich with 6-stage timeline if stage_events provided
        if stage_events is not None:
            stages = _build_stages_for_episode(pe, stage_events)
            episode["stages"] = stages
            episode["timeline_complete"] = validate_timeline_complete(episode)
        else:
            # Without stage_events, we can only mark preempt; timeline is
            # inherently incomplete.
            episode["stages"] = {s: None for s in TIMELINE_STAGES}
            episode["stages"]["preempt"] = pe_ts
            episode["timeline_complete"] = False

        episodes.append(episode)

    restore_gaps = [
        e["restore_to_admission_gap_s"]
        for e in episodes
        if e["restore_to_admission_gap_s"] is not None
    ]

    # Determine overall timeline_status
    any_complete = any(e.get("timeline_complete") for e in episodes)
    if any_complete:
        timeline_status = "complete"
    else:
        timeline_status = "incomplete"

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
        "timeline_status": timeline_status,
    }


def verify_kv_capacity_from_log(
    log_path: str | Path,
    target_kv_gib: float,
    tolerance_gib: float = 2.0,
) -> dict[str, Any]:
    """Parse a server log and verify actual KV cache memory matches target.

    This is a fail-closed verification: if the KV cache memory cannot be found
    in the log, ``within_tolerance`` is False.

    Args:
        log_path: path to the vLLM server stdout log.
        target_kv_gib: expected KV cache memory in GiB.
        tolerance_gib: acceptable difference in GiB (default 2.0).

    Returns:
        Dict with:
        - ``actual_kv_gib``: float | None
        - ``target_kv_gib``: float
        - ``tolerance_gib``: float
        - ``diff_gib``: float | None
        - ``within_tolerance``: bool
        - ``error``: str | None (error message if verification could not run)
    """
    log_path = Path(log_path)
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except (OSError, IOError) as exc:
        return {
            "actual_kv_gib": None,
            "target_kv_gib": target_kv_gib,
            "tolerance_gib": tolerance_gib,
            "diff_gib": None,
            "within_tolerance": False,
            "error": f"cannot read log file: {exc}",
        }

    info = parse_kv_cache_info(log_text)
    actual = info.get("kv_cache_memory_gib")

    if actual is None:
        return {
            "actual_kv_gib": None,
            "target_kv_gib": target_kv_gib,
            "tolerance_gib": tolerance_gib,
            "diff_gib": None,
            "within_tolerance": False,
            "error": "KV cache memory not found in server log",
        }

    diff = abs(actual - target_kv_gib)
    within = diff <= tolerance_gib
    return {
        "actual_kv_gib": actual,
        "target_kv_gib": target_kv_gib,
        "tolerance_gib": tolerance_gib,
        "diff_gib": round(diff, 4),
        "within_tolerance": within,
        "error": None,
    }


def parse_server_log(log_path: str | Path) -> dict[str, Any]:
    """Parse a complete vLLM server log file.

    Returns a dict with all parsed sections:
      - ``kv_cache_info``: dict from parse_kv_cache_info
      - ``memory_breakdown``: dict from parse_memory_breakdown
      - ``engine_stats``: list from parse_engine_stats
      - ``preemption_events``: list from parse_preemption_events
      - ``victim_selection_events``: list from parse_victim_selection_events
      - ``stage_events``: dict from parse_stage_events
      - ``cpu_offload_events``: list from parse_cpu_offload_events
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
        "stage_events": parse_stage_events(log_text),
        "cpu_offload_events": parse_cpu_offload_events(log_text),
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
    parser.add_argument(
        "--verify-kv",
        type=float,
        default=None,
        help="Verify KV cache matches this target (GiB) and exit",
    )
    args = parser.parse_args()

    if args.verify_kv is not None:
        result = verify_kv_capacity_from_log(args.log_file, args.verify_kv)
        output = json.dumps(result, indent=2, ensure_ascii=False)
        print(output)
        return

    result = parse_server_log(args.log_file)
    timeline = reconstruct_preempt_timeline(
        result["preemption_events"],
        result["engine_stats"],
        result["stage_events"],
    )
    result["preempt_timeline"] = timeline

    output = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
