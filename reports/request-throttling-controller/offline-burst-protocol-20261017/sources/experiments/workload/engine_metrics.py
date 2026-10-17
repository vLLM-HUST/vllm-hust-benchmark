"""Sample the engine's own counters once a second while a replay runs.

vLLM exposes Prometheus metrics at GET /metrics (not under /v1). A background
thread reads it every `interval_seconds` and appends one line to metrics.jsonl:

    {"t": 12.0, "engine": "http://127.0.0.1:8100/v1",
     "vllm:kv_cache_usage_perc": 0.41, "vllm:num_requests_running": 7, ...}

one line per engine per sample when several engines sit behind a router.

`t` is seconds since the replay started, on the same clock as the request
records, so engine state and request latencies line up on one time axis.

What each sampled metric means (names and meanings from vLLM's metrics
documentation, https://docs.vllm.ai/en/latest/design/metrics.html):

  gauges (a value right now)
    vllm:kv_cache_usage_perc        fraction of KV-cache blocks in use, 0..1
    vllm:num_requests_running       requests in the running batch
    vllm:num_requests_waiting       requests queued, not yet scheduled
  counters (only ever increase; the plots take differences per second)
    vllm:prompt_tokens_total        input tokens processed (prefill)
    vllm:generation_tokens_total    output tokens generated (decode)
    vllm:prefix_cache_queries_total tokens looked up in the prefix cache
    vllm:prefix_cache_hits_total    of those, tokens found (no recompute)
    vllm:num_preemptions_total      requests evicted from the batch for lack of KV

A metric can carry several label sets (one per engine or per model); values
with the same name are summed.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from pathlib import Path

from thunderagent_controller_mod._runtime.metrics.prometheus_text_format import iter_samples

SAMPLED_METRICS = (
    "vllm:kv_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:num_preemptions_total",
)


def metrics_url(base_url: str) -> str:
    """http://host:port/v1 -> http://host:port/metrics"""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return f"{root}/metrics"


def read_once(url: str) -> dict[str, float]:
    with urllib.request.urlopen(url, timeout=5) as reply:
        text = reply.read().decode(errors="replace")
    values: dict[str, float] = {}
    for name, _labels, value in iter_samples(text):
        if name in SAMPLED_METRICS:
            values[name] = values.get(name, 0.0) + value
    return values


class EngineMetricsSampler:
    """Start before the first request, stop after the last; writes metrics.jsonl."""

    def __init__(
        self,
        engine_urls: list[str],
        out_path: Path,
        start: float,
        interval_seconds: float = 1.0,
    ):
        self.engine_urls = engine_urls
        self.out_path = out_path
        self.start = start
        self.interval_seconds = interval_seconds
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.samples = 0
        self.failures = 0

    def loop(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("w") as output:
            while not self.stopping.is_set():
                for engine in self.engine_urls:
                    sampled_at = time.monotonic() - self.start
                    try:
                        values = read_once(metrics_url(engine))
                    except Exception:  # noqa: BLE001 - a missed sample is a gap
                        self.failures += 1
                        continue
                    self.samples += 1
                    line = {"t": round(sampled_at, 3), "engine": engine} | values
                    output.write(json.dumps(line) + "\n")
                output.flush()
                self.stopping.wait(self.interval_seconds)

    def __enter__(self) -> EngineMetricsSampler:
        self.thread.start()
        return self

    def __exit__(self, *exception) -> None:
        self.stopping.set()
        self.thread.join()
