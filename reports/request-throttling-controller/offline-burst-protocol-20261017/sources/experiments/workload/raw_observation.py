"""Record ground truth while a replay runs, as it comes, without deriving anything.

Two background samplers, both writing gzip-compressed JSON lines, one line per
source per sample, every line stamped with `t` (seconds since the replay
started, the replay's own clock) and `wall` (Unix time, to line up with logs):

EngineRawMetricsSampler -> engine_metrics_raw.jsonl.gz, every `interval` s:
    {"t": 12.0, "wall": 1790..., "engine": "http://127.0.0.1:8100/v1",
     "samples": [["vllm:num_requests_running", {"engine": "0", ...}, 7.0],
                 ["vllm:time_to_first_token_seconds_bucket", {"le": "0.5"}, 123.0],
                 ...]}
    Every `vllm:*` sample of GET /metrics, labels included: gauges, counters,
    and every histogram bucket, sum and count (which vLLM metric means what:
    https://docs.vllm.ai/en/latest/design/metrics.html). `*_created`
    timestamps are dropped; nothing else is.

DeviceSampler -> device.jsonl.gz, every `interval` s, one line per card:
    {"t": 12.0, "wall": ..., "card": 0, "hbm_capacity_mb": 65536,
     "hbm_usage_rate_percent": 87, "aicore_usage_rate_percent": 35,
     "hbm_bandwidth_usage_rate_percent": 12, "npu_utilization_percent": 40,
     "process_memory_mb": {"3937521": 53962}}
    Read from `npu-smi info -t usages -i <card>` (rates, as npu-smi reports
    them) and `npu-smi info` (memory per process on the card, MB).
"""

from __future__ import annotations

import gzip
import json
import re
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from engine_metrics import metrics_url

from thunderagent_controller_mod._runtime.metrics.prometheus_text_format import iter_samples


class _Sampler:
    def __init__(self, out_path: Path, start: float, interval_seconds: float):
        self.out_path = out_path
        self.start = start
        self.interval_seconds = interval_seconds
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.samples = 0
        self.failures = 0

    def lines(self) -> list[dict]:  # one sample round; overridden
        raise NotImplementedError

    def loop(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(self.out_path, "at") as output:
            while not self.stopping.is_set():
                stamp = {"t": round(time.monotonic() - self.start, 3),
                         "wall": round(time.time(), 3)}  # fmt: skip
                try:
                    for line in self.lines():
                        output.write(json.dumps(stamp | line) + "\n")
                        self.samples += 1
                except Exception:  # noqa: BLE001 - a missed sample is a gap, not a crash
                    self.failures += 1
                output.flush()
                self.stopping.wait(self.interval_seconds)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exception) -> None:
        self.stopping.set()
        self.thread.join()


class EngineRawMetricsSampler(_Sampler):
    def __init__(self, engine_urls, out_path, start, interval_seconds=1.0):
        super().__init__(out_path, start, interval_seconds)
        self.engine_urls = engine_urls

    def lines(self) -> list[dict]:
        result = []
        for engine in self.engine_urls:
            with urllib.request.urlopen(metrics_url(engine), timeout=5) as reply:
                text = reply.read().decode(errors="replace")
            samples = [
                [name, labels, value]
                for name, labels, value in iter_samples(text)
                if name.startswith("vllm:") and not name.endswith("_created")
            ]
            result.append({"engine": engine, "samples": samples})
        return result


USAGE_FIELDS = {
    "HBM Capacity(MB)": "hbm_capacity_mb",
    "HBM Usage Rate(%)": "hbm_usage_rate_percent",
    "Aicore Usage Rate(%)": "aicore_usage_rate_percent",
    "HBM Bandwidth Usage Rate(%)": "hbm_bandwidth_usage_rate_percent",
    "NPU Utilization(%)": "npu_utilization_percent",
}


def run(command: list[str]) -> str:
    return subprocess.run(
        command, capture_output=True, text=True, timeout=20, check=False
    ).stdout


class DeviceSampler(_Sampler):
    def __init__(self, cards, out_path, start, interval_seconds=5.0):
        super().__init__(out_path, start, interval_seconds)
        self.cards = cards

    def lines(self) -> list[dict]:
        table = run(["npu-smi", "info"])
        memory: dict[int, dict[str, int]] = {}
        in_processes = False
        for row in table.splitlines():
            if "Process id" in row:
                in_processes = True
                continue
            match = re.match(r"\|\s*(\d+)\s+\d+\s*\|\s*(\d+)\s*\|[^|]*\|\s*(\d+)", row)
            if in_processes and match:
                card, pid, megabytes = (int(x) for x in match.groups())
                memory.setdefault(card, {})[str(pid)] = megabytes
        result = []
        for card in self.cards:
            usage = run(["npu-smi", "info", "-t", "usages", "-i", str(card)])
            line: dict = {"card": card}
            for row in usage.splitlines():
                key, _, value = row.strip().partition(":")
                if key.strip() in USAGE_FIELDS and value.strip().isdigit():
                    line[USAGE_FIELDS[key.strip()]] = int(value.strip())
            line["process_memory_mb"] = memory.get(card, {})
            result.append(line)
        return result
