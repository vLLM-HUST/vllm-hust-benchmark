#!/usr/bin/env python3
"""Read how much context a running vLLM engine can hold, and save it as JSON, so
a workload can be sized to the machine it will run on.

    python experiments/workload/engine_capacity.py \\
        --base-url http://127.0.0.1:8100/v1 \\
        --out data/workloads/engine_capacity/<name>.json

Everything comes from the engine itself; nothing is typed in:

- `max_model_len` (GET /v1/models): the most tokens one request may hold,
  prompt plus output. A round longer than this is rejected with HTTP 400.
- `vllm:cache_config_info` (GET /metrics), a gauge whose labels are the KV-cache
  configuration:
    block_size          tokens per KV-cache block
    num_gpu_blocks      blocks in device memory (NPU / GPU)
    num_cpu_blocks      blocks in host memory when KV offloading is on, else None
    kv_offloading_size  the offloading size the engine was started with, else None
  so the KV-cache capacity in tokens is blocks x block_size, on the device and
  (if offloading is on) in host memory. It is the same number the engine logs at
  start-up as "GPU KV cache size: N tokens".

The output also records which model and which endpoint were measured.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from engine_metrics import metrics_url

from thunderagent_controller_mod._runtime.metrics.prometheus_text_format import iter_samples


def read_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as reply:
        return json.loads(reply.read())


def as_int(label: str | None) -> int | None:
    return None if label in (None, "None", "") else int(float(label))


def measure(base_url: str) -> dict:
    models = read_json(f"{base_url.rstrip('/')}/models")["data"]
    with urllib.request.urlopen(metrics_url(base_url), timeout=10) as reply:
        text = reply.read().decode(errors="replace")
    cache = next(
        labels
        for name, labels, _ in iter_samples(text)
        if name == "vllm:cache_config_info"
    )
    block_size = as_int(cache.get("block_size"))
    gpu_blocks = as_int(cache.get("num_gpu_blocks"))
    cpu_blocks = as_int(cache.get("num_cpu_blocks"))
    return {
        "endpoint": base_url,
        "model": models[0]["id"],
        "model_path": models[0].get("root"),
        "max_model_len": models[0]["max_model_len"],
        "block_size": block_size,
        "num_gpu_blocks": gpu_blocks,
        "num_cpu_blocks": cpu_blocks,
        "kv_offloading_size": cache.get("kv_offloading_size"),
        "kv_capacity_tokens_device": gpu_blocks * block_size,
        "kv_capacity_tokens_host": cpu_blocks * block_size if cpu_blocks else 0,
        "gpu_memory_utilization": cache.get("gpu_memory_utilization"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8100/v1")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    capacity = measure(arguments.base_url)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(capacity, indent=1) + "\n")
    print(json.dumps(capacity, indent=1))
    print(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
