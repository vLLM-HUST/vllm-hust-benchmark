"""Check observed launch treatment against this experiment's frozen host profile."""

import json
import math

from evidence import require


def verify_lifecycle_records(
    lifecycle: dict, records: dict, contract_sha: str, previous_shutdown: float
) -> dict:
    """Join original process/readiness/completion receipts without inventing observations."""
    process, ready, completion, shutdown = (
        records[key] for key in ("process", "ready", "completion", "shutdown")
    )
    require(
        process["lifecycle"] == lifecycle
        and process["contract_sha256"] == contract_sha,
        "Lifecycle contract or definition differs",
    )
    require(
        type(process["pid"]) is int
        and process["pid"] > 0
        and process["pid"] == process["pgid"] == ready["pid"],
        "Lifecycle process identity differs",
    )
    require(
        isinstance(process["start_ticks"], str) and process["start_ticks"].isdigit(),
        "Lifecycle start ticks are invalid",
    )
    times = [row["at"] for row in (process, ready, completion, shutdown)]
    require(
        all(
            isinstance(value, (int, float)) and math.isfinite(value) for value in times
        ),
        "Lifecycle timestamp is invalid",
    )
    require(
        previous_shutdown < times[0] < times[1] <= times[2] <= times[3],
        "Lifecycle timestamps overlap or are reversed",
    )
    require(
        completion["request_execution_complete"] is True
        and completion["requests"] == lifecycle["request_count"],
        "Lifecycle requests are incomplete",
    )
    require(
        type(shutdown["server_returncode"]) is int,
        "Lifecycle lacks final server return code",
    )
    snapshot = records.get("snapshot")
    if snapshot is None:
        require(
            lifecycle["name"] == "quality8-off" and not lifecycle["enabled"],
            "Lifecycle actual runtime snapshot missing",
        )
    else:
        require(
            snapshot["pid"] == process["pid"]
            and times[0] <= snapshot["at"] <= times[2],
            "Lifecycle runtime snapshot belongs to another process or time",
        )
        verify_runtime(snapshot, lifecycle)
    return {
        "identity": [process["pid"], process["start_ticks"]],
        "started_at": times[0],
        "ready_at": times[1],
        "completed_at": times[2],
        "shutdown_at": times[3],
        "server_returncode": shutdown["server_returncode"],
        "actual_runtime_snapshot": snapshot is not None,
    }


EXPERIMENT = "/root/.cache/pyramidkv-issue31-20261103"
MODEL = "/root/workspace/Qwen3.5-35B-A3B"


def verify_runtime(snapshot: dict, lifecycle: dict) -> None:
    expected = {
        "--host": "127.0.0.1",
        "--port": "8013",
        "--served-model-name": "qwen35-pyramidkv",
        "--tensor-parallel-size": "2",
        "--distributed-executor-backend": "mp",
        "--worker-cls": "vllm_ascend_pyramidkv.adapter.host_worker.PyramidKVHostWorker",
        "--dtype": "bfloat16",
        "--max-model-len": "32768",
        "--max-num-seqs": "4",
        "--max-num-batched-tokens": "4096",
        "--gpu-memory-utilization": "0.9",
        "--kv-cache-memory-bytes": str(lifecycle["kv_bytes_per_worker"]),
        "--seed": "17",
        "--mamba-cache-mode": "align",
        "--generation-config": "vllm",
        "--additional-config": {"enable_cpu_binding": False},
        "--limit-mm-per-prompt": {"image": 0, "video": 0},
        "--compilation-config": {
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "cudagraph_capture_sizes": [3, 6, 12],
            "max_cudagraph_capture_size": 12,
        },
        "--speculative-config": {"method": "mtp", "num_speculative_tokens": 2},
    }
    flags = {
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--async-scheduling",
        "--enable-prompt-tokens-details",
    }
    argv = snapshot["argv"]
    require(
        argv[:5] == ["python", "-m", "vllm.entrypoints.cli.main", "serve", MODEL],
        "Runtime entrypoint differs",
    )
    found = {}
    index = 5
    while index < len(argv):
        key = argv[index]
        require(
            key not in found and key in flags | set(expected),
            f"Runtime duplicate or unknown argument: {key}",
        )
        if key in flags:
            found[key] = True
            index += 1
        else:
            require(index + 1 < len(argv), f"Runtime argument lacks a value: {key}")
            value = argv[index + 1]
            found[key] = json.loads(value) if isinstance(expected[key], dict) else value
            index += 2
    require(
        found == dict(expected, **dict.fromkeys(flags, True)),
        "Runtime serving profile differs",
    )
    enabled = str(int(lifecycle["enabled"]))
    fixed_environment = {
        "PYRAMIDKV_ENABLED": enabled,
        "VLLM_ASCEND_PYRAMIDKV_ENABLED": enabled,
        "VLLM_ASCEND_PYRAMIDKV_BATCHED_METADATA": "1",
        "VLLM_ASCEND_PYRAMIDKV_EXPERIMENTAL_MTP2": "1",
        "ASCEND_RT_VISIBLE_DEVICES": "0,1",
        "MODEL_PATH": MODEL,
        "SERVE_PORT": "8013",
        "PYRAMIDKV_KV_CACHE_MEMORY_BYTES": str(lifecycle["kv_bytes_per_worker"]),
        "PYRAMIDKV_CONFIG": EXPERIMENT + "/pyramidkv-aligned.json",
        "VLLM_ASCEND_PYRAMIDKV_CONFIG": EXPERIMENT + "/pyramidkv-aligned.json",
        "VLLM_PLUGINS": "ascend,ascend_model,ascend_model_loader,ascend_kv_connector,ascend_pyramidkv",
    }
    env = snapshot["selected_environment"]
    for name, value in fixed_environment.items():
        require(env.get(name) == value, f"Runtime environment differs: {name}")
    prefix = [
        EXPERIMENT + "/release-overlay",
        "/root/workspace/pyramidkv-runtime/src/vllm",
        "/root/workspace/pyramidkv-runtime/src/vllm-ascend",
    ]
    require(
        env.get("PYTHONPATH", "").split(":")[:3] == prefix,
        "Runtime plugin/host import paths differ",
    )
