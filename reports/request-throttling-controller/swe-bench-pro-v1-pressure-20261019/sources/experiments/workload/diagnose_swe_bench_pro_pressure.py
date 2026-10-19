"""Read-only pressure diagnosis of a previously verified real-Agent archive.

Scaling below is a planning projection, never a measured larger-batch result.
The pinned upstream capacity input and scheduling policy are not modified.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import tarfile


def diagnose(archive, expected_sha256):
    archive = Path(archive)
    with archive.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected_sha256:
        raise ValueError("Original archive SHA256 mismatch")
    batches = []
    with tarfile.open(archive) as source:
        def read(name):
            return json.loads(source.extractfile(name).read())

        for name in sorted(source.getnames()):
            if not name.endswith("/batch-report.json") or not name.startswith("suite/pair-"):
                continue
            base = name.rsplit("/", 1)[0]
            report = read(name)
            start = min(x["started_monotonic"] for x in report["tasks"])
            end = max(x["finished_monotonic"] for x in report["tasks"])
            samples = [json.loads(line) for line in source.extractfile(
                base + "/observations-private.jsonl").read().splitlines()]
            samples = [x for x in samples if start <= x["monotonic"] <= end]
            backends = [v for x in samples for v in x.get("scheduler", {}).get("backends", {}).values()]
            if not backends:
                raise ValueError("No actual scheduler observations: " + base)
            capacities = sorted({x["capacity"] for x in backends})
            if len(capacities) != 1:
                raise ValueError("Capacity changed within original batch")
            metrics = read(base + "/engine-metrics-before.json")
            labels = re.search(r'^vllm:cache_config_info\{([^}]+)\}', metrics, re.M)
            if labels is None:
                raise ValueError("Original engine cache configuration absent")
            labels = dict(re.findall(r'(\w+)="([^"]*)"', labels[1]))
            raw = int(labels["block_size"]) * int(labels["num_gpu_blocks"])
            if raw != capacities[0]:
                raise ValueError("Original capacity differs from pinned raw block formula")
            engine_capacity = int(labels["kv_cache_size_tokens"])
            engine_samples = []
            for x in samples:
                value = re.search(r'^vllm:kv_cache_usage_perc\{[^}]*\}\s+([0-9.eE+-]+)',
                                  x.get("engine", ""), re.M)
                if value:
                    engine_samples.append(float(value[1]))
            batches.append({
                "batch": base.split("/")[-1], "programs": report["batch_size"],
                "observations": len(samples), "declared_capacity": raw,
                "engine_reported_cache_size_tokens": engine_capacity,
                "raw_capacity_over_engine_report_percent": 100 * (raw / engine_capacity - 1),
                "active_estimated_tokens_max": max(x["active_program_tokens"] for x in backends),
                "charged_tokens_max": raw - min(x["remaining_capacity"] for x in backends),
                "remaining_capacity_min": min(x["remaining_capacity"] for x in backends),
                "sampled_overflow_count": sum(x["remaining_capacity"] < 0 for x in backends),
                "sampled_waiting_max": max(len(x.get("scheduler", {}).get("waiting", [])) for x in samples),
                "sampled_engine_cache_usage_max": max(engine_samples) if engine_samples else None,
                "observer_error_samples": sum(any(k.endswith("_error") for k in x) for x in samples),
                "required_replication_ignoring_sharing_and_timing":
                    math.floor(raw / max(raw - x["remaining_capacity"] for x in backends)) + 1,
            })
    if len(batches) != 6:
        raise ValueError("Expected the original six completed batches")
    return {"status": "READ_ONLY_DIAGNOSIS", "archive_sha256": actual, "batches": batches,
            "reason": "Pinned scheduler pauses only when remaining_capacity < 0; no sampled overflow in original run",
            "capacity_boundary": "Raw blocks times block size differs from hybrid engine-reported capacity; neither is silently changed",
            "projection_boundary": "Replicas may share prefixes and change timing/work. Projection does not guarantee pause, memory pressure or benefit",
            "new_model_calls": 0}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    result = diagnose(a.archive, a.sha256)
    Path(a.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
