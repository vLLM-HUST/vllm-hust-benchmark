"""Adapt saved real-Agent metrics to the existing plot_run engine figure.

No new plot style, sampling or rate implementation. No invented request latency.
Hybrid KV tokens in this figure are an engine-reported capacity equivalent;
raw occupancy fractions remain in metrics.jsonl and Grafana.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from engine_metrics import SAMPLED_METRICS
from plot_run import plot_engine
from thunderagent_controller_mod._runtime.metrics.prometheus_text_format import iter_samples


def export(observations, batch_report, before_metrics, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = json.loads(Path(batch_report).read_text())
    start = min(x["started_monotonic"] for x in report["tasks"])
    end = max(x["finished_monotonic"] for x in report["tasks"])
    before = json.loads(Path(before_metrics).read_text())
    match = re.search(r'^vllm:cache_config_info\{([^}]+)\}', before, re.M)
    if not match:
        raise ValueError("Original engine cache configuration required")
    labels = dict(re.findall(r'(\w+)="([^"]*)"', match[1]))
    # This is the actual engine's hybrid-capacity estimate, not the router's
    # block_size*num_gpu_blocks input. Preserve both in the figure manifest.
    capacity = int(labels["kv_cache_size_tokens"])
    samples = []
    missed = 0
    for line in Path(observations).read_text().splitlines():
        row = json.loads(line)
        if not start <= row["monotonic"] <= end:
            continue
        if "engine" not in row:
            missed += 1
            continue
        sample = {"t": row["monotonic"] - start, "engine": "model"}
        for name, _labels, value in iter_samples(row["engine"]):
            if name in SAMPLED_METRICS:
                sample[name] = sample.get(name, 0) + value
        samples.append(sample)
    if not samples:
        raise ValueError("No actual engine samples in Agent window")
    (output / "metrics.jsonl").write_text("".join(json.dumps(x) + "\n" for x in samples))
    manifest = {
        "sources_sha256": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                           for p in [observations, batch_report, before_metrics]},
        "engine_samples": len(samples), "missing_engine_observations": missed,
        "engine_reported_capacity_tokens": capacity,
        "router_raw_block_capacity": int(labels["block_size"]) * int(labels["num_gpu_blocks"]),
        "kv_axis_boundary": "Occupancy fraction times engine hybrid capacity estimate; token-equivalent, not exact resident attention tokens",
        "standard_plot": "experiments/workload/plot_run.py:plot_engine; unchanged palette/style and 10-second rate window",
        "no_request_latency_figure": "Non-streaming Agent archive does not independently record per-request first token times",
    }
    (output / "figure-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return samples, capacity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--batch-report", type=Path, required=True)
    parser.add_argument("--before-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples, capacity = export(args.observations, args.batch_report, args.before_metrics, args.output)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot
    for extension in ["png", "pdf"]:
        plot_engine(pyplot, samples, {"model": capacity}, args.output / ("engine_over_time." + extension),
                    args.output.name + " | hybrid KV token-equivalent estimate")
    print("STANDARD_ENGINE_FIGURES_WRITTEN", args.output)
