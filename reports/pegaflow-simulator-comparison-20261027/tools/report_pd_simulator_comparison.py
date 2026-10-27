"""Normalize paired simulator traces and render diagnostic comparisons."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path

from compare_pd_simulators import bundle_input, quantiles, read, write


def normalized(result, engine, order, offered):
    rows = {r["id"]: r for r in result["per_request"]}
    if (
        set(rows) != set(order)
        or len(rows) != len(order)
        or len(result["per_request"]) != len(order)
    ):
        raise ValueError("Requests must complete exactly once")
    output = {}
    for key in order:
        row = rows[key]
        ready = row["ready"] if engine == "ours" else offered[key] + row["dependency"]
        first = row["first"] if engine == "ours" else offered[key] + row["nominal"]
        end = row["end"]
        if not all(math.isfinite(v) for v in (ready, first, end)):
            raise ValueError("Nonfinite clock")
        if first < ready - 1e-9 or end < first - 1e-9:
            raise ValueError("Invalid first/full timeline")
        if abs(first - ready - row["TTFT"]) > 1e-9:
            raise ValueError("Inconsistent TTFT definition")
        output[key] = {
            "id": key,
            "ready": ready,
            "first": first,
            "end": end,
            "TTFT": first - ready,
            "E2E": end - ready,
            "P": row["P" if engine == "ours" else "p"],
            "D": row["D" if engine == "ours" else "d"],
        }
    return output


def summarize(result, engine, norm, rows):
    stages = result["apis" if engine == "ours" else "stages"]
    if Counter(r["id"] for r in stages) != Counter({key: 2 for key in norm}):
        raise ValueError("Each request needs a P and D service record")
    if any(
        v > result["comparison"]["case"]["host_gib"] * 2**30
        for v in result["peak_host_bytes"]
    ):
        raise ValueError("Host capacity exceeded")
    actions = result["cache_events" if engine == "ours" else "actions"]
    hit_key = "cached_tokens" if engine == "ours" else "prefix"
    hits = {str(n): sum(r[hit_key] for r in stages if r["node"] == n) for n in range(4)}
    return {
        "TTFT": quantiles([r["TTFT"] for r in norm.values()]),
        "E2E": quantiles([r["E2E"] for r in norm.values()]),
        "finish_seconds": max(r["end"] for r in norm.values()),
        "D_counts": dict(Counter(str(r["D"]) for r in norm.values())),
        "cached_tokens": hits,
        "D_uncached_tokens": sum(r["tokens"] for r in rows) - hits["2"] - hits["3"],
        "transfer_bytes": sum(
            r.get("payload", 0)
            for r in actions
            if r["kind"] in ("copy", "implicit_copy")
        ),
        "peak_host_bytes": result["peak_host_bytes"],
        "final_virtual_bytes": [r["virtual_bytes"] for r in result["final_cache"]],
        "wall_seconds": result["comparison"]["wall_seconds"],
    }


def build(root, inputs, destination):
    contract = read(root / "contract.json")
    bundle = bundle_input(inputs)
    protocol = read(inputs / "protocol.json")
    order = protocol["test_order"]
    offered = dict(zip(order, contract["shared"]["arrivals_seconds"], strict=True))
    paired, table, all_norm = [], {}, {}
    for case in contract["cases"]:
        name = case["name"]
        raw, norm, arms = {}, {}, {}
        for engine in ("ours", "peer"):
            raw[engine] = read(root / "runs" / f"{name}-{engine}.json.gz")
            if raw[engine]["comparison"]["case"] != case:
                raise ValueError("Case parameters differ")
            norm[engine] = normalized(raw[engine], engine, order, offered)
            previous = {}
            for key in order:
                row = norm[engine][key]
                task = next(r["task"] for r in bundle["rows"] if r["id"] == key)
                expected_ready = max(offered[key], previous.get(task, 0.0))
                if abs(expected_ready - row["ready"]) > 1e-8:
                    raise ValueError("Session dependency differs from contract")
                previous[task] = row["end"]
            arms[engine] = summarize(raw[engine], engine, norm[engine], bundle["rows"])
        absolute = []
        for key in order:
            a, b = norm["ours"][key], norm["peer"][key]
            if (a["P"], a["D"]) != (b["P"], b["D"]):
                raise ValueError("P/D assignment differs")
            delta = b["TTFT"] - a["TTFT"]
            absolute.append(abs(delta))
            paired.append(
                {
                    "case": name,
                    "id": key,
                    "ours_TTFT": a["TTFT"],
                    "peer_TTFT": b["TTFT"],
                    "peer_minus_ours": delta,
                    "ours_ready": a["ready"],
                    "peer_ready": b["ready"],
                    "ours_end": a["end"],
                    "peer_end": b["end"],
                }
            )
        stage_a = {(r["id"], r["node"]): r for r in raw["ours"]["comparison_stages"]}
        stage_b = {(r["id"], r["node"]): r for r in raw["peer"]["comparison_stages"]}
        if set(stage_a) != set(stage_b):
            raise ValueError("Stage identity differs")
        diff = [
            k
            for k in order
            if abs(norm["ours"][k]["TTFT"] - norm["peer"][k]["TTFT"]) > 1e-6
        ]
        comparison = {
            "MAE_seconds": sum(absolute) / len(absolute),
            "max_absolute_error_seconds": max(absolute),
            "requests_different_at_1us": len(diff),
            "first_different_in_input_order": diff[0] if diff else None,
            "stage_prefix_disagreements": sum(
                stage_a[k]["selected_blocks"] != stage_b[k]["selected_blocks"]
                for k in stage_a
            ),
            "TTFT_mean_relative_peer_minus_ours_pct": 100
            * (arms["peer"]["TTFT"]["mean"] / arms["ours"]["TTFT"]["mean"] - 1),
            "TTFT_p99_relative_peer_minus_ours_pct": 100
            * (arms["peer"]["TTFT"]["p99"] / arms["ours"]["TTFT"]["p99"] - 1),
        }
        if (
            name in ("cold-host16-D1", "cold-host16-D4", "cache-host16-D1")
            and max(absolute) > 1e-9
        ):
            raise ValueError("Shared deterministic control did not match")
        table[name] = {"settings": case, "arms": arms, "comparison": comparison}
        all_norm[name] = norm
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(root / "runs", destination / "runs", dirs_exist_ok=True)
    shutil.copyfile(root / "contract.json", destination / "contract.json")
    write(destination / "summary.json", table)
    with (destination / "paired-requests.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    output_inputs = destination / "inputs"
    write(output_inputs / "simulation-inputs.json.gz", bundle)
    write(output_inputs / "protocol.json", protocol)
    shutil.copytree(
        inputs / "calibration-seed",
        output_inputs / "calibration-seed",
        dirs_exist_ok=True,
    )
    tools = destination / "tools"
    tools.mkdir(exist_ok=True)
    for name in ("compare_pd_simulators.py", "report_pd_simulator_comparison.py"):
        shutil.copyfile(Path(__file__).with_name(name), tools / name)
    draw(destination, all_norm)
    write(
        destination / "manifest.json",
        {
            "kind": "offline-model-comparison",
            "requests_each": len(order),
            "paired_cases": len(table),
            "simulation_runs": len(table) * 2,
            "source_commits": contract["source_commits"],
            "source_inputs": "Original 4bfa935 input bundle and isolated32 seed; benchmark serial-validation collection",
            "harness_sha256": hashlib.sha256(
                (tools / "compare_pd_simulators.py").read_bytes()
            ).hexdigest(),
            "validation": "unique95, causal clocks/session dependencies, same P/D, finite TTFT, two stages/request, host budgets, three exact controls",
            "model_boundary": "Common execution-kernel diagnostic; strategy/control CPU disabled; eager128; GPU APC off; real production performance unqualified",
        },
    )
    print(
        json.dumps({name: row["comparison"] for name, row in table.items()}, indent=2)
    )


def draw(root, results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "svg.fonttype": "none", "font.size": 10}
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    cases = [
        ("cache-host16-D4", "16 GiB CPU cache / D concurrency 4"),
        ("cache-host2-D4", "2 GiB CPU cache / D concurrency 4"),
        ("cache-host16-D4-sharing75", "16 GiB / decode sharing 0.75 (sensitivity)"),
        ("cache-host2-D4-physical-copy", "2 GiB / peer committed-copy flag corrected"),
    ]
    colors = {"ours": "#2563eb", "peer": "#d97706"}
    for ax, (case, title) in zip(axes.flat, cases, strict=True):
        for engine in ("ours", "peer"):
            values = sorted(r["TTFT"] for r in results[case][engine].values())
            percentages = [100 * (i + 1) / len(values) for i in range(len(values))]
            ax.plot(
                percentages,
                values,
                color=colors[engine],
                label="Our model" if engine == "ours" else "Collaborator model",
                linewidth=2,
            )
        ax.set(
            title=title,
            xlabel="Request percentile (%)",
            ylabel="TTFT (s)",
            xlim=(0, 100),
            ylim=(0, None),
        )
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, loc="upper left")
    fig.suptitle(
        "Simulator cross-check | same 95 requests, arrivals, routes and singleton costs\nDiagnostic contract: control cost 0; eager; 128-token budget; P serial",
        fontsize=13,
    )
    fig.savefig(root / "ttft-comparison.png", dpi=170)
    fig.savefig(root / "ttft-comparison.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.runs, args.inputs, args.output)


if __name__ == "__main__":
    main()
