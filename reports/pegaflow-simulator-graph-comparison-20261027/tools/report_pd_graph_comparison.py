"""Archive and plot a measured-graph-cost kernel comparison."""

import argparse
import csv
import shutil
from pathlib import Path

from compare_pd_simulators import bundle_input, read, write
from report_pd_simulator_comparison import normalized, summarize


def build(args):
    contract = read(args.runs / "contract.json")
    bundle = bundle_input(args.inputs)
    protocol = read(args.inputs / "protocol.json")
    order = protocol["test_order"]
    offered = dict(zip(order, contract["shared"]["arrivals_seconds"], strict=True))
    summary, pairs, norms = {}, [], {}
    for case in contract["cases"]:
        name = case["name"]
        arms, normalized_arms = {}, {}
        for engine in ("ours", "peer"):
            result = read(args.runs / "runs" / f"{name}-{engine}.json.gz")
            norm = normalized(result, engine, order, offered)
            previous = {}
            for row in bundle["rows"]:
                request = norm[row["id"]]
                expected = max(offered[row["id"]], previous.get(row["task"], 0.0))
                if abs(request["ready"] - expected) > 1e-8:
                    raise ValueError("Session causality differs")
                previous[row["task"]] = request["end"]
            arms[engine] = summarize(result, engine, norm, bundle["rows"])
            arms[engine]["D_peak_sequences"] = (
                {
                    str(n): max(
                        (
                            len(s["active"])
                            for s in result["engine_steps"]
                            if s["node"] == n
                        ),
                        default=0,
                    )
                    for n in (2, 3)
                }
                if engine == "ours"
                else {
                    str(n): result["D_summary"][str(n)]["peak_active"] for n in (2, 3)
                }
            )
            normalized_arms[engine] = norm
        absolute = []
        for key in order:
            a, b = (normalized_arms[e][key] for e in ("ours", "peer"))
            if (a["P"], a["D"]) != (b["P"], b["D"]):
                raise ValueError("Route differs")
            delta = b["TTFT"] - a["TTFT"]
            absolute.append(abs(delta))
            pairs.append(
                {
                    "case": name,
                    "id": key,
                    "ours_TTFT": a["TTFT"],
                    "peer_TTFT": b["TTFT"],
                    "peer_minus_ours": delta,
                }
            )
        summary[name] = {
            "arms": arms,
            "MAE_seconds": sum(absolute) / len(absolute),
            "max_absolute_error_seconds": max(absolute),
            "requests_different_at_1us": sum(v > 1e-6 for v in absolute),
        }
        norms[name] = normalized_arms
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(args.runs / "runs", args.output / "runs", dirs_exist_ok=True)
    shutil.copyfile(args.runs / "contract.json", args.output / "contract.json")
    write(args.output / "summary.json", summary)
    inputs = args.output / "inputs"
    write(inputs / "simulation-inputs.json.gz", bundle)
    write(inputs / "protocol.json", protocol)
    shutil.copytree(
        args.inputs / "calibration-seed",
        inputs / "calibration-seed",
        dirs_exist_ok=True,
    )
    shutil.copyfile(args.graph_model, inputs / "step-cost-model.json")
    shutil.copyfile(args.graph_components, inputs / "components.json")
    tools = args.output / "tools"
    tools.mkdir(exist_ok=True)
    for name in (
        "compare_pd_simulators.py",
        "pd_graph_comparison.py",
        "report_pd_simulator_comparison.py",
        "report_pd_graph_comparison.py",
    ):
        shutil.copyfile(Path(__file__).with_name(name), tools / name)
    with (args.output / "paired-requests.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    plot(args.output, norms)


def plot(root, norms):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for cap, ax in zip((1, 4), axes, strict=True):
        for engine, label, color in (
            ("ours", "Our kernel", "#2563eb"),
            ("peer", "Collaborator kernel", "#d97706"),
        ):
            values = sorted(
                r["TTFT"] for r in norms[f"graph-host16-D{cap}"][engine].values()
            )
            ax.plot(
                [100 * (i + 1) / len(values) for i in range(len(values))],
                values,
                label=label,
                color=color,
                linewidth=2,
            )
        ax.set(
            title=f"D concurrency {cap} per instance",
            xlabel="Request percentile (%)",
            ylabel="TTFT (s)",
            xlim=(0, 100),
            ylim=(0, None),
        )
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.suptitle(
        "Same measured graph-step costs | CPU/NPU KV 16 GiB each\n95 requests; 2048-token budget; P serial; GPU APC off",
        fontsize=12,
    )
    fig.savefig(root / "ttft-graph-comparison.png", dpi=170)
    fig.savefig(root / "ttft-graph-comparison.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runs", "inputs", "output", "graph-model", "graph-components"):
        parser.add_argument("--" + name, type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
