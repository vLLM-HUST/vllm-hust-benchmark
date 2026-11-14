"""Render project final goals separately from milestone verdicts; no new measurements."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.summary.read_bytes())
    final = data["project_final_assessment"]
    if not final["fully_specified"]:
        raise ValueError("Final target specification is incomplete")
    targets = final["targets"]
    performance = targets["3.4.2"]["numeric_results_by_profile"]
    cost = next(
        row
        for row in targets["3.4.4"]["numeric_cost_results_by_profile"]
        if row["dataset"] == "mmlu"
    )
    import matplotlib

    if matplotlib.__version__ != "3.10.8":
        raise ValueError("Expected matplotlib3.10.8")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["svg.hashsalt"] = "project-final-goals-20261113-v1"
    panels = [
        (
            "4.1 Workload-reference efficiency",
            "Target >=90%",
            f"Recorded lower bound >={float(targets['3.4.1']['evidence_lower_bound_percent']):.2f}%",
            "Historical out256 only; 90% not demonstrated",
        ),
        (
            "4.2 Throughput and latency",
            "Target >=2x; latency -20%",
            f"{sum(p['numeric_threshold_met'] for p in performance)}/{len(performance)} profiles meet numeric thresholds",
            "Missing numeric targets: "
            + ", ".join(
                p["dataset"] for p in performance if not p["numeric_threshold_met"]
            )
            if not all(p["numeric_threshold_met"] for p in performance)
            else "All profiles meet numeric thresholds; quality shown separately",
        ),
        (
            "4.3 Context stability",
            "Target >=32K",
            "Current continuous C8: 4K / 16K",
            "No qualified 32K continuous / tail-latency evidence",
        ),
        (
            "4.4 Cost and utilization",
            "Target cost -30% + multi-tenant",
            f"MMLU equal-rate estimate: -{cost['equal_rate_unit_cost_reduction_percent']:.2f}%",
            "High multi-tenant utilization not demonstrated",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.3))
    for axis, (name, goal, observed, limit) in zip(axes.flat, panels, strict=True):
        axis.set_facecolor("#fff7ed")
        axis.set_title(name, loc="left", fontsize=11, weight="bold")
        axis.text(
            0.04, 0.73, goal, fontsize=15, weight="bold", transform=axis.transAxes
        )
        axis.text(0.04, 0.47, observed, fontsize=10, transform=axis.transAxes)
        axis.text(
            0.04, 0.20, limit, fontsize=8.5, color="#9a3412", transform=axis.transAxes
        )
        axis.set_xticks([])
        axis.set_yticks([])
    verdict = "established" if final["acceptance_established"] else "NOT demonstrated"
    fig.suptitle("Project final goals - final acceptance " + verdict, fontsize=15)
    fig.text(
        0.5,
        0.018,
        "Confirmed: original peak term uses the matched-workload reference; Qwen3.5-35B-A3B. Milestones remain separate.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.output / "project-final-targets.png",
        dpi=160,
        metadata={"Software": "Matplotlib3.10.8"},
    )
    svg = args.output / "project-final-targets.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
