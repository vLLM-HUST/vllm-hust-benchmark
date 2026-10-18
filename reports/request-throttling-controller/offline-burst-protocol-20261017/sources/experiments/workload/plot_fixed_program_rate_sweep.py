#!/usr/bin/env python3
"""Plot the canonical fixed-workload comparison CSV without recomputing metrics."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--title", default=None)
    arguments = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with arguments.comparison.open() as handle:
        rows = list(csv.DictReader(handle))
    valid_rows = [row for row in rows if row.get("valid_comparison") == "True"]
    if not valid_rows:
        raise SystemExit(
            "No validated results to plot; "
            "incomplete results are not performance evidence."
        )
    labels = {
        "vllm-direct-service": "Direct vLLM",
        "thunderagent-program-throttling": "ThunderAgent program throttling",
    }
    styles = [("#0072B2", "o", "-"), ("#D55E00", "s", "--")]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for index, configuration in enumerate(
        sorted({row["configuration"] for row in valid_rows})
    ):
        points = sorted(
            [row for row in valid_rows if row["configuration"] == configuration],
            key=lambda row: float(row["rate_programs_per_second"]),
        )
        for axis, metric, label in zip(
            axes.flat,
            [
                "makespan_seconds",
                "program_latency_mean_seconds",
                "program_latency_p90_seconds",
                "steps_per_minute",
            ],
            [
                "Whole-batch makespan (seconds)",
                "Mean program latency (seconds)",
                "P90 program latency (seconds)",
                "Completed model calls per minute",
            ],
            strict=True,
        ):
            axis.plot(
                [float(row["rate_programs_per_second"]) for row in points],
                [float(row[metric]) for row in points],
                color=styles[index % len(styles)][0],
                marker=styles[index % len(styles)][1],
                linestyle=styles[index % len(styles)][2],
                label=labels.get(configuration, configuration),
            )
            axis.set_xlabel("Program arrival rate (programs/second)")
            axis.set_ylabel(label)
            axis.set_ylim(0, max(float(row[metric]) for row in valid_rows) * 1.15)
            axis.set_xticks(
                sorted({float(row["rate_programs_per_second"]) for row in valid_rows})
            )
            axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=9)
    smoke = "smoke" in str(arguments.comparison.parent)
    title = arguments.title or (
        "Pipeline validation only: no performance conclusion"
        if smoke
        else "Fixed workload: single-seed pilot (validated points only)"
    )
    figure.suptitle(title, fontsize=12)
    output = arguments.comparison.parent / "figures"
    output.mkdir(exist_ok=True)
    for extension in ["png", "pdf"]:
        figure.savefig(output / ("fixed_program_rate_sweep." + extension), dpi=180)
    (output / "figure_metadata.json").write_text(
        json.dumps(
            {
                "title": title,
                "source": str(arguments.comparison.resolve()),
                "validated_rows": len(valid_rows),
                "omitted_rows": len(rows) - len(valid_rows),
                "axis_minimum": 0,
                "uncertainty": "single run; no confidence intervals",
                "series_labels": labels,
            },
            indent=2,
        )
        + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
