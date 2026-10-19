"""Plot verified real-Agent batch time and work using the existing plot palette."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from plot_run import BLUE, ORANGE, style


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = json.loads(arguments.report.read_text())
    pairs = sorted({x["pair"] for x in report["batches"]})
    assert 1 <= len(pairs) <= 3
    assert all(sum(x["pair"] == pair and x["arm"] == arm for x in report["batches"]) == 1
               for pair in pairs for arm in ["native", "controller"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for arm, color, offset in [("native", BLUE, -0.19), ("controller", ORANGE, 0.19)]:
        batches = sorted((x for x in report["batches"] if x["arm"] == arm), key=lambda x: x["pair"])
        for axis, values in [
            (axes[0], [x["agent_batch_elapsed_seconds"] for x in batches]),
            (axes[1], [x["usage"]["completion_tokens"] for x in batches]),
        ]:
            bars = axis.bar([x + offset for x in range(len(pairs))], values, width=0.36,
                            label=arm.title(), color=color, zorder=3)
            axis.bar_label(bars, labels=[f"{x:,.1f}" if axis is axes[0] else f"{x:,}" for x in values],
                           padding=3, fontsize=8)
    for axis, label in zip(axes, ["Complete Agent-batch time (seconds)", "Actual output work (tokens)"]):
        style(axis, label)
        axis.set_xticks(range(len(pairs)), [f"Pair {pair}\n" + (
            "Controller → Native" if pair == 2 else "Native → Controller") for pair in pairs])
        axis.set_ylim(bottom=0, top=axis.get_ylim()[1] * 1.12)
        axis.legend(frameon=False, loc="upper left", ncol=2)
    counts = sorted({x["observed_terminal_tasks"] for x in report["batches"]})
    assert len(counts) == 1
    figure.suptitle(f"Same {counts[0]} real Agent instances, same unified HUST model service", fontsize=13)
    figure.text(0.5, 0.025,
                "Tool waits included; grading excluded. Work and terminal paths differ.\n"
                "These observed times do not establish a Controller scheduling benefit.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.10, 1, 0.94))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "pdf"]:
        figure.savefig(arguments.output.with_suffix("." + extension), dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
