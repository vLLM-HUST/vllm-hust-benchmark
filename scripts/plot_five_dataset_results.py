"""Plot only audited public metrics, retaining every dataset and unresolved grade."""

import argparse
import json
from pathlib import Path
import matplotlib

if matplotlib.__version__ != "3.10.8":
    raise RuntimeError("Require matplotlib3.10.8")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 - version/backend pinned first

matplotlib.rcParams["svg.hashsalt"] = "sealed-five-datasets-20261113-v1"
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--summary", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
d = json.loads(a.summary.read_text())
a.out.mkdir(parents=True, exist_ok=True)
labels = ["MMLU200", "HLE200", "Frontier160", "SWE trace32", "Terminal trace32"]
x = list(range(5))
colors = ["#64748b", "#2563eb"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for axis, key, title in zip(
    axes,
    ["output_tokens_per_second", "mean_ttft_ms", "mean_tpot_ms"],
    [
        "Solver output tokens/s",
        "Mean TTFT (ms, log scale)",
        "Mean TPOT (ms, log scale)",
    ],
):
    for index, role in enumerate(["B0", "B1"]):
        axis.bar(
            [n + (index - 0.5) * 0.36 for n in x],
            [ds["arms"][role][key] for ds in d["datasets"]],
            width=0.36,
            color=colors[index],
            label=role,
        )
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    if key != "output_tokens_per_second":
        axis.set_yscale("log")
fig.suptitle(
    "All five frozen profiles · paired native evidence; "
    f"{d['reused_stages']} verified reused stages, {d['new_inference_stages']} new native stages"
)
fig.tight_layout()
fig.savefig(a.out / "throughput-latency.png", dpi=170)
fig.savefig(a.out / "throughput-latency.svg", metadata={"Date": None})
plt.close(fig)
fig, axis = plt.subplots(figsize=(10, 4.6))
labels = []
success = []
unresolved = []
other = []
for ds in d["datasets"][:3]:
    for role in ["B0", "B1"]:
        q = ds["quality"][role]
        n = ds["tasks_per_arm"]
        s = (
            q["correct"]
            if ds["id"] == "mmlu"
            else sum(g["successes"] for g in q["tracks"].values())
        )
        u = (
            0
            if ds["id"] == "mmlu"
            else sum(
                g.get("unresolved", g.get("judge_unresolved"))
                for g in q["tracks"].values()
            )
        )
        labels.append(ds["name"] + " " + role)
        success.append(100 * s / n)
        unresolved.append(100 * u / n)
        other.append(100 * (n - s - u) / n)
axis.bar(
    labels, success, label="Observed success / planned denominator", color="#166534"
)
axis.bar(
    labels, unresolved, bottom=success, label="Unresolved (retained)", color="#d97706"
)
axis.bar(
    labels,
    other,
    bottom=[s + u for s, u in zip(success, unresolved)],
    label="Other nonsuccess",
    color="#cbd5e1",
)
axis.set_ylim(0, 100)
axis.set_ylabel("Percent of planned tasks")
axis.set_title(
    "Original graders; unresolved are not confirmed incorrect\nSWE/Terminal trace inference has no task-quality score"
)
axis.tick_params(axis="x", labelrotation=20)
axis.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(a.out / "quality-coverage.png", dpi=170)
fig.savefig(a.out / "quality-coverage.svg", metadata={"Date": None})
plt.close(fig)
print("PASS: paired performance and denominator-preserving quality plots")

# Canonical SVG copies follow the repository text-format rules.
for svg in a.out.glob("*.svg"):
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
