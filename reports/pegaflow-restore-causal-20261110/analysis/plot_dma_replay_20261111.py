"""Plot full-output replay and separately labelled short-output diagnostics."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


LABELS = {
    "random-dma-stage-4sessions-rps3-s17": "随机 D，原输出",
    "sticky-dma-stage-4sessions-rps3-s17": "固定 D，原输出",
    "random-dma-repeat-4sessions-rps3-s17": "随机 D，同条件复测",
    "random-dma-cap16-4sessions-rps3-s17": "随机 D，4 会话，输出≤16",
    "random-dma-cap16-6sessions-rps3-s17": "随机 D，6 会话，输出≤16",
    "random-dma-cap16-4sessions-dinflight1-rps3-s17": "随机 D，4 会话，D 前端=1",
}
STAGES = [
    ("P_queue_s", "P 前端排队", "#adc5d5"),
    ("source_budget_wait_s", "P 源预算等待", "#b8b1d8"),
    ("P_service_s", "P 服务", "#4779a7"),
    ("publication_wait_s", "保存完成等待", "#e7b75e"),
    ("D_queue_s", "D 接入前预算/排队", "#be886a"),
    ("D_to_first_s", "D 接入至首 token", "#63a59c"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--font", default="C:/Windows/Fonts/msyh.ttc")
    args = parser.parse_args()
    root = args.root
    summary = json.loads((root / "analysis/model-replay-summary.json").read_text())
    selected = [r for r in summary["runs"] if r["label"] in LABELS and r["complete"]]
    selected.sort(key=lambda r: list(LABELS).index(r["label"]))
    font = font_manager.FontProperties(fname=args.font)
    plt.rcParams.update(
        {
            "font.family": font.get_name(),
            "axes.unicode_minus": False,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "font.size": 10,
        }
    )
    colors = ["#3272b7", "#448f71", "#9169b3", "#db963f", "#c15e52", "#4a9498"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), layout="constrained")
    for index, run in enumerate(selected):
        name = run["label"]
        records = [
            json.loads(x)
            for x in (root / "raw/runs" / name / "requests.jsonl")
            .read_text()
            .splitlines()
        ]
        values = sorted(
            r["first_s"] - r["ready_s"] for r in records if r["status"] == "ok"
        )
        target = axes[0, 1] if "cap16" in name else axes[0, 0]
        target.step(
            [100 * (i + 1) / len(values) for i in range(len(values))],
            values,
            where="post",
            color=colors[index],
            label=f"{LABELS[name]}（n={len(values)}）",
            linestyle="--" if "repeat" in name else "-",
            linewidth=1.8,
        )
    for ax, title in zip(
        axes[0], ["完整轨迹输出：TTFT 百分位", "短输出诊断：TTFT 百分位"]
    ):
        ax.set(
            title=title,
            xlabel="请求百分位（从快到慢）",
            ylabel="TTFT（秒）",
            xlim=(0, 100),
            ylim=(0, None),
        )
        ax.grid(alpha=0.17)
        ax.legend(frameon=False, fontsize=8)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylim(0.1, None)
    axes[0, 1].set_ylabel("TTFT（秒，对数刻度）")
    left = [0.0] * len(selected)
    for key, label, color in STAGES:
        values = [r["stages"][key]["mean"] for r in selected]
        axes[1, 0].barh(
            range(len(selected)), values, left=left, label=label, color=color
        )
        left = [a + b for a, b in zip(left, values)]
    axes[1, 0].set_yticks(
        range(len(selected)), [LABELS[r["label"]] for r in selected], fontsize=8
    )
    axes[1, 0].invert_yaxis()
    axes[1, 0].set(title="平均首 token 等待分解", xlabel="秒")
    axes[1, 0].legend(
        frameon=False,
        fontsize=8,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
    )
    ax = axes[1, 1]
    rates = [r["completed_rps"] for r in selected]
    ax.bar(range(len(selected)), rates, color=colors[: len(selected)], width=0.65)
    ax.set_xticks(
        range(len(selected)),
        [LABELS[r["label"]].replace("，", "\n") for r in selected],
        fontsize=8,
    )
    ax.set(title="完整条件的请求吞吐", ylabel="完成请求 / 秒")
    for i, (rate, run) in enumerate(zip(rates, selected)):
        ax.text(
            i,
            rate,
            f"{rate:.3f}\n{run['output_tokens_s']:.1f} tok/s",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.margins(y=0.23)
    ax.grid(axis="y", alpha=0.17)
    fig.suptitle("DMA 就绪双缓冲：真实 2P2D 请求重放", fontsize=16, fontweight="bold")
    fig.supxlabel(
        "14B BF16 · 24GiB CPU 池/实例 · graph · 单轮观测；短输出组为机制诊断，后续输入采用固定轨迹历史。",
        fontsize=9,
    )
    fig.savefig(root / "analysis/model-replay-overview.png", dpi=170)
    plt.close(fig)
    print(root / "analysis/model-replay-overview.png")


if __name__ == "__main__":
    main()
