"""Render the fixed diagnostic summaries; no hardware work is performed."""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/msyh.ttc"))
    args = parser.parse_args()
    if args.font.exists():
        font_manager.fontManager.addfont(str(args.font))
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=args.font
        ).get_name()
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    root = args.root
    first = json.loads(
        (root / "analysis/checkpoint-1810-summary.json").read_text(encoding="utf-8")
    )
    data = json.loads(
        (root / "analysis/followup-1837-summary.json").read_text(encoding="utf-8")
    )["hosts"]["hust-zsx-81"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.subplots_adjust(top=0.83, bottom=0.10, hspace=0.52, wspace=0.30)
    fig.suptitle(
        "本地缓存拷贝归因：固定数据的控制与干预",
        fontsize=19,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.93,
        "Ascend 910B2 / CANN 9.1.0    ·    2026-11-10",
        ha="center",
        color="#526070",
    )
    ax = axes[0, 0]
    for host, color, label in [
        ("hust-zsx", "#d87729", "原宿主机 0.2"),
        ("hust-zsx-81", "#2563bb", "81（宿主机 0.4）"),
    ]:
        rows = sorted(
            (
                x
                for x in first["hosts"][host]["descriptor_sweep"][
                    "descriptor-v1-mapped"
                ]
                if x["direction"] == 0
            ),
            key=lambda x: x["descriptors"],
        )
        ax.errorbar(
            [x["descriptors"] for x in rows],
            [x["median_s"] for x in rows],
            yerr=[
                [x["median_s"] - x["min_s"] for x in rows],
                [x["max_s"] - x["median_s"] for x in rows],
            ],
            color=color,
            marker="o",
            linewidth=1.6,
            capsize=4,
            label=label,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([8, 64, 512, 4096, 16384], ["8", "64", "512", "4096", "16384"])
    ax.set_xlabel("描述符数量")
    ax.set_ylabel("H2D 完成时间（秒，对数）")
    ax.set_title(
        "A  固定 1GiB：任务数量影响\nMAPPED；中位数与范围，每条件 3 次",
        loc="left",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    colors = ["#64748b", "#2563bb", "#159877"]
    ax = axes[0, 1]
    for i, mode in enumerate([0, 10, 11]):
        values = [
            x["complete_s"]
            for x in data["native_stripe"]
            if x["policy"] == mode and x["phase"].startswith("stage")
        ]
        for j, value in enumerate(values):
            ax.scatter(
                i + (j - (len(values) - 1) / 2) * 0.07, value, s=47, color=colors[i]
            )
        ax.hlines(
            statistics.median(values),
            i - 0.20,
            i + 0.20,
            color=colors[i],
            linewidth=2.5,
        )
        ax.text(i, 0.60, f"n={len(values)}", ha="center", fontsize=9, color="#526070")
    ax.set_ylim(0, 0.65)
    ax.set_xticks([0, 1, 2], ["原始单 stream", "16 条 stream", "32 条 stream"])
    ax.set_ylabel("恢复完成时间（秒）")
    ax.set_title(
        "B  81 原生 IPC 恢复\n3.77GiB；15,456 描述符；目标映射相同",
        loc="left",
        fontsize=11,
    )
    ax = axes[1, 0]
    for i, mode in enumerate([0, 10, 11]):
        values = [
            x["TTFT_s"]
            for x in data["model_stripe"]
            if x["fixture"] == 2 and x["phase"] != "seed" and x["policy"] == mode
        ]
        for j, value in enumerate(values):
            pos = i + (j - (len(values) - 1) / 2) * 0.10
            ax.scatter(pos, value, s=55, color=colors[i])
            ax.annotate(
                f"{value:.3f}",
                (pos, value),
                xytext=(0, 8 if j == len(values) - 1 else -16),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
    ax.set_ylim(0, 1.45)
    ax.set_xticks([0, 1, 2], ["原始（2 次）", "16 条（1 次）", "32 条（1 次）"])
    ax.set_ylabel("TTFT（秒）")
    ax.set_title(
        "C  81 完整模型：44,470-token 输入\n每次恢复 347 块 / 8.13GiB / 33,312 描述符",
        loc="left",
        fontsize=11,
    )
    ax = axes[1, 1]
    api, sync, cpu = [], [], []
    for mode in [0, 1, 2]:
        rows = [x for x in data["chain-control-v1"] if x["mode"] == mode]
        api.append(statistics.mean(x["api_ns"] for x in rows) / 1e6)
        sync.append(statistics.mean(x["sync_ns"] for x in rows) / 1e6)
        cpu.append(statistics.mean(x["cpu_ns"] for x in rows) / 1e6)
    ax.bar([0, 1, 2], api, color="#3174bc", label="提交 API")
    ax.bar([0, 1, 2], sync, bottom=api, color="#bad5ec", label="最后同步")
    ax.scatter(
        [0, 1, 2],
        cpu,
        color="#111827",
        marker="D",
        s=28,
        label="调用线程 CPU",
        zorder=4,
    )
    ax.set_xticks([0, 1, 2], ["单 stream", "16 条独立", "16 条顺序依赖"])
    ax.set_ylabel("时间（毫秒）")
    ax.set_title(
        "D  分离“提交等待”和“搬运并行”\n固定 1GiB / 16,384 描述符；均值，每组 4 次",
        loc="left",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="upper right")
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.2)
    fig.text(
        0.5,
        0.035,
        "小样本用于归因；原宿主机的多 stream 干预待复测。CPU 标记只计调用线程。",
        ha="center",
        fontsize=10,
        color="#526070",
    )
    fig.savefig(root / "restore-interventions.png", dpi=165, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
