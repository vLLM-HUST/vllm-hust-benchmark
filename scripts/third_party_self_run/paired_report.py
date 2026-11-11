"""Recompute a paired diagnostic and keep four formal target decisions separate."""

import hashlib
from html import escape
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def gain(a, b, improvement=False):
    if a is None or b is None or a <= 0:
        return None
    return 100 * (1 - b / a) if improvement else 100 * (b / a - 1)


def build(root):
    data = {}
    configs = {}
    manifest_path = root / "execution-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for arm in ["B0", "B1"]:
        out = root / arm / "results"
        path = out / "diagnostic-summary.json"
        data[arm] = (
            json.loads(path.read_text())
            if path.exists()
            else {
                "planned": manifest.get("native_requests_per_arm", 10),
                "completed": 0,
                "groups": {},
            }
        )
        path = root / arm / "diagnostic-config.json"
        configs[arm] = json.loads(path.read_text()) if path.exists() else {}
    if all(configs.values()):
        if (
            configs["B0"].get("profile") != "baseline"
            or configs["B1"].get("profile") != "hust"
        ):
            raise ValueError(
                "Expected actual 0.18 baseline and HUST profiles; upstream stock cannot be relabelled B1"
            )
        if configs["B0"]["source_weights"] != configs["B1"]["source_weights"]:
            raise ValueError("Different model manifests; paired comparison rejected")
        if configs["B0"]["cards"] != configs["B1"]["cards"]:
            raise ValueError("Physical card sets differ; paired comparison rejected")
        if sha(root / "B0/diagnostic-workload.json") != sha(
            root / "B1/diagnostic-workload.json"
        ):
            raise ValueError("Frozen request workloads differ")
    paired = {}
    for group in ["mmlu-diagnostic", "context-4096", "context-16384"]:
        values = {a: data[a]["groups"].get(group, {}) for a in data}
        a, b = values["B0"], values["B1"]
        result = {
            "arms": values,
            "throughput_gain_pct": gain(a.get("output_tps"), b.get("output_tps")),
        }
        for key in ["ttft_ms", "tpot_ms"]:
            result[key + "_improvement_pct"] = gain(
                a.get(key, {}).get("mean"), b.get(key, {}).get("mean"), True
            )
        if a.get("output_tps") and b.get("output_tps"):
            result["equal_rate_card_time_reduction_pct"] = 100 * (
                1 - a["output_tps"] / b["output_tps"]
            )
        paired[group] = result
    reasons = {
        "3.4.1": "未采集与本轮同步时间对应的有效FLOPs及独立、适用、整链验证的参考算力；不能用AI Core利用率替代，也不借用历史59.7573%。",
        "3.4.2": "下列是真实双卡、固定请求的轻负载诊断。正式高并发多轮/工具调用、同开放到达序列、独立质量与SLO门槛尚未完成；数值跨过门槛不等于完整目标达标。",
        "3.4.3": "本次覆盖2个MMLU-Pro样本、4K/16K长度探测各4次；未执行五项数据集既定全任务与原生评分器，也未取得连续窗口/最小样本等正式阈值。",
        "3.4.4": "缺本机综合费用范围和费率依据，以及冻结的高负载利用率改善判据。下列卡时/输出token比是等费率假设下的资源代理，不能充当实际人民币账单。",
    }
    version_path = root / "version-check.json"
    version = json.loads(version_path.read_text()) if version_path.exists() else None
    if version and not version["passed"]:
        reasons = {
            key: "受测版本/实现身份预检未通过，未启动模型或应用补丁。详见version-check.json；"
            + reason
            for key, reason in reasons.items()
        }
    summary = {
        "schema": "vllm-hust-paired-diagnostic/v1",
        "data": data,
        "paired": paired,
        "formal_targets": {
            key: {"status": "unassessable", "reason": reason}
            for key, reason in reasons.items()
        },
        "formal_complete": False,
        "version_check": version,
        "native_complete": all(
            d["completed"] == d["planned"] and d.get("receipt_exported")
            for d in data.values()
        ),
    }
    (root / "paired-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    charts = root / "charts"
    charts.mkdir(exist_ok=True)
    figure_specs = [
        (
            "compute.png",
            "3.4.1 Effective compute",
            [],
            "Missing same-run FLOPs and eligible independent reference",
        ),
        (
            "throughput-latency.png",
            "3.4.2 Paired native diagnostic",
            [
                ("Output token/s", "output_tps", None),
                ("Mean native TTFT (ms)", "ttft_ms", "mean"),
                ("Mean TPOT (ms/token)", "tpot_ms", "mean"),
            ],
            None,
        ),
        (
            "stability.png",
            "3.4.3 Context-length diagnostic",
            [
                ("4K output token/s", "output_tps", "context-4096"),
                ("16K output token/s", "output_tps", "context-16384"),
            ],
            None,
        ),
        (
            "cost.png",
            "3.4.4 Cost evidence",
            [],
            "No qualified fee / high-load utilization comparison; proxy only",
        ),
    ]
    for filename, title, fields, missing in figure_specs:
        fig, axes = plt.subplots(
            max(1, len(fields)),
            figsize=(9, 2.5 * max(1, len(fields))),
            squeeze=False,
            layout="constrained",
        )
        fig.suptitle(title)
        if missing:
            axes[0, 0].axis("off")
            axes[0, 0].text(0.5, 0.5, missing, ha="center", va="center", wrap=True)
        for ax, (label, key, inner) in zip(axes[:, 0], fields):
            group = (
                inner
                if inner in ["context-4096", "context-16384"]
                else "mmlu-diagnostic"
            )
            observed = []
            for arm in ["B0", "B1"]:
                x = data[arm]["groups"].get(group, {}).get(key)
                if isinstance(x, dict):
                    x = x.get(inner)
                observed.append(x)
            for i, (arm, value) in enumerate(zip(["B0", "B1"], observed)):
                if value is not None:
                    ax.barh(i, value, color=["#8c9bac", "#18728e"][i])
                    ax.text(value, i, f" {value:.4g}", va="center")
                else:
                    ax.text(
                        0.5,
                        0.5 - i * 0.2,
                        arm + ": no valid measurement",
                        transform=ax.transAxes,
                        ha="center",
                    )
            ax.set_title(label, loc="left")
            ax.set_yticks([0, 1], ["B0", "B1"])
            ax.grid(axis="x", alpha=0.2)
        fig.savefig(charts / filename, dpi=165)
        plt.close(fig)
    sections = []
    names = [
        "单芯片有效算力",
        "吞吐与时延",
        "五数据集及16K稳定性",
        "单位token成本与利用率",
    ]
    for (key, reason), title, (filename, *_) in zip(
        reasons.items(), names, figure_specs
    ):
        sections.append(
            f'<section><h2>{key} {title}：无法判定</h2><p>{escape(reason)}</p><img src="charts/{filename}"></section>'
        )
    measured = paired["mmlu-diagnostic"]
    observed = "".join(
        f"<tr><td>{escape(key)}</td><td>{'无有效配对' if value is None else format(value, '.4f') + '%'}</td></tr>"
        for key, value in measured.items()
        if key != "arms"
    )
    (root / "report.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>真实推理配对与四项目标</title>'
        "<style>body{max-width:1000px;margin:30px auto;font:16px/1.6 system-ui;padding:20px;color:#182331}img{width:100%}"
        "table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccd5de}section{margin-top:35px}</style>"
        "<h1>真实推理配对与四项目标</h1><p><strong>本次正式四项目标均未具备完整判定条件。</strong>原始失败、重试、缺失和实测值分别保留。"
        "此页不把轻负载诊断改称完整验收。</p>"
        f"<p>B0完成 {data['B0']['completed']}/{data['B0']['planned']}；B1完成 {data['B1']['completed']}/{data['B1']['planned']}。</p>"
        '<p><a href="B0/results/report.html">B0原始报告与失败日志</a> · <a href="B1/results/report.html">B1原始报告与失败日志</a> · '
        '<a href="paired-summary.json">全部原始量及复算JSON</a> · <a href="execution-manifest.json">执行清单</a></p>'
        "<h2>固定MMLU-Pro样本的诊断配对</h2><p>TTFT取原生首token延迟；TPOT从原生单调时钟首token至generate返回。"
        "按逐请求等权均值统计；吞吐分母含本组所有generate墙钟，不丢弃首请求或编译慢样本。"
        "资源卡时改善仅在两侧等费率、相同资源占用假设下解释。</p><table>"
        + observed
        + "</table>"
        + "".join(sections)
        + "</html>",
        encoding="utf-8",
    )
    manifest = {
        p.relative_to(root).as_posix(): sha(p)
        for p in root.rglob("*")
        if p.is_file()
        and not any(s in p.parts for s in ["cache", "__pycache__"])
        and p != root / "checksums.json"
    }
    (root / "checksums.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("PAIRED REPORT", root / "report.html")


if __name__ == "__main__":
    build(Path(sys.argv[1]).resolve())
