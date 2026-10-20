"""Render repeated same-workload local-argmax results."""

from pathlib import Path

from vspec_candidate import load

ROOT = Path(__file__).resolve().parent
BEGIN = "<!-- BEGIN VSPEC_SCREEN -->"
END = "<!-- END VSPEC_SCREEN -->"


def render():
    summary, _ = load()
    lines = [
        BEGIN,
        "### 5.6 MTP local argmax同工况配对复测",
        "",
        "测试对象为vLLM-HUST E24/R36 + KV逐请求重算 + Prefill Microflow FA方案，开关项仅为MTP草稿token选择：OFF使用完整logits，ON使用vSpec的TP局部argmax。模型、BF16、TP2、学校运行栈、每卡20 GiB状态与8 GiB host缓存、APC、原生MTP2、FULL图、async scheduling与Microflow保持不变。未启用自适应草稿长度或其他vSpec组件。",
        "",
        "采用同一MMLU-Pro 200请求、无限到达率、out256负载。三组OFF/ON顺序独立启动，各轮从同一编译缓存快照初始化，vLLM AOT缓存独立建立；四卡保留、两卡执行。资格检查不纳入三组统计，全部有效轮次均保留。",
        "",
        "| 组次 | Local argmax | 成功/失败 | 输出吞吐 token/秒 | 平均TTFT 秒 | 平均TPOT 毫秒/token |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["runs"]:
        m = row["metrics"]
        lines.append(
            f"| {row['pair_index']} | {row['mode'].upper()} | 200/0 | {m['output_throughput']:.6f} | {m['mean_ttft_ms'] / 1000:.6f} | {m['mean_tpot_ms']:.6f} |"
        )
    lines += [
        "",
        "| 指标 | OFF中位数〔范围〕 | ON中位数〔范围〕 |",
        "| --- | ---: | ---: |",
    ]
    for key, label, scale in [
        ("output_throughput", "输出吞吐 token/秒", 1),
        ("mean_ttft_ms", "平均TTFT 秒", 0.001),
        ("mean_tpot_ms", "平均TPOT 毫秒/token", 1),
    ]:
        cells = []
        for mode in ("off", "on"):
            v = summary["aggregates"][mode][key]
            cells.append(
                f"{v['median'] * scale:.6f}〔{v['min'] * scale:.6f}, {v['max'] * scale:.6f}〕"
            )
        lines.append(f"| {label} | {' | '.join(cells)} |")
    admitted = [
        label
        for key, label in [
            ("output_throughput", "吞吐"),
            ("mean_ttft_ms", "TTFT"),
            ("mean_tpot_ms", "TPOT"),
        ]
        if summary["improves_current_best_per_metric"][key]
    ]
    lines += [
        "",
        f"三组配对吞吐比的几何平均变化为{summary['on_vs_off_throughput_pct']:+.4f}%；ON吞吐中位数较既有Microflow B1的{summary['current_best_per_metric']['output_throughput']:.6f} token/秒变化{summary['on_vs_current_throughput_b1_pct']:+.4f}%。",
        "更新当前B1的指标："
        + ("、".join(admitted) if admitted else "无，保留既有最优配置")
        + "。三组结果不构成统计显著性或跨工况收益结论。",
        "",
        "双rank分别使用真实BF16模型头，在1/24/36/48行、零输入与正弦输入上检查完整logits和局部argmax输出一致性；正式窗口保留草稿选择算子的图捕获、真实草稿图重放、Microflow图重放及逐请求KV重算记录。每轮200请求成功、输出51200 tokens，服务退出后确认设备释放。",
        "",
        "本节为服务性能筛选，256-token输出不构成完整MMLU-Pro答题评分，任务正确率未重新评测。",
        "",
        "证据：[三组复算汇总](vspec-school028/summary.json)、[测试计划](vspec-school028/repetition-plan.json)、[原始归档与复现说明](vspec-school028/README.md)。",
        END,
    ]
    report = ROOT / "中期测试报告.md"
    text = report.read_text()
    section = "\n".join(lines)
    if BEGIN in text:
        text = text[: text.index(BEGIN)] + section + text[text.index(END) + len(END) :]
    else:
        text = text.replace("## 六、有效算力结果", section + "\n\n## 六、有效算力结果")
    report.write_text(text)


if __name__ == "__main__":
    render()
