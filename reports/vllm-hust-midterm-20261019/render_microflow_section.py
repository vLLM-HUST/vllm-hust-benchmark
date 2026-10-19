"""Render the three independent matched screens without selecting a peak run."""

from pathlib import Path

from microflow_candidate import load


ROOT = Path(__file__).resolve().parent
BEGIN = "<!-- BEGIN MICROFLOW_SCREEN -->"
END = "<!-- END MICROFLOW_SCREEN -->"


def render():
    summary, _ = load()
    lines = [
        BEGIN,
        "### 5.5 Prefill Microflow FA同工况配对复测",
        "",
        "测试对象为vLLM-HUST E24/R36 + KV逐请求重算方案，开关项仅为target全注意力层的Prefill Microflow分片调度。模型、BF16、TP2、学校运行栈、每卡20 GiB状态与8 GiB host缓存、APC、MTP2、FULL图和async scheduling保持不变。GDN及MTP draft沿用原路径。",
        "",
        "采用原MMLU-Pro 200请求、无限到达率及out256。三组OFF/ON分别独立启动，顺序执行；四卡保留、两卡执行。Triton/Inductor缓存由同一份既有B1缓存快照复制到各自独立目录，vLLM AOT缓存独立初始化；每轮保存初始缓存清单摘要。统计量采用三次运行的中位数与完整范围，不选择最高轮次。",
        "",
        "| 组次 | Microflow | 成功/失败 | 输出吞吐 token/秒 | 平均TTFT 秒 | 平均TPOT 毫秒/token |",
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
            values = summary["aggregates"][mode][key]
            cells.append(
                f"{values['median'] * scale:.6f}〔{values['min'] * scale:.6f}, {values['max'] * scale:.6f}〕"
            )
        lines.append(f"| {label} | {' | '.join(cells)} |")
    lines += [
        "",
        f"三组配对吞吐比的几何平均变化为{summary['on_vs_off_throughput_pct']:+.4f}%；ON吞吐中位数较原B1的{summary['current_best_per_metric']['output_throughput']:.6f} token/秒变化{summary['on_vs_current_throughput_b1_pct']:+.4f}%。样本量为三组独立启动，小幅差异不代表统计显著性或跨工况稳定收益。各指标的当前最优配置按5.4节独立选取。",
        "",
        "每个ON窗口的两rank均通过1K/2K/4K同分片串行与并发计算的逐元素一致检查，并记录10个target FA层的图捕获及请求窗口内对应形状的真实target图重放。每轮200条KV策略记录与200条引擎缓存盐接收记录逐一匹配。分段KV复用与GDN Microflow均未执行。全部请求成功，所有服务退出后均确认设备释放。",
        "",
        "本节为服务性能筛选，256-token输出不是完整MMLU-Pro答题评分，任务正确率未重新评测，也不构成完整GDN+FA Microflow的任务质量验收。空编译缓存及组合未启用的诊断窗口单独归档，不混入上述三组统计。",
        "",
        "证据：[三组复算汇总](microflow-school028/summary.json)、[重复测试口径](microflow-school028/repetition-plan.json)、[原始归档与复现说明](microflow-school028/README.md)。各轮保留请求输出、token计时、MTP计数、APC记录、控制动作、设备采样、窗口边界、退出回执及源码与文件SHA256。",
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
