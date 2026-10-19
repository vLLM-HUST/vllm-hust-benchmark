"""Render measured follow-up results and per-metric B1 selection together."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BEGIN = "<!-- BEGIN B1_FOLLOWUP -->"
END = "<!-- END B1_FOLLOWUP -->"


def render():
    pair = json.loads((ROOT / "aligned-mmlu-results.json").read_text())
    arms = {"E16/R36": pair["b1"]}
    screens = {}
    for key, label in [
        ("e32", "E32/R36"),
        ("e24", "E24/R36"),
        ("kvmat", "E16/R36 + KV materialization"),
    ]:
        screen = json.loads((ROOT / f"b1-{key}-screen.json").read_text())
        screens[key] = screen
        if screen["status"] == "measured":
            arms[label] = screen["measured"]
    best = {
        "output_throughput": max(arms, key=lambda key: arms[key]["output_tps"]),
        "mean_ttft": min(arms, key=lambda key: arms[key]["ttft_s"]["mean"]),
        "mean_tpot": min(arms, key=lambda key: arms[key]["tpot_s"]["mean"]),
    }
    lines = [
        BEGIN,
        "### 5.4 E24/R36与KV materialization同工况补测",
        "",
        "沿用5.2节的MMLU-Pro 200请求、同一输入哈希、模型修订、BF16、TP2及学校运行栈，保持每卡20 GiB状态预算、8 GiB host缓存和R36。候选使用APC、async scheduling、MTP2及FULL图；执行宽度和KV控制分别记录。每轮独立服务，四卡保留、两卡执行。",
        "",
        "| 配置 | 成功/失败 | 输出吞吐 token/秒 | 相对B0 | 平均TTFT 秒 | 平均TPOT 毫秒/token | TPOT P99 毫秒/token |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, result in {"Native B0": pair["b0"], **arms}.items():
        gain = 100 * (result["output_tps"] / pair["b0"]["output_tps"] - 1)
        lines.append(
            f"| {label} | {result['requests']}/{result['failures']} | {result['output_tps']:.6f} | {gain:+.2f}% | {result['ttft_s']['mean']:.6f} | {1000 * result['tpot_s']['mean']:.6f} | {1000 * result['tpot_s']['p99']:.6f} |"
        )
    lines += [
        "",
        f"本工况当前已测吞吐B1为{best['output_throughput']}；平均TTFT和平均TPOT的B1分别为{best['mean_ttft']}、{best['mean_tpot']}。各指标对应配置独立，不合并为单一部署的联合结果。",
        "",
    ]
    for kind in ("e24", "kvmat"):
        screen = screens[kind]
        lines.append(
            f"运行`{screen['run_id']}`：状态`{screen['status']}`；原始证据见`b1-{kind}-screen-evidence.tar.gz`，复算结果见`b1-{kind}-screen.json`。"
        )
        if screen["status"] == "measured":
            lines.append(
                f"较E16/R36吞吐变化{screen['throughput_change_vs_prior_pct']:+.2f}%。256-token输出与E16/R36逐请求全文一致数为{screen['bounded_output_comparison']['exact_matches_vs_e16']}/200；该比较不是答案正确率。"
            )
        lines.append("")
    if screens["kvmat"]["status"] == "measured":
        control = screens["kvmat"]["control_evidence"]
        lines += [
            f"KV materialization采用固定源码的0.28受限适配；原请求未添加复用锚点。记录{control['policy_observations']}次策略观测、{control['engine_salt_admissions']}次引擎缓存盐接收，盐摘要逐一匹配，输入合计{control['engine_prompt_tokens']:,} token。控制范围为逐请求重算；分段复用为`not-exercised`，缓存盐接收不等于复用或节省计算。",
            "",
        ]
    lines += [
        "本轮为单次独立候选筛选，结果描述已测服务性能，不代表多次重复的统计显著性或任务质量等价。每轮输出均为51,200 token；原始SSE计时、MTP计数、设备采样、启动与释放回执及源码哈希随归档保留。任务正确率未重新评分。",
        END,
    ]
    for key, label in [("e24", "E24/R36"), ("kvmat", "E16/R36 + KV materialization")]:
        screens[key]["selected_as_b1"] = best["output_throughput"] == label
        screens[key]["selection_scope"] = (
            "Current measured output throughput among E16, E32, E24 and KV candidates; task accuracy not rescored"
        )
        (ROOT / f"b1-{key}-screen.json").write_text(
            json.dumps(screens[key], indent=2) + "\n"
        )
    (ROOT / "b1-current-best.json").write_text(
        json.dumps(
            {
                "scope": "MMLU-Pro 200 inf out256 school 0.28 service performance",
                "selection": best,
                "measurements": arms,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    report = ROOT / "中期测试报告.md"
    text = report.read_text().replace("报告版本：V0.6", "报告版本：V0.7")
    section = "\n".join(lines)
    if BEGIN in text:
        text = text[: text.index(BEGIN)] + section + text[text.index(END) + len(END) :]
    else:
        text = text.replace("## 六、有效算力结果", section + "\n\n## 六、有效算力结果")
    report.write_text(text)


if __name__ == "__main__":
    render()
