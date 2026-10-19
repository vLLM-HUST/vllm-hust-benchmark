"""Render the measured dataset supplement from independently recomputed values."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def render(data, runtime):
    a, b = data["b0"], data["b1"]
    rows = [
        (
            "请求数 / 失败",
            f"{a['requests']} / {a['failures']}",
            f"{b['requests']} / {b['failures']}",
        ),
        (
            "实际输入 / 输出token",
            f"{a['input_tokens']:,} / {a['output_tokens']:,}",
            f"{b['input_tokens']:,} / {b['output_tokens']:,}",
        ),
        (
            "测量时长，秒",
            f"{a['benchmark_duration_s']:.3f}",
            f"{b['benchmark_duration_s']:.3f}",
        ),
        ("输出吞吐，token/秒", f"{a['output_tps']:.2f}", f"{b['output_tps']:.2f}"),
    ]
    for key, label, scale in [
        ("ttft_s", "TTFT，秒", 1),
        ("tpot_s", "TPOT，毫秒/token", 1000),
    ]:
        for stat, title in [("mean", "均值"), ("p95", "P95"), ("p99", "P99")]:
            rows.append(
                (
                    label + "，" + title,
                    f"{a[key][stat] * scale:.3f}",
                    f"{b[key][stat] * scale:.3f}",
                )
            )
    for key, label in [
        ("reserved_card_seconds_per_output_token", "请求区间4卡预约，卡秒/token"),
        ("allocated_card_seconds_per_output_token", "请求区间2卡服务，卡秒/token"),
    ]:
        rows.append((label, f"{a[key]:.6f}", f"{b[key]:.6f}"))
    rows.append(
        (
            "完整生命周期4卡预约，卡秒/token",
            f"{2 * a['lifecycle_allocated_card_seconds_per_output_token']:.6f}",
            f"{2 * b['lifecycle_allocated_card_seconds_per_output_token']:.6f}",
        )
    )
    rows.append(
        (
            "卡4 / 卡5平均AI Core忙碌率",
            " / ".join(
                f"{a['device_utilization'][str(i)]['mean_ai_core_percent']:.2f}%"
                for i in [4, 5]
            ),
            " / ".join(
                f"{b['device_utilization'][str(i)]['mean_ai_core_percent']:.2f}%"
                for i in [4, 5]
            ),
        )
    )
    table = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    packages = runtime["packages"]
    changes = [
        ("M02 输出吞吐提升", "output_throughput_change_percent", 10),
        ("M03 平均TTFT下降", "mean_ttft_reduction_percent", 5),
        ("M04 平均TPOT下降", "mean_tpot_reduction_percent", 5),
        ("M06 请求区间资源成本下降", "resource_cost_reduction_percent", 10),
    ]
    gates = "\n".join(
        f"| {label} | {data[key]:.2f}% | ≥{threshold}% | {'达到' if data[key] >= threshold else '未达到'} |"
        for label, key, threshold in changes
    )
    return f"""### 5.2 MMLU-Pro同机配对补测

测试对象为vLLM-HUST，模型为Qwen3.5-35B-A3B BF16。运行编号为B0
`mmlu-b0-r2`、B1 `mmlu-b1-r2`，按B0、B1顺序运行，各一个TP2服务。
物理卡4、5执行推理，卡6、7在整个请求区间均空闲；实验预约4卡。
本节主资源成本包含全部4张预约卡，并列2卡服务口径。

两臂使用相同运行栈：vLLM `{packages["vllm"]}`、Ascend
`{packages["vllm-ascend"]}`、Torch `{packages["torch"]}`、torch-npu
`{packages["torch-npu"]}`。完整环境及33项源码校验记录见`aligned-runtime.json`。
B0沿用原数据集基线：EP开启、APC关闭、无MTP、FULL_DECODE_ONLY、上下文32768、
批处理预算8192、16个执行席位。B1沿用学校栈优化配置：EP关闭、APC开启、MTP2、FULL、
上下文65536、批处理预算4096、E16/R36、每rank 20 GiB State及8 GiB host cache。
两臂使用相同的Triton包名识别修复；B1 host适配库由原源码重新编译，数值设备库未变。

输入为原MMLU-Pro性能测试的200条提示，JSONL与原文件SHA256完全一致。
使用原`vllm bench serve`协议、无限到达率、256输出token上限、temperature=0、seed=0，
保留服务默认thinking设置及客户端单请求预检查。时延为客户端SSE分块计时；
TTFT包含突发排队，TPOT按原客户端定义计算。本次为单次配对，未重新评分256-token输出。
P95/P99采用最近秩法，各臂n=200，对应第190、198个排序样本；原客户端线性分位数另行保留。

| 指标 | B0 | vLLM-HUST B1 |
| --- | ---: | ---: |
{table}

| 指标 | 配对变化 | 内部阈值 | 本工况判定 |
| --- | ---: | ---: | --- |
{gates}

P99 TPOT增加{data["p99_tpot_change_percent"]:.2f}%，完整生命周期资源成本增加
{data["lifecycle_resource_cost_change_percent"]:.2f}%。生命周期区间为控制器启动至设备释放，
包含加载、编译、客户端初始化和关闭，不与请求测量区间混用。
设备采样覆盖B0全部{a["request_span_s"]:.3f}秒及B1全部{b["request_span_s"]:.3f}秒请求区间，
未覆盖时间为0；AI Core忙碌率不作为有效FLOP达成率。

B1记录{b["speculative_decode"]["spec_decode_num_drafts"]:,}次draft、
{b["speculative_decode"]["spec_decode_draft_tokens"]:,}个draft token、
{b["speculative_decode"]["spec_decode_accepted_tokens"]:,}个接受token，接受率
{b["speculative_decode"]["spec_decode_acceptance_rate"]:.2f}%。两臂客户端及服务退出码均为0，
设备均已释放；B1关闭时保留了共享内存及semaphore清理告警。

证据为`aligned-mmlu-common.tar.gz`、`aligned-mmlu-b0.tar.gz`和`aligned-mmlu-b1.tar.gz`。
`verify_aligned_pair.py`从逐请求记录复算`aligned-mmlu-results.json`，并核对源码、输入及资源释放状态。
"""


def main():
    data = json.loads((ROOT / "aligned-mmlu-results.json").read_text())
    runtime = json.loads((ROOT / "aligned-runtime.json").read_text())
    section = render(data, runtime)
    path = ROOT / "中期测试报告.md"
    report = path.read_text()
    begin, end = "<!-- BEGIN ALIGNED_MMLU -->", "<!-- END ALIGNED_MMLU -->"
    assert report.count(begin) == report.count(end) == 1
    prefix, rest = report.split(begin)
    _, suffix = rest.split(end)
    path.write_text(prefix + begin + "\n\n" + section + "\n" + end + suffix)


if __name__ == "__main__":
    main()
