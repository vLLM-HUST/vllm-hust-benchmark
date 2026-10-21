"""Render recovered model measurements from the portable evidence package."""

import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent


def render():
    base = ROOT / "m01-recovery/model-runtime-20261021"
    index = json.loads((base / "archive.json").read_text())
    rows = []
    totals = []
    with tarfile.open(base / index["archive"]) as archive:
        for run in index["runs"]:
            hardware = json.load(archive.extractfile(run["hardware"]))
            windows = {window["name"]: window for window in hardware["windows"]}
            for name in [
                "first_request_to_last_response",
                "first_request_to_last_forward_event_outer",
            ]:
                assert all(
                    device["minimum_channel_coverage"] == 1
                    for device in windows[name]["hbm_devices"]
                )
            coverage = min(
                device["minimum_channel_coverage"]
                for device in windows["first_request_to_all_export_synchronizations"][
                    "hbm_devices"
                ]
            )
            rows.append(
                f"| {run['hardware_sampling_seconds']} | {run['completed']} / {run['failed']} | "
                f"{run['output_tokens']} | {run['client_duration_seconds']:.6f} | "
                f"{run['instrumented_output_tokens_per_second']:.6f} | {100 * coverage:.6f}% |"
            )
            totals.append(str(run["basic_operation_subtotal"]))
    assert len(rows) == 2
    return "\n".join(
        [
            "#### 恢复环境同工况计量",
            "",
            "测试对象为 Qwen3.5-35B-A3B BF16、TP2、E24/R36、KV materialization 重算路径与 FA Microflow，"
            "上下文 65536、每卡 KV 20 GiB、主机状态缓存 8 GiB、MTP 2 tokens、FULL 图模式。"
            "恢复环境使用 Triton-Ascend `3.6.0+dev20261007182438`，FIA 使用上述 CANN 9.1.0 重建产物；"
            "模型 Python 源码未修改，重建运行时的二进制产物与原运行时不同。",
            "",
            "该环境完成双卡 GDN 递推的 eager 与变输入图重放验证，以及两卡各 24 项状态提交用例。"
            "两轮完整模型测量均使用新进程、新输出目录和空编译缓存，不继承上一轮推理状态。"
            "两卡 Microflow 在 1024、2048、4096 token 分区检查中逐项相等，34 档 FULL 图捕获完成。",
            "",
            "工作负载为原 MMLU-Pro 200 请求、输出上限 256、无限到达率、temperature=0。"
            "本节为计量插桩运行，吞吐为该轮观测值，不替换第五章未插桩性能结果。",
            "",
            "| 硬件采样时长（秒） | 成功 / 失败请求 | 输出 token | 请求时长（秒） | 诊断吞吐（token/秒） | 全导出同步区间 HBM 最小覆盖率 |",
            "| --- | --- | --- | --- | --- | --- |",
            *rows,
            "",
            "两轮的请求区间与末次前向事件区间 HBM 计数均完整覆盖；360 秒采样另覆盖至全部导出同步完成。"
            "HCCS 按两卡 TX 求和，不叠加 RX。客户端和硬件采集器均退出 0，服务关闭后四张 NPU 均已释放。",
            "",
            f"两轮请求区间的十类基础运算分项小计分别为 {totals[0]} 和 {totals[1]}。"
            "该小计不是完整有效 F；跨流依赖、完整算术覆盖及同形状、精度、状态访问和拓扑下的并发参考容量仍需核验。"
            "M01 达成率与中期整体验收结论保持未判定。",
            "",
            "原始请求、设备记录、硬件数据库、源码及逐文件 SHA256 见"
            "[恢复计量证据](m01-recovery/model-runtime-20261021/archive.json)。",
            "",
        ]
    )


if __name__ == "__main__":
    report = ROOT / "中期测试报告.md"
    text = report.read_text()
    assert "#### 恢复环境同工况计量" not in text
    assert "报告版本：V0.52" in text
    text = text.replace("报告版本：V0.52", "报告版本：V0.53", 1)
    text = text.replace(
        "#### r21 设备映射与同轮计量", render() + "\n#### r21 设备映射与同轮计量", 1
    )
    old = (
        "运行时包版本及二进制恢复尚未完成，完整有效 F、匹配容量及 M01\n达成率仍未判定。"
    )
    new = "完整模型恢复测量见下文“恢复环境同工况计量”；完整有效 F、匹配容量及 M01\n达成率仍未判定。"
    assert old in text
    report.write_text(text.replace(old, new, 1))
