"""Render the projection reference table from measured per-rank summaries."""

import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent


def render(data):
    rows = []
    for rank in (0, 1):
        dense = data["dense"]["ranks"][rank]
        rows.append(
            [
                "密集投影",
                rank,
                dense["cases"],
                dense["calls"],
                5,
                dense["sum_case_medians_weighted_ms"],
            ]
        )
        moe = data["moe"]["ranks"][rank]
        distributions, remainder = divmod(moe["timings"], 3 * len(moe["operators"]))
        assert remainder == 0
        for op in moe["operators"]:
            item = op["cohorts"]["inside"]
            rows.append(
                [
                    f"专家投影 {op['operator']}",
                    rank,
                    distributions,
                    item["calls"],
                    3,
                    item["sum_case_medians_weighted_ms"],
                ]
            )
        vocab = data["vocab"]["ranks"][rank]
        for role, label in (
            ("target", "目标模型词表投影"),
            ("draft", "草稿模型词表投影"),
        ):
            item = vocab["cohorts"][f"{role}:inside"]
            rows.append(
                [
                    label,
                    rank,
                    f"{vocab['shapes']}（两角色合计）",
                    item["calls"],
                    3,
                    item["sum_case_medians_weighted_ms"],
                ]
            )
    lines = [
        "#### 恢复环境投影算子参考标定",
        "",
        "测试对象为 Qwen3.5-35B-A3B BF16、TP2 的密集投影、MoE 专家投影和词表投影。"
        "使用与 r7 计量相同的恢复运行环境和两张 Ascend 910B2；PyTorch 为 2.13.0+cpu，torch-npu 为 2.13.0.rc1。"
        "三次标定分别使用新进程、独立目录和空编译缓存，执行前检查四卡空闲并独占预约，结束后四卡释放。",
        "",
        "输入形状及调用权重来自 r7 原始轨迹，不沿用较早实验的调用次数。密集投影覆盖 121 种形状；"
        "专家投影覆盖每卡 36,708 个实际分组分布，w13、w2 各测三次，共保存 440,496 条计时；"
        "词表投影覆盖 45 种形状、每卡 2,622 次调用、96,376 行。专家与词表调用按已有设备事件及 2 ms 时钟映射容差归属请求窗口。",
        "",
        "每种密集形状测五组、每组 100 次图重放；每个专家分布测三次单次重放；每种词表形状测三组、每组 10 次重放。"
        "独立 CPU FP32 采样数值校验通过，保留全部重复值。下表参考时间为各形状单次耗时中位数乘以 r7 对应调用次数后求和。",
        "",
        "| 组件 | Rank | 形状或分布数 | r7 调用次数 | 重复组数 | 加权组件参考时间（ms） |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        "| " + " | ".join(map(str, row[:-1])) + f" | {row[-1]:.3f} |" for row in rows
    )
    lines.extend(
        [
            "",
            "专家 w2 的两卡加权参考时间存在差异，分别保留，不选取较快卡替代另一卡。"
            "各组的重复值、离散程度、误差、版本与实际导入路径均见归档。",
            "",
            "上述结果是独立、热重放的组件参考，不是整模型关键路径。密集与专家形状包含图填充，不能将物理矩阵运算数冒充有效 F；"
            "词表参考不包含 gather 和采样。合成输入和驻留权重不复现整模型工作集，设备事件计时可能包含主机提交间隙。"
            "本次补齐三组投影参考，不替代注意力、GDN、非矩阵运算、HBM 与通信的完整匹配参考，也不证明完整执行流依赖。"
            "完整有效 F 和 M01 达成率仍不报告，正式 B0/B1 性能及任务质量数值保持不变。",
            "",
            "证据：[三组投影参考归档索引](m01-recovery/projection-references-20261021/archive.json)，"
            "包含原始 Release 地址、逐文件 SHA256、设备状态、退出与资源释放记录。",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    with tarfile.open(
        ROOT / "m01-recovery/projection-references-20261021/evidence.tar.xz"
    ) as archive:
        data = json.load(archive.extractfile("summaries.json"))
    path = ROOT / "中期测试报告.md"
    text = path.read_text()
    assert "报告版本：V0.54" in text
    assert "#### 恢复环境投影算子参考标定" not in text
    text = text.replace("报告版本：V0.54", "报告版本：V0.55", 1)
    text = text.replace(
        "#### GDN 元数据发布与预处理图重放验证",
        render(data) + "\n#### GDN 元数据发布与预处理图重放验证",
        1,
    )
    path.write_text(text)


if __name__ == "__main__":
    main()
