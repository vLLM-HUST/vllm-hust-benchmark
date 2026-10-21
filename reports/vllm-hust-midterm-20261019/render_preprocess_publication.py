"""Render the bounded publication qualification from archived measurements."""

import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent


def main():
    with tarfile.open(
        ROOT / "m01-recovery/preprocess-publication-20261021/evidence.tar.xz"
    ) as archive:
        audit = json.load(
            archive.extractfile(
                "m01-preprocess-publication-recovery-r3/independent-audit.json"
            )
        )
        identity = json.load(
            archive.extractfile("source/recovery-request-identity-r1.json")
        )
    assert audit["status"] == "cpu_reaudit_passed"
    lines = [
        "#### GDN 元数据发布与预处理图重放验证",
        "",
        "测试对象为恢复环境的实际 `MTPFrame.fill_mtp → publish → preprocess` 路径，源码哈希与锁定模型 Python 文件一致。"
        "使用两张 Ascend 910B2、BF16 输入、24 请求容量、MTP2；decode 图容量为 80，混合 prefill 图容量为 1024、2048、4096。"
        "每个容量测试两个元数据 bank 和两种 Q 布局，交替发布长短请求与不同 accepted counts，每组重放四次。",
        "",
        "CPU 独立构造 prefill、verify、restore 索引；核验设备发布后的请求边界、槽位、accepted counts 和映射填充区。"
        "Q/K、门控及 beta 按既定容差与 CPU 公式比较，V 拷贝要求精确一致。保存设备输出后独立重新生成输入并复算。",
        "",
        "| 指标 | 结果 |",
        "| --- | --- |",
        f"| 通过的图重放用例 | {audit['cases']} |",
        f"| 独立复算数值张量检查 | {audit['numerical_tensor_checks']} |",
    ]
    for name, value in audit["max_abs_by_output"].items():
        lines.append(f"| {name} 最大绝对误差 | {value:.12g} |")
    lines.extend(
        [
            "",
            "前两次探针分别因缺少 TASK_QUEUE_ENABLE=0、将 72 个有效槽位误作 decode 图容量而在计算准入阶段失败；"
            "失败日志原样保留。修正探针后使用新进程、新目录和空编译缓存运行，退出码为 0，四张卡均释放，未修改模型实现。",
            "",
            "本结果证明合成输入下实际元数据发布及预处理图重放的局部正确性。每次重放后同步检查，未覆盖完整模型的并发批次重叠、"
            "全部执行流依赖或恢复历史运行的设备映射；不作为完整模型质量、完整有效 F 或 M01 达成率通过依据。",
            "",
            f"两轮完整模型性能记录的 {len(identity['records'])} 个输入已与固定题目及标签凭证逐项对齐；"
            "输出仅保存文本和长度，全部达到 256-token 配置上限，未保留逐请求结束原因、生成 token IDs 或 logits。"
            "文本差异不能据此归因，字母提取不能替代完整回答质量评测。",
            "",
            "原始张量、脚本、失败尝试及逐文件 SHA256 见[发布链路验证证据](m01-recovery/preprocess-publication-20261021/archive.json)。",
            "",
        ]
    )
    report = ROOT / "中期测试报告.md"
    text = report.read_text()
    assert "报告版本：V0.53" in text
    assert "#### GDN 元数据发布与预处理图重放验证" not in text
    text = text.replace("报告版本：V0.53", "报告版本：V0.54", 1)
    text = text.replace(
        "#### r21 设备映射与同轮计量",
        "\n".join(lines) + "\n#### r21 设备映射与同轮计量",
        1,
    )
    report.write_text(text)


if __name__ == "__main__":
    main()
