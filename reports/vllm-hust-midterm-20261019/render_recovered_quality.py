"""Render quality observations from the complete recovered-run archive."""

import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
RUN = "m01-recovery-model-quality-r1"


def render(data, prior):
    result = json.loads(data[f"{RUN}/mmlu-pro.json"])
    receipt = json.loads(data[f"{RUN}/receipt.json"])
    assert receipt["status"] == "measured" and receipt["all_devices_released"]
    assert result["completed"] == result["total"] == 200 and result["failed"] == 0
    rows = []
    for run, label in (
        ("mmlu-quality-b0-r4", "既有环境 Native B0"),
        ("mmlu-quality-microflow-r1", "既有环境 vLLM-HUST B1"),
    ):
        old = next(r for r in prior["runs"] if r["run"] == run)
        rows.append(
            [
                label,
                "200 / 0",
                f"{old['correct']}/200",
                old["finish_breakdown"]["stop"]["count"],
                old["finish_breakdown"]["length"]["count"],
            ]
        )
    stops = sum(c["finish_reason"] == "stop" for c in result["cases"])
    lengths = sum(c["finish_reason"] == "length" for c in result["cases"])
    assert stops + lengths == 200
    rows.append(
        ["恢复环境 vLLM-HUST B1", "200 / 0", f"{result['correct']}/200", stops, lengths]
    )
    b0 = json.loads(data["quality-recovery-r1-pair-b0.json"])
    micro = json.loads(data["quality-recovery-r1-pair-microflow.json"])
    lines = [
        "### 5.8 恢复环境完整回答质量复核",
        "",
        "测试模型为 Qwen3.5-35B-A3B BF16、TP2；vLLM-HUST 方案为 E24/R36、KV materialization 与 Prefill Microflow，"
        "保持 64K 上下文、FULL 图、MTP2、APC、每卡 20 GiB State 和 8 GiB host cache。"
        "恢复环境使用新服务及空编译缓存，关闭计量观察器。此项为质量复核，不用于重新选择吞吐 B1。",
        "",
        "采用原 MMLU-Pro 200 题、原提示词与固定评分器；输出上限 8192、temperature=0、请求 seed=0、无限到达率。"
        "200 题的服务端 tokenization 与既有质量运行逐项一致。完整 SSE、结束原因、逐题答案与失败均保留，评分分母固定为 200。",
        "",
        "| 运行与方案 | 请求成功 / 失败 | 固定评分器判对 | 自然结束数 | 达到输出上限数 |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    lines.extend(
        [
            "",
            f"恢复环境相对既有 B0 的判对率变化为 {b0['delta_percentage_points']:+.2f} 个百分点；"
            f"相对既有 Microflow B1 为 {micro['delta_percentage_points']:+.2f} 个百分点。"
            "上述比较跨运行环境，不是同一环境下新增的 B0/B1 配对性能测量。达到输出上限的回答仍按原评分器计分，"
            "不将提取命中等同于自然完成，不由单次固定子集推断全集质量或统计非劣性。",
            "",
            "全部 200 条原始 SSE 已独立重建，逐题评分及配对转移已复算；服务清理完成并释放四卡。"
            "具体导入路径、源码哈希、请求计划、缓存初始化、设备状态及原始记录见"
            "[恢复环境质量归档](quality-regression/recovered-20261021/archive.json)。",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    directory = ROOT / "quality-regression/recovered-20261021"
    index = json.loads((directory / "archive.json").read_text())
    raw = b"".join((directory / part["name"]).read_bytes() for part in index["parts"])
    assert hashlib.sha256(raw).hexdigest() == index["archive_sha256"]
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        data = {m.name: archive.extractfile(m).read() for m in archive if m.isfile()}
    prior = json.loads((ROOT / "quality-regression/finish-breakdown.json").read_text())
    report = ROOT / "中期测试报告.md"
    text = report.read_text()
    assert "### 5.8 恢复环境完整回答质量复核" not in text
    report.write_text(
        text.replace(
            "## 六、有效算力结果", render(data, prior) + "\n## 六、有效算力结果", 1
        )
    )


if __name__ == "__main__":
    main()
