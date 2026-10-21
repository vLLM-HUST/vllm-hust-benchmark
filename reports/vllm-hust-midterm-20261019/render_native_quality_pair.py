"""Render the recovered native/candidate pair from archived observations."""

import hashlib
import io
import json
from pathlib import Path
import tarfile

NATIVE = "m01-recovery-model-quality-b0-r3"
CANDIDATE = "m01-recovery-model-quality-r1"


def render(native, candidate):
    pair = json.loads(native["quality-recovery-native-pair-r3.json"])
    assert pair["engines"] == [NATIVE, CANDIDATE]
    results = []
    for data, run in ((native, NATIVE), (candidate, CANDIDATE)):
        receipt = json.loads(data[f"{run}/receipt.json"])
        assert receipt["status"] == "measured" and receipt["all_devices_released"]
        result = json.loads(data[f"{run}/mmlu-pro.json"])
        assert result["completed"] == result["total"] == 200 and result["failed"] == 0
        results.append(result)
    for name in ("quality-request-plan.json", "quality-tokenize-preflight.json"):
        assert json.loads(native[f"{NATIVE}/{name}"]) == json.loads(
            candidate[f"{CANDIDATE}/{name}"]
        )
    delta = (results[1]["correct"] - results[0]["correct"]) / 2
    assert delta == pair["delta_percentage_points"]
    lines = [
        "### 5.9 恢复环境 Native 与优化方案配对质量",
        "",
        "采用同一 Qwen3.5-35B-A3B BF16 模型、TP2 设备、固定 MMLU-Pro 200 题、原评分器及相同请求。"
        "输出上限 8192、temperature=0、seed=0，保留默认 thinking 和 EOS。两臂逐题分词及请求计划一致，"
        "独立服务和空编译缓存，不继承前轮推理状态。此项为固定子集质量比较，不替代正式吞吐结果。",
        "",
        "Native B0 使用原 32K、EP 开启、APC 关闭、无 MTP、FULL_DECODE_ONLY 配置；"
        "B1 为第5.8节的 E24/R36、KV materialization 与 Prefill Microflow 方案。"
        "两臂使用同一恢复基础运行栈；Native 运行时单独恢复五个候选修改模块的原生源码，"
        "保留同一 CANN 重建产物，完整文件清单和差异哈希随证据保存。该 Native 不是官方 0.18/0.18 基线。",
        "",
        "| 方案 | 请求成功 / 失败 | 固定评分器判对 | 自然结束数 | 达到输出上限数 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label, result in zip(
        ("恢复环境 Native B0", "恢复环境 vLLM-HUST B1"), results, strict=True
    ):
        stops = sum(c["finish_reason"] == "stop" for c in result["cases"])
        lengths = sum(c["finish_reason"] == "length" for c in result["cases"])
        assert stops + lengths == 200
        lines.append(
            f"| {label} | 200 / 0 | {result['correct']}/200 | {stops} | {lengths} |"
        )
    transitions = pair["paired_transitions"]
    lines.extend(
        [
            "",
            f"B1 相对 B0 的判对率变化为 {delta:+.2f} 个百分点。"
            f"逐题配对中，B0 错误而 B1 正确为 {transitions.get('incorrect_to_correct', 0)} 题，"
            f"B0 正确而 B1 错误为 {transitions.get('correct_to_incorrect', 0)} 题。"
            "达到输出上限的回答按原评分器计分并保留；不由一次固定子集比较推断全集质量或统计非劣性。",
            "",
            "两臂完整 SSE 已重建并按原评分器复算，设备均已释放。"
            "[Native 原始证据及配对审计](quality-regression/native-recovered-20261021/archive.json)，"
            "[B1 原始证据](quality-regression/recovered-20261021/archive.json)。",
            "",
            "Native 前两次尝试分别因 CANN Python 路径缺失和候选 runner 的 BetterScale 依赖失败；"
            "首次未发出请求，第二次200请求均传输失败，均不形成任务准确率结果。"
            "两次均已释放设备，原始记录分别保存于同一证据 Release 的 "
            "`m01-quality-b0-recovery-r1-evidence.tar.gz` 与 `m01-quality-b0-recovery-r2-evidence.tar.gz`。",
            "",
        ]
    )
    return "\n".join(lines)


def load(directory):
    index = json.loads((directory / "archive.json").read_text())
    chunks = []
    for part in index["parts"]:
        raw = (directory / part["name"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == part["sha256"]
        chunks.append(raw)
    raw = b"".join(chunks)
    assert hashlib.sha256(raw).hexdigest() == index["archive_sha256"]
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        data = {m.name: archive.extractfile(m).read() for m in archive if m.isfile()}
    for member in index["members"]:
        assert hashlib.sha256(data[member["name"]]).hexdigest() == member["sha256"]
    return data


if __name__ == "__main__":
    root = Path(__file__).resolve().parent / "quality-regression"
    print(
        render(
            load(root / "native-recovered-20261021"), load(root / "recovered-20261021")
        )
    )
