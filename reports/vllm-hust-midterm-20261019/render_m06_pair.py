"""Render M06 boundaries from hash-verified observations, without amortization."""

import hashlib
import json
from pathlib import Path


def load(directory):
    index = json.loads((directory / "archive.json").read_text())
    for entry in index["files"]:
        raw = (directory / entry["name"]).read_bytes()
        assert len(raw) == entry["bytes"]
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
    return json.loads((directory / "m06-prechecked-pair-result-r1.json").read_text())


def render(result):
    b0, b1 = (result["arms"][arm] for arm in ("b0", "b1"))
    assert result["original_prechecked_protocol_confirmation"]
    assert result["precheck_requests_per_arm"] == 1
    assert result["repetitions"] == 1 and not result["quality_scored"]
    for arm in (b0, b1):
        assert arm["requests"] == 200 and arm["failed"] == 0
        assert arm["output_tokens"] == 51200
        assert (
            abs(
                arm["lifecycle_reservation_card_s_per_token"]
                - 4 * arm["lifecycle_s"] / arm["output_tokens"]
            )
            < 1e-12
        )
    metrics = [
        ("输出吞吐（token/s）", "output_tok_per_s", True),
        ("平均 TTFT（s）", "mean_ttft_s", False),
        ("平均 TPOT（ms/token）", "mean_tpot_s", False),
        ("P99 TPOT（ms/token）", "p99_tpot_s", False),
        ("请求阶段成本（卡秒/token）", "request_reservation_card_s_per_token", False),
        ("全流程成本（卡秒/token）", "lifecycle_reservation_card_s_per_token", False),
    ]
    lines = [
        "### 5.10 恢复环境配对性能与全流程成本",
        "",
        "本项采用与第5.9节一致的恢复基础运行栈和 Native/B1 部署方案。B1 启用 E24/R36、"
        "Prefill Microflow 和 KV materialization 请求重计算策略。固定 MMLU-Pro 原200条输入，"
        "temperature=0、seed=0、输出上限256、无限到达率，不打乱输入。该限长性能测试不计任务准确率，"
        "不替代第5.9节输出上限8192的质量测试，也不作为官方0.18/0.18基线。",
        "",
        "顺序为 B1 后 B0，各运行一次；每臂启动独立服务并使用空编译缓存，均独占四张卡、TP2使用其中两张。"
        "不继承前轮推理状态；操作系统文件页缓存未全局清空，因此不声明冷磁盘条件。"
        "每臂先完成一次客户端预检查，再测量200条请求。预检查耗时计入全流程，输出不计入200条请求的分母。"
        "两臂引擎关闭等待均显式为60秒，B1管理器等待90秒，外层等待120秒。",
        "",
        "| 指标 | Native B0 | vLLM-HUST B1 | B1相对B0变化 |",
        "| --- | --- | --- | --- |",
    ]
    for label, key, higher in metrics:
        scale = 1000 if "TPOT" in label else 1
        change = 100 * (b1[key] / b0[key] - 1)
        lines.append(
            f"| {label} | {b0[key] * scale:.6f} | {b1[key] * scale:.6f} | {change:+.2f}% |"
        )
    lines += [
        "",
        "变化列为 `(B1/B0-1)×100%`；吞吐越高越好，时延与成本越低越好。成本主口径为"
        "`4×占用秒数/成功输出token数`；按两张实际执行卡计算时绝对值减半，相对变化不变。"
        "此为同单价假设下的资源成本，不是货币费用。",
        "",
        "| 阶段（秒） | Native B0 | vLLM-HUST B1 |",
        "| --- | --- | --- |",
    ]
    for label, key in [
        ("启动至客户端", "pre_client_s"),
        ("客户端执行（含预检查）", "client_s"),
        ("客户端结束至设备释放", "post_client_release_s"),
        ("全流程", "lifecycle_s"),
    ]:
        lines.append(f"| {label} | {b0[key]:.3f} | {b1[key]:.3f} |")
    lines += [
        "",
        "两臂各成功200条、失败0条、输出51,200 token。请求阶段吞吐和平均TTFT达到内部提升阈值，"
        "平均TPOT改善未达到5%；P99 TPOT为200个样本的最近秩观测，不推断统计显著性。"
        "请求阶段单位资源成本降低超过10%，但本次全流程成本增加，M06全流程验收未通过。"
        "启动阶段包含加载与编译，不扣除启动或退出时间，不推定长期运行的摊销收益。",
        "",
        "两臂API完成关闭、设备全部释放，无引擎超时强杀；日志均有 EngineDeadError，"
        "不据此宣称所有子进程完全优雅退出。首次补测未执行客户端预检查，作为协议偏差记录单独归档，"
        "不与本组拼接或替代原协议结果。既有工况结果保留，本组不构成跨环境排名更新。",
        "",
        "证据：[配对结果](m06-paired-20261021/m06-prechecked-pair-result-r1.json)、"
        "[完整归档清单](m06-paired-20261021/archive.json)、"
        "[协议偏差记录](m06-paired-20261021/m06-pair-result-r2.json)。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(render(load(Path(__file__).resolve().parent / "m06-paired-20261021")))
