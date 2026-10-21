"""Render the independent API repair confirmation without replacing old scores."""

import json
from pathlib import Path


def render(index):
    assert index["status"] == "measured"
    audit = index["audit"]
    result = index["result"]
    assert audit["qualification_passed"] and audit["all_devices_released"]
    assert result["correct"] == sum(c["correct"] for c in audit["cases"])
    rows = [
        "### 7.3 思考字段分离后的独立确认",
        "",
        "沿用7.2节同一模型、恢复运行栈、E24/R36、KV materialization与Prefill Microflow组合，"
        "保留FULL图、MTP2、thinking开启及原六题的完整请求、分词和严格正文评分器。"
        "服务新增原生`--reasoning-parser qwen3`，客户端兼容收集`reasoning`与`reasoning_content`；"
        "两个字段同时出现且内容冲突时判为接口错误，不拼接或挑选答案。",
        "",
        "新服务使用独立空编译缓存，先完成一条固定算术请求以确认实际API返回独立思考字段及最终正文，"
        "资格请求不计入六题分母。随后六题各执行一次，不改变输入或根据结果调参重跑。"
        "输出token按API的completion_tokens统计，包含思考生成，不只计最终答案。"
        "操作系统文件页缓存未全局清空，不声明冷磁盘条件。",
        "",
        "| 题目 | 实际输入，token | 正文严格匹配 | 独立思考字段 | 输出，token | 结束原因 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case in audit["cases"]:
        verdict = "正确" if case["correct"] else "错误"
        separation = "已分离" if case["reasoning_separated"] else "未分离"
        rows.append(
            f"| {case['case_id']} | {case['input_tokens']:,} | {verdict} | "
            f"{separation} | {case['completion_tokens']} | {case['finish_reason']} |"
        )
    rows.extend(
        [
            "",
            f"新运行完成{result['completed']}/6题、传输失败{result['failed']}题，"
            f"正文严格匹配{result['correct']}/6题。实际输入均不少于16,384 token，"
            "思考文本与最终正文分离。7.2节原运行的正文严格匹配0/6及补充解析6/6单独保留。"
            "结论仅覆盖本组六题合成检索和接口行为，不代表主数据集准确率、"
            "复杂长文理解、多并发质量或吞吐收益。",
            "",
            "全部七条请求具有不同运行时缓存盐，与原六题运行不相交，KV策略均为recompute且复用token为零。"
            "两卡执行、四卡独占，完整SSE、来源哈希、设备状态与退出记录已保存，四卡全部释放。",
            "",
            (
                "API完成关闭。"
                if audit["api_shutdown_complete"]
                else "未记录API关闭完成标记。"
            )
            + (
                "退出日志仍有EngineDeadError。"
                if audit["engine_dead_error_logged"]
                else "退出日志未见EngineDeadError。"
            )
            + "资源释放证据不等同于所有子进程完全优雅退出。",
            "",
            "证据：[固定计划、完整请求、SSE及校验清单](m05-parser-confirmation-20261021/archive.json)。",
        ]
    )
    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent / "m05-parser-confirmation-20261021"
    print(render(json.loads((folder / "archive.json").read_text())))
