"""Render current-B1 M05 evidence without treating retrieval as performance."""

import json
from pathlib import Path


def render(index):
    audit = index["audit"]
    supplement = index["supplementary_final_answer_diagnosis"]
    final_cases = {c["case_id"]: c for c in supplement["cases"]}
    rows = [
        "### 7.2 当前B1恢复环境长输入检索",
        "",
        "测试对象为Qwen3.5-35B-A3B BF16、TP2、E24/R36、KV materialization与Microflow组合。"
        "保持65,536上下文、FULL图、MTP2、每卡20 GiB状态预算及8 GiB host cache，"
        "thinking沿用服务默认值。该配置与5.8节恢复环境质量复核一致，不将旧环境结果合并为同一次运行。",
        "",
        "合成档案中植入六位检索码，在16K、32K两档输入的10%、50%、90%填充位置各执行一题，"
        "C1串行、temperature=0、请求seed=0、输出上限8192。预先冻结完整请求与token IDs，"
        "发题前与服务端tokenization核对。预定判分对整个content正文去除首尾空白后作严格匹配。"
        "另对原始输出中唯一的`</think>`结束标记之后的答案段作补充解析，不从思考文本搜索正确码，"
        "不替换预定判分。该测试不计入主数据集准确率或性能排名。",
        "",
        "| 题目 | 实际输入，token | 正文严格匹配 | 最终答案段匹配（补充） | 输出，token | 结束原因 |",
        "| ---- | --------------: | ------------ | -------------------- | ----------: | -------- |",
    ]
    for case in audit["cases"]:
        verdict = "通过" if case["correct"] else "未通过"
        final = (
            "正确" if final_cases[case["case_id"]]["final_segment_exact"] else "错误"
        )
        rows.append(
            f"| {case['case_id']} | {case['input_tokens']:,} | {verdict} | "
            f"{final} | {case['completion_tokens']} | {case['finish_reason']} |"
        )
    rows.extend(
        [
            "",
            f"完成{index['result']['completed']}/6题、传输失败{index['result']['failed']}题，"
            f"正文严格匹配{index['result']['correct']}/6题，"
            f"最终答案段匹配{supplement['supplementary_final_segment_correct']}/6题。"
            "六题正文均包含思考文本，独立reasoning字段为空，因此正文格式检查未通过；"
            "不能将该结果解释为六题检索码均错误。实际输入均超过16,384 token；"
            "结论限于本表长度与检索任务，不外推更长上下文、多并发质量或复杂任务通过率。",
            "",
            "使用新服务、独立空编译缓存及逐请求缓存隔离；六个运行时缓存盐均有引擎接收记录，"
            "KV策略均为recompute且复用token数为零。完整SSE、源码、输入、设备采样和退出记录已归档，"
            "四张NPU已释放。首次启动因控制器HTTP就绪状态判断错误而中止，零请求；"
            "修正后使用另一新服务与空缓存运行，首次记录一并保留。",
            "",
            "证据：[完整请求、SSE及校验清单](long-context-current-b1-20261021/archive.json)。",
        ]
    )
    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent / "long-context-current-b1-20261021"
    print(render(json.loads((folder / "archive.json").read_text())))
