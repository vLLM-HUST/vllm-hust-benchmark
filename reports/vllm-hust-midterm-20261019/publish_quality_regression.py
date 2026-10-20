"""Publish verified paired quality counts without changing performance values."""

import argparse
import json
from pathlib import Path

REPORT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    destination = REPORT / "quality-regression"
    destination.mkdir(exist_ok=False)
    for source, target in (
        ("quality-microflow-pair.json", "microflow-pair.json"),
        ("quality-kvmat-pair.json", "kvmat-pair.json"),
        ("quality-finish-breakdown.json", "finish-breakdown.json"),
    ):
        (destination / target).write_bytes((args.root / source).read_bytes())
    micro = json.loads((destination / "microflow-pair.json").read_text())
    kvmat = json.loads((destination / "kvmat-pair.json").read_text())
    finishes = json.loads((destination / "finish-breakdown.json").read_text())
    assert micro["b0"] == kvmat["b0"]
    assert micro["request_plan_sha256"] == kvmat["request_plan_sha256"]
    rows = {row["run"]: row for row in finishes["runs"]}
    selections = [
        (micro["engines"][0], "B0原生", micro["b0"]),
        (micro["engines"][1], "vLLM-HUST E24/R36 + KV + FA Microflow", micro["b1"]),
        (kvmat["engines"][1], "vLLM-HUST E16/R36 + KV", kvmat["b1"]),
    ]
    table = [
        "| 方案 | 判对/200 | 正确率 | 自然结束且判对 | 达到输出上限 | 其中提取命中 | 传输失败 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run, label, score in selections:
        row = rows[run]
        assert row["correct"] == score["correct"] and row["total"] == 200
        split = row["finish_breakdown"]
        assert split["length"]["count"] == score["truncated"]
        table.append(
            f"| {label} | {score['correct']}/200 | {score['correct'] / 2:.2f}%"
            f" | {split['stop']['scorer_matches']} | {split['length']['count']}"
            f" | {split['length']['scorer_matches']} | 0 |"
        )
    text = "\n".join(table) + "\n\n"
    text += (
        "达到上限的输出仍按同一固定评分器提取答案；其中提取命中不等同于自然完成解答。"
        "三组均完成200条请求，结束后四卡确认释放。\n\n"
    )
    for label, pair in (("Microflow", micro), ("E16/R36+KV", kvmat)):
        transitions = pair["paired_transitions"]
        text += (
            f"{label}相对B0的正确率变化为{pair['delta_percentage_points']:+.2f}个百分点；"
            f"{transitions.get('incorrect_to_correct', 0)}题由错变对，"
            f"{transitions.get('correct_to_incorrect', 0)}题由对变错。"
            " "
        )
    text += (
        "\n\n上述为单次固定子集观测，不作为全集质量等价或统计非劣性的结论。\n\n"
        "证据：[Microflow配对复算](quality-regression/microflow-pair.json)、"
        "[E16/R36+KV配对复算](quality-regression/kvmat-pair.json)、"
        "[结束原因分解](quality-regression/finish-breakdown.json)、"
        "[原始归档索引](quality-regression/raw/archive.json)。\n\n"
    )
    report = REPORT / "中期测试报告.md"
    content = report.read_text()
    assert "### 5.7 MMLU-Pro性能输入的任务质量回归" in content
    assert "<!-- BEGIN QUALITY_RESULTS -->" not in content
    marker = "## 六、有效算力结果"
    assert content.count(marker) == 1
    content = content.replace(
        marker,
        "<!-- BEGIN QUALITY_RESULTS -->\n\n"
        + text
        + "<!-- END QUALITY_RESULTS -->\n\n"
        + marker,
    )
    report.write_text(content)


if __name__ == "__main__":
    main()
