"""Generate current navigation and bounded task-result documentation."""

import json
from pathlib import Path

import build_midterm_v090 as projection

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "reports/swe-native018-continuation-20261026"


def render(value):
    lines = [
        "# SWE-bench-Pro V1 指定基线后续实测",
        "",
        "Qwen3.5-35B-A3B，vLLM 0.18.0 / vLLM-Ascend 0.18.0。每题一次Agent，100步、1200秒、max_tokens=4096、temperature=0、seed=0、thinking=false、零自动重试。五项既有资格按实际运行身份复用；仅追加此前未执行任务。",
        "",
        projection.previous.previous.summary(value),
        "",
        "| 任务序号 | 官方原记录 | 模型调用 | Agent秒 | 状态 | 证据范围 |",
        "|---|---:|---:|---:|---|---|",
        *["| " + " | ".join(row) + " |" for row in projection.rows(value)],
        "",
        "## 评分与覆盖",
        "",
        "原官方评分、完整已执行分母、失败和未出现的预期测试名均保留。应用前、应用后及测试前（实际待评分）文件身份分开记录；评分进程退出0不等于全部测试成功。没有改补丁、评分器或预算，没有重新运行旧题。",
        "",
    ]
    for entry in value["entries"]:
        note = projection.coverage_note(entry)
        if note:
            lines.extend([note.strip(), ""])
    lines.extend(
        [
            "## 证据与复算",
            "",
            "inputs.json绑定上一版规范目录和本批summary/review/release/custody。历史12题原件不变；catalog.json重新计算唯一任务数、官方成绩及覆盖限制。",
            "",
            "summary-*.json为原始记录派生摘要，review-*.json为独立复核的脱敏投影，release-*.json为实际服务释放回执，custody-*.json绑定受限原始证据封存。*.py.source保留原派生程序；其执行依赖受控CPU原件，不是公开重判入口。",
            "",
            "受限原件位于授权CPU持久卷swe-pro-shuhao-20261025的runs目录及对应模型窗口目录，通过run_id与SHA索引。公开包不包含私有题目、回答、轨迹、补丁或凭据。",
            "",
            "公开离线检查只验证聚合算术、源码/证据摘要绑定和公开文件大小/SHA，不重新运行私有任务。",
            "",
            "```bash",
            "python3 reports/swe-native018-continuation-20261026/verify.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    value = json.loads((FOLDER / "catalog.json").read_text())
    (FOLDER / "README.md").write_text(render(value))
    path = ROOT / "reports/four-primary-datasets-20261025/README.md"
    lines = path.read_text().splitlines()
    found = set()
    for i, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        dataset = line.split("|")[1].strip()
        if dataset == "SWE-bench-Pro":
            found.add(dataset)
            lines[i] = (
                "| SWE-bench-Pro | "
                + projection.previous.previous.summary(value)
                + " | 目标编译失败及补丁应用限制分别保留；不构成全量或B0/B1配对。 |"
            )
        elif dataset in ("TerminalBench 2.1", "TerminalBench2.1"):
            found.add("TerminalBench 2.1")
            lines[i] = (
                "| TerminalBench 2.1 | "
                + projection.terminal_scope()
                + " | 模型任务0/89；其他任务环境控制及模型执行尚未完成。 |"
            )
    assert found == {"SWE-bench-Pro", "TerminalBench 2.1"}
    text = "\n".join(lines) + "\n"
    text = text.replace(
        "../swe-native018-next-repository-20261026/",
        "../swe-native018-continuation-20261026/",
    )
    text = text.replace(
        "已有镜像及控制覆盖八题，不等于全部 731\n题可执行。",
        "镜像准备与已执行结果分别统计，已测范围以规范任务目录为准，不等于全部731题可执行。",
    )
    link = "- Terminal ARM64原任务控制：[OpenSSL nop/oracle](../terminal-openssl-arm64-controls-20261026/README.md)。"
    if link not in text:
        text += "\n" + link + "\n"
    path.write_text(text)


if __name__ == "__main__":
    main()
