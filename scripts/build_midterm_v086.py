"""Publish incremental SWE observations, retaining original grades and limits."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPORT = "reports/vllm-hust-midterm-20261019/中期测试报告.md"
OLD_REPORT = "reports/vllm-hust-midterm-20261019/中期测试报告-v085.md"
SOURCE = "reports/swe-native018-incremental-20261026/catalog.json"
OUTPUT = ROOT / "website-publication/data/midterm-v086.json"


def rows(value, english=False):
    return [
        [
            str(i + 1),
            f"{x['official_resolved']} / 1",
            str(x["model_calls"]),
            f"{x['elapsed_seconds']:.6f}",
            x["status"],
            ("Application unproven" if english else "补丁应用未获证明")
            if x["patch_application_evidence_limited"]
            else ("Official grade" if english else "官方评分"),
        ]
        for i, x in enumerate(value["entries"])
    ]


def summary(value, english=False):
    attempted = value["attempted_unique_tasks"]
    resolved = value["recorded_official_resolved"]
    limited = value["application_evidence_limited_tasks"]
    if english:
        return (
            f"Designated 0.18 baseline: {attempted}/731 unique tasks attempted; recorded official scores {resolved}/{attempted}, "
            f"zero indeterminate official grades. {limited} task has unproven patch application; retain its original score without claiming complete patch validation. "
            "Not full-dataset acceptance or a B0/B1 performance pair."
        )
    return (
        f"指定0.18基线已执行{attempted}/731个不同任务，官方原记录{resolved}/{attempted}、未决0；"
        f"其中{limited}题补丁应用未获证明，原分数保留但不作为完整补丁验证。不是全量验收或B0/B1性能配对。"
    )


def render_report(old, value):
    assert "报告版本：V0.85" in old
    old = old.replace(
        "报告版本：V0.85（补齐既有C16指定基线逐请求时延）",
        "报告版本：V0.86（SWE-bench-Pro指定基线增量实测与评分证据）",
        1,
    )
    start = old.index("#### SWE-bench-Pro指定0.18单Agent实测")
    end = old.index("#### 0.18 Native与已有优化方案同题质量", start)
    block = "\n".join(
        [
            "#### SWE-bench-Pro指定0.18增量实测",
            "",
            "Qwen3.5-35B-A3B，vLLM 0.18.0 / vLLM-Ascend 0.18.0。各题一次Agent尝试，100步、1200秒、max_tokens=4096、temperature=0、seed=0、thinking=false、零自动重试；新任务与独立官方评分容器，MTU1400。后续任务复用同运行身份的五项资格，不重复资格或已测题。",
            "",
            "| 任务序号 | 官方原记录 | 模型调用 | Agent秒 | 状态 | 证据范围 |",
            "|---|---:|---:|---:|---|---|",
            *["| " + " | ".join(row) + " |" for row in rows(value)],
            "",
            summary(value),
            "",
            *[
                f"任务{i + 1}官方测试明细："
                + "、".join(f"{k} {v}" for k, v in entry["parsed_test_counts"].items())
                + "。"
                for i, entry in enumerate(value["entries"])
                if entry.get("parsed_test_counts")
            ],
            "",
            "首题100步耗尽，官方评分0/1。第二题主动提交的补丁包含镜像已有Redis文件的新文件段，官方入口未保存git apply退出码或应用后源码；源码表明存在文件冲突，before-all失败不能独立证明测试了已应用补丁。官方false及未决0保留，未修改补丁或重新评分。此前同题r2初始文件检查失败时模型调用0，未计入任务分母。",
            "",
            "评分阶段不调用模型。客户端或服务退出0不等于任务判对；各任务完整资源释放回执、实际身份及受限原件哈希见规范记录。",
            "",
            "第三题45次调用后主动提交，独立复核为有效未通过结果。原始Mocha记录190项（189通过、1失败）；官方解析器按名称去重为173项（172通过、1失败），17项重名合并。F2P为0/1，P2P为172/172，无缺失项。初始工作树及未跟踪文件均干净，补丁与官方评分输入逐字节一致，不继承第二题的文件冲突限制。",
            "",
            "证据：[增量规范汇总与边界](../swe-native018-incremental-20261026/README.md)、[可复算任务目录](../swe-native018-incremental-20261026/catalog.json)、[首题原始归档](../swe-native018-single-agent-20261026/README.md)。私有题目、回答、补丁和轨迹不公开。",
            "",
        ]
    )
    old = old[:start] + block + "\n" + old[end:]
    lines = old.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        cells = line.split("|")
        if cells[1].strip() == "SWE-bench-Pro":
            cells[2] = " 三组各3600秒真实Agent记录保留；" + summary(value) + " "
            lines[i] = "|".join(cells)
        elif cells[1].strip() == "数据集覆盖":
            cells[2] = (
                " MMLU-Pro、FrontierScience及HLE完整结果保留；" + summary(value) + " "
            )
            lines[i] = "|".join(cells)
    return "\n".join(lines) + "\n"


def build(revision, read=None):
    if read is None:

        def read(path):
            return subprocess.check_output(
                ["git", "show", f"{revision}:{path}"], cwd=ROOT
            )

    report = read(REPORT)
    assert "报告版本：V0.86" in report.decode()
    value = json.loads(read(SOURCE))
    data = json.loads(read("website-publication/data/midterm-v085.json"))
    data.update(
        version="V0.86",
        source_commit=revision,
        report_sha256=hashlib.sha256(report).hexdigest(),
        report_url=f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{revision}/{REPORT}",
        swe_incremental_tasks=value,
    )
    section = next(
        s for s in data["sections"] if s["id"] == "swe-single-agent-observation"
    )
    section.update(
        title="SWE-bench-Pro指定基线增量实测 / Incremental designated SWE baseline",
        scope=summary(value),
        scope_en=summary(value, True),
        source_heading="SWE-bench-Pro指定0.18增量实测",
        tables=[
            dict(
                headers=[
                    "任务序号",
                    "官方原记录",
                    "模型调用",
                    "Agent秒",
                    "状态",
                    "证据范围",
                ],
                rows=rows(value),
            )
        ],
        tables_en=[
            dict(
                headers=[
                    "Task",
                    "Official record",
                    "Model calls",
                    "Agent seconds",
                    "Status",
                    "Evidence scope",
                ],
                rows=rows(value, True),
            )
        ],
        evidence=[
            dict(
                path=SOURCE,
                kind="blob",
                label="增量结果及源哈希 / Incremental results and source hashes",
            )
        ],
    )
    section["scope"] += (
        " 第二题的官方负结果保留；git apply状态未采集，不能认定测试了已应用补丁。"
    )
    section["scope_en"] += (
        " The second task retains its recorded negative; git-apply status was not captured, so testing of the applied patch is unproven."
    )
    remaining = next(s for s in data["sections"] if s["id"] == "v15-remaining")
    remaining["evidence"].append(
        dict(path=SOURCE, kind="blob", label="SWE增量任务 / SWE incremental tasks")
    )
    for key, english in [("tables", False), ("tables_en", True)]:
        for table in remaining[key]:
            for row in table["rows"]:
                if row[0] == "SWE-bench-Pro":
                    row[1] = (
                        "Three existing 3,600-second runs retained; "
                        if english
                        else "三组各3600秒真实Agent记录保留；"
                    ) + summary(value, english)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision")
    parser.add_argument("--prepare-report", action="store_true")
    args = parser.parse_args()
    if args.prepare_report:
        (ROOT / REPORT).write_text(
            render_report(
                (ROOT / OLD_REPORT).read_text(), json.loads((ROOT / SOURCE).read_text())
            )
        )
    else:
        assert args.revision
        OUTPUT.write_text(
            json.dumps(build(args.revision), indent=2, ensure_ascii=False) + "\n"
        )
