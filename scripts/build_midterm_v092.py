"""Project the next measured SWE task without rewriting historical observations."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import build_midterm_v088 as common
import build_midterm_v090 as swe
import build_midterm_v091 as previous

ROOT = Path(__file__).resolve().parents[1]
REPORT = previous.REPORT
OLD_REPORT = "reports/vllm-hust-midterm-20261019/中期测试报告-v091.md"
SOURCE = "reports/swe-native018-navidrome-20261027/catalog.json"
TERMINAL = "reports/terminal-cancel-async-controls-20261027/summary.json"
OUTPUT = ROOT / "website-publication/data/midterm-v092.json"


def note(english=False):
    if english:
        return (
            " Navidrome: one Agent attempt, 53 model calls, 511.436103 seconds, Submitted. "
            "The unmodified 9941-byte patch applied successfully; after/graded file identities match. "
            "The original Scanner Suite ran 27 specs: 26 passed and one failed on the Images field. "
            "These nested specs do not replace the official target-name score: F2P 0/1, P2P 0 targets, no missing targets. "
            "The original parser records TestScanner as failed and the task result is false. "
            "Original TagLib deprecation warnings remain; no Agent or grading rerun."
        )
    return (
        " Navidrome仅一次Agent，53次模型调用、511.436103秒，主动提交。原9941字节补丁应用成功，"
        "应用后与测试前实际待评分文件身份一致。原Scanner Suite执行27项spec，26通过、1项Images字段断言失败；"
        "嵌套spec不代替官方目标测试名评分：F2P为0/1、P2P无目标，预期测试名缺失0。"
        "原解析器记录TestScanner失败，官方任务判false。保留原TagLib弃用警告，未重跑Agent或评分。"
    )


def terminal_note(english=False):
    if english:
        return (
            "cancel-async-tasks ARM64 environment controls: original nop 0/6, reward 0; "
            "original oracle 6/6, reward 1. Fresh containers, one attempt each, zero model calls; both released. "
            "Original tests, timing thresholds and 900-second budget unchanged; 1 CPU, 2 GiB, MTU1400. "
            "Pinned uv0.9.5 transport cache only. The earlier missing-Compose failure occurred before any container or test, "
            "reward null retained. These controls add no model coverage: the existing OpenSSL result remains 1/89."
        )
    return (
        "cancel-async-tasks的ARM64环境控制：原nop为0/6、reward=0，原oracle为6/6、reward=1；"
        "各一次、独立全新容器、模型调用0，均已释放。原测试、时间阈值及900秒预算不变，"
        "1CPU、2GiB、MTU1400，仅缓存固定uv0.9.5传输依赖。此前Compose插件缺失发生于容器及测试启动前，"
        "原null保留，不计任务失败。这两项控制不增加模型覆盖，既有OpenSSL模型结果仍为1/89。"
    )


def render_report(old, value, prior):
    assert "报告版本：V0.91" in old
    text = old.replace(
        "报告版本：V0.91（Terminal原答案恢复评分）",
        "报告版本：V0.92（SWE增量与Terminal评分控制）",
        1,
    )
    start = text.index("| 任务序号 | 官方原记录 |")
    end = text.index("\n\n", start)
    table = [
        "| 任务序号 | 官方原记录 | 模型调用 | Agent秒 | 状态 | 证据范围 |",
        "|---|---:|---:|---:|---|---|",
        *["| " + " | ".join(row) + " |" for row in swe.rows(value)],
    ]
    text = text[:start] + "\n".join(table) + text[end:]
    assert common.summary(prior) in text
    text = text.replace(common.summary(prior), common.summary(value))
    marker = "证据：[本批规范结果与复算]"
    assert marker in text
    text = text.replace(marker, note().strip() + "\n\n" + marker, 1)
    marker = "#### 0.18 Native与已有优化方案同题质量"
    text = text.replace(
        marker,
        "#### Terminal异步任务ARM64评分控制\n\n"
        + terminal_note()
        + "\n\n证据：[规范结果及复核](../terminal-cancel-async-controls-20261027/README.md)。\n\n"
        + marker,
        1,
    )
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("|") and line.split("|")[1].strip() == "TerminalBench 2.1":
            lines[i] = (
                "| TerminalBench 2.1 | "
                + previous.scope()
                + " "
                + terminal_note()
                + " | 其余88题与B1同题配对尚未完成 |"
            )
    text = "\n".join(lines) + "\n"
    return text.replace(
        "../swe-native018-continuation-20261026/",
        "../swe-native018-navidrome-20261027/",
    )


def build(revision, read=None):
    if read is None:

        def read(path):
            return subprocess.check_output(
                ["git", "show", f"{revision}:{path}"], cwd=ROOT
            )

    data = previous.build(revision, lambda p: read(OLD_REPORT if p == REPORT else p))
    report = read(REPORT)
    value = json.loads(read(SOURCE))
    terminal = json.loads(read(TERMINAL))
    assert terminal["model_calls"] == terminal["model_task_attempts"] == 0
    assert "报告版本：V0.92" in report.decode()
    assert value["attempted_unique_tasks"] == 15
    assert value["recorded_official_resolved"] == 2
    data.update(
        version="V0.92",
        source_commit=revision,
        report_sha256=hashlib.sha256(report).hexdigest(),
        report_url=f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{revision}/{REPORT}",
        swe_incremental_tasks=value,
        terminal_cancel_controls=terminal,
    )
    section = next(
        s for s in data["sections"] if s["id"] == "swe-single-agent-observation"
    )
    for english, key in ((False, "scope"), (True, "scope_en")):
        section[key] = common.summary(value, english)
        section[key] += "".join(
            swe.coverage_note(e, english) for e in value["entries"][:-1]
        )
        section[key] += note(english)
    section["tables"][0]["rows"] = swe.rows(value)
    section["tables_en"][0]["rows"] = swe.rows(value, True)
    evidence = {
        "path": SOURCE,
        "kind": "blob",
        "label": "Navidrome增量 / Navidrome increment",
    }
    section["evidence"].insert(0, evidence)
    remaining = next(s for s in data["sections"] if s["id"] == "v15-remaining")
    remaining["evidence"].append(evidence.copy())
    control_evidence = {
        "path": TERMINAL,
        "kind": "blob",
        "label": "Terminal异步控制 / Async controls",
    }
    remaining["evidence"].append(control_evidence.copy())
    control_section = {
        "id": "terminal-cancel-async-controls",
        "title": "Terminal异步评分控制 / Async verifier controls",
        "scope": terminal_note(),
        "scope_en": terminal_note(True),
        "source_heading": "Terminal异步任务ARM64评分控制",
        "evidence": [control_evidence],
    }
    for key, english in (("tables", False), ("tables_en", True)):
        control_section[key] = [
            {
                "headers": ["Control", "Tests passed", "Reward", "Model calls"]
                if english
                else ["控制", "测试通过", "奖励", "模型调用"],
                "rows": [
                    [
                        c["role"],
                        f"{c['passed']} / {c['tests']}",
                        str(int(c["reward"])),
                        "0",
                    ]
                    for c in terminal["controls"]
                ],
            }
        ]
    data["sections"].append(control_section)
    for key, english in (("tables", False), ("tables_en", True)):
        for table in remaining[key]:
            for row in table["rows"]:
                if row[0] == "SWE-bench-Pro":
                    row[1] = (
                        (
                            "Three prior 3600-second Agent records retained. "
                            if english
                            else "三组各3600秒真实Agent记录保留；"
                        )
                        + common.summary(value, english)
                        + note(english)
                    )
                elif row[0] == "TerminalBench 2.1":
                    row[1] = previous.scope(english) + " " + terminal_note(english)
    return data


def prepare():
    value = json.loads((ROOT / SOURCE).read_text())
    prior = json.loads((ROOT / swe.SOURCE).read_text())
    (ROOT / REPORT).write_text(
        render_report((ROOT / OLD_REPORT).read_text(), value, prior)
    )
    path = ROOT / "reports/four-primary-datasets-20261025/README.md"
    lines = path.read_text().splitlines()
    found = 0
    for i, line in enumerate(lines):
        if line.startswith("|") and line.split("|")[1].strip() == "SWE-bench-Pro":
            lines[i] = (
                "| SWE-bench-Pro | "
                + common.summary(value)
                + note()
                + " | 其余716题及B1同题配对尚未完成。 |"
            )
            found += 1
        elif line.startswith("|") and line.split("|")[1].strip() == "TerminalBench 2.1":
            lines[i] = (
                "| TerminalBench 2.1 | "
                + previous.scope()
                + " "
                + terminal_note()
                + " | 其余88题及B1同题配对尚未完成。 |"
            )
    assert found == 1
    path.write_text("\n".join(lines) + "\n")
    directory = (ROOT / SOURCE).parent
    (directory / "README.md").write_text(
        "# SWE-bench-Pro指定0.18基线：Navidrome增量实测\n\n"
        + common.summary(value)
        + "\n\n"
        "## 配置与方法\n\nQwen3.5-35B-A3B，vLLM 0.18.0 / vLLM-Ascend 0.18.0；"
        "100步、1200秒、max_tokens=4096、temperature=0、seed=0、thinking=false、零自动重试。"
        "复用相同运行身份的5项资格，不增加资格请求。原镜像go.mod fixture按文件与diff精确绑定，不删除或改写；"
        "Agent与评分各使用独立容器，原官方评分器不变。\n\n## 结果\n\n"
        + note().strip()
        + "\n\n模型服务、Agent及评分容器均已释放。历史14题及其应用/编译限制保留，未覆盖原件。"
        "\n\n## 证据\n\nsummary-navidrome.json为聚合；review-navidrome.json为独立复核，"
        "release-navidrome.json及custody-navidrome.json记录释放和受限原件身份。"
        "私有题目、答案、补丁、请求标识和轨迹不公开。"
        "\n\n离线执行 `python3 verify.py` 复核公开聚合与哈希，不声称无需受限原件即可重跑原始审计。\n"
    )
    index = ROOT / "reports/vllm-hust-midterm-20261019/README.md"
    link = (
        "- Navidrome增量：[规范结果](../swe-native018-navidrome-20261027/README.md)。"
    )
    if link not in index.read_text():
        index.write_text(index.read_text().rstrip() + "\n\n" + link + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-report", action="store_true")
    parser.add_argument("--revision")
    args = parser.parse_args()
    if args.prepare_report:
        prepare()
    else:
        assert args.revision
        OUTPUT.write_text(
            json.dumps(build(args.revision), ensure_ascii=False, indent=2) + "\n"
        )
