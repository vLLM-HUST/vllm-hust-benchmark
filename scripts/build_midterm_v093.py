"""Publish the original failed trial and separately scored partial artifact."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import build_midterm_v092 as previous

ROOT = Path(__file__).resolve().parents[1]
REPORT = previous.REPORT
OLD_REPORT = "reports/vllm-hust-midterm-20261019/中期测试报告-v092.md"
SOURCE = "reports/terminal-cancel-async-partial-recovery-20261027/summary.json"
OUTPUT = ROOT / "website-publication/data/midterm-v093.json"


def scope(english=False):
    if english:
        return (
            "TerminalBench2.1: 2/89 unique tasks attempted; 87 unattempted. "
            "OpenSSL completed-answer recovery passed 6/6 tests, reward 1. "
            "cancel-async-tasks ended with ContextLengthExceededError before submission or grading; "
            "its original reward remains null. Separate verifier-only recovery of the unchanged "
            "unsubmitted partial artifact passed 1/6 tests, reward 0. "
            "This is not a second Agent attempt or a completed-submission score; no full-dataset or B0/B1 paired acceptance."
        )
    return (
        "TerminalBench2.1已尝试2/89题，87题未执行。OpenSSL完整答案恢复评分6/6通过、reward=1；"
        "cancel-async-tasks在提交和评分前因ContextLengthExceededError终止，原reward=null。"
        "未提交的原部分代码单独恢复评分，1/6通过、reward=0，不计第二次Agent或完整提交评分；不构成全量或B0/B1配对验收。"
    )


def note(english=False):
    if english:
        return (
            "Qwen3.5-35B-A3B, vLLM 0.18.0 / Ascend 0.18.0, Harbor 0.24.0 Terminus2, ARM64. "
            "One original Agent: 332.567032 seconds, 18 HTTP requests (17 successful, one context-limit rejection), "
            "223132 prompt and 13450 output tokens. At least 28673 input tokens plus 4096 requested output exceeded 32768. "
            "The original 100-turn, 900-second Agent and verifier budgets, seed 17 and retry policy are unchanged. "
            "The 1103-byte partial run.py was reconstructed from the original command and terminal echo, not an observed filesystem snapshot; "
            "original mode/hash unknown, recovery mode 0644. No interface repair. "
            "Verifier-only recovery took 13.483108 seconds: file-existence passed, five functional tests failed. "
            "Two traces directly show missing run_tasks imports; three show zero started tasks, with the common import cause inferred from source. "
            "The original missing-env warning remains. 1 CPU, 2 GiB, MTU1400, original tests and pinned readonly transport; "
            "answer and transport hashes unchanged before/after. Zero new Agent/model/control runs; all resources released."
        )
    return (
        "测试对象为Qwen3.5-35B-A3B，vLLM 0.18.0 / Ascend 0.18.0、Harbor 0.24.0 Terminus2及ARM64派生镜像。"
        "原Agent耗时332.567032秒，18次HTTP请求含17次成功和1次上下文拒绝；成功请求输入223132、输出13450 token。"
        "被拒请求至少28673输入token加4096输出预算超过32768上下文；原100turn、Agent及Verifier各900秒、seed=17及重试策略不变。"
        "1103字节run.py按原命令及终端回显确定性重建，非现场文件快照；原SHA及模式未知，0644为恢复选择，未修补接口。"
        "仅Verifier恢复耗时13.483108秒，文件存在测试通过、5项功能测试失败。两项并发测试原trace直接显示缺少run_tasks导入；"
        "三项取消测试显示启动任务数为零，其共同导入原因仅为源码推断。原env缺失警告保留。"
        "资源1CPU、2GiB、MTU1400，原测试及只读固定依赖传输不变，答案与传输前后SHA一致。新增Agent、模型调用、控制实验均为0，全部资源已释放。"
    )


def render_report(old):
    assert "报告版本：V0.92" in old
    text = old.replace(
        "报告版本：V0.92（SWE增量与Terminal评分控制）",
        "报告版本：V0.93（Terminal部分产物独立恢复评分）",
        1,
    )
    # Update current coverage only; immutable prior reports retain their original scope.
    text = text.replace(
        "已执行1/89题，该题恢复评分通过，另88题未执行",
        "OpenSSL本题恢复评分通过；当前总体覆盖见部分产物恢复评分",
    )
    text = text.replace(
        "既有OpenSSL模型结果仍为1/89",
        "既有OpenSSL本题结果保留，当前总体覆盖见部分产物恢复评分",
    )
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("|") and line.split("|")[1].strip() == "TerminalBench 2.1":
            lines[i] = (
                "| TerminalBench 2.1 | " + scope() + " | 其余87题及B1同题配对尚未完成 |"
            )
    text = "\n".join(lines) + "\n"
    marker = "#### 0.18 Native与已有优化方案同题质量"
    assert marker in text
    return text.replace(
        marker,
        "#### Terminal部分产物恢复评分\n\n"
        + scope()
        + "\n\n"
        + note()
        + "\n\n证据：[规范结果、逐测试状态及离线复核](../terminal-cancel-async-partial-recovery-20261027/README.md)。\n\n"
        + marker,
        1,
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
    assert "报告版本：V0.93" in report.decode()
    assert value["original_attempt"]["reward"] is None
    assert value["recovery"]["reward"] == 0 and value["recovery"]["passed"] == 1
    data.update(
        version="V0.93",
        source_commit=revision,
        report_sha256=hashlib.sha256(report).hexdigest(),
        report_url=f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{revision}/{REPORT}",
        terminal_cancel_partial_recovery=value,
    )
    evidence = {
        "path": SOURCE,
        "kind": "blob",
        "label": "部分产物恢复评分 / Partial-artifact grading",
    }
    for section in data["sections"]:
        if section["id"] in (
            "terminal-model-recovery",
            "terminal-cancel-async-controls",
        ):
            for key, english in (("scope", False), ("scope_en", True)):
                section[key] = (
                    section.get(key, section["scope"])
                    .replace(
                        "88 of 89 tasks remain unattempted.",
                        "Current total: 2/89 tasks attempted, 87 unattempted.",
                    )
                    .replace("另88题未执行", "当前共2/89题已尝试、87题未执行")
                )
                section[key] = (
                    section[key]
                    .replace(
                        "已执行1/89题，该题恢复评分通过，", "OpenSSL本题恢复评分通过；"
                    )
                    .replace("既有OpenSSL模型结果仍为1/89", "既有OpenSSL本题结果保留")
                    .replace(
                        "the existing OpenSSL result remains 1/89",
                        "the existing OpenSSL task result is retained",
                    )
                )
                section[key] += " " + scope(english)
        if section["id"] == "v15-remaining":
            section["evidence"].append(evidence.copy())
            for key, english in (("tables", False), ("tables_en", True)):
                for table in section[key]:
                    for row in table["rows"]:
                        if row[0] == "TerminalBench 2.1":
                            row[1] = scope(english)
                            if len(row) > 2:
                                row[2] = (
                                    "87 tasks and B1 pairing remain."
                                    if english
                                    else "87题及B1同题配对尚未完成。"
                                )
    new = {
        "id": "terminal-cancel-partial-recovery",
        "title": "Terminal部分产物评分 / Partial-artifact grading",
        "scope": scope() + " " + note(),
        "scope_en": scope(True) + " " + note(True),
        "source_heading": "Terminal部分产物恢复评分",
        "evidence": [evidence],
    }
    for key, english in (("tables", False), ("tables_en", True)):
        new[key] = [
            {
                "headers": ["Phase", "Model calls", "Tests", "Reward", "Scope"]
                if english
                else ["阶段", "模型调用", "测试", "奖励", "范围"],
                "rows": [
                    [
                        "Original Agent" if english else "原Agent",
                        "17 success + 1 rejected",
                        "not run",
                        "null",
                        "task_complete=false",
                    ],
                    [
                        "Verifier-only" if english else "仅恢复评分",
                        "0",
                        "1 / 6",
                        "0",
                        "unsubmitted partial artifact"
                        if english
                        else "未提交的部分产物",
                    ],
                ],
            }
        ]
    data["sections"].append(new)
    return data


def prepare():
    (ROOT / REPORT).write_text(render_report((ROOT / OLD_REPORT).read_text()))
    path = ROOT / "reports/four-primary-datasets-20261025/README.md"
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("|") and line.split("|")[1].strip() == "TerminalBench 2.1":
            lines[i] = (
                "| TerminalBench 2.1 | "
                + scope()
                + " | 其余87题及B1同题配对尚未完成。 |"
            )
    path.write_text("\n".join(lines) + "\n")
    directory = (ROOT / SOURCE).parent
    (directory / "README.md").write_text(
        "# TerminalBench2.1部分产物恢复评分\n\n## 结果\n\n"
        + scope()
        + "\n\n## 配置与方法\n\n"
        + note()
        + "\n\n## 证据\n\nsummary.json为规范聚合；tests.json及storage-public/ctrf-name-status.json为6项原测试状态。"
        "两个profiling-review及independent-review.json为独立复核，storage-public保留白名单投影与原件SHA索引。"
        "原题目、答案、run.py正文、请求标识及轨迹不公开；原失败、null与受限原件均保留。"
        "\n\n运行 `python3 verify.py` 可离线复核公开聚合、逐项状态及哈希；原始日志复核仍需受控原件，不等同重新执行实验。\n"
    )
    index = ROOT / "reports/vllm-hust-midterm-20261019/README.md"
    link = "- Terminal部分产物恢复评分：[规范结果](../terminal-cancel-async-partial-recovery-20261027/README.md)。"
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
