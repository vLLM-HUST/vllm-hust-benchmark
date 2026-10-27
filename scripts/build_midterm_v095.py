"""Project original-answer recovery and bounded Cython diagnosis without new attempts."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import build_midterm_v094 as previous

ROOT = Path(__file__).resolve().parents[1]
REPORT = previous.REPORT
OLD_REPORT = "reports/vllm-hust-midterm-20261019/中期测试报告-v094.md"
RECOVERY = "reports/swe-protonmail-dependency-recovery-20261027/summary.json"
TRACE = "reports/terminal-cython-single-trace-20261027/summary.json"
OUTPUT = ROOT / "website-publication/data/midterm-v095.json"


def recovery_note(english=False):
    if english:
        return (
            "Protonmail original submitted answer: after restoring the unchanged original test and building "
            "the original canvas 2.11.2 against Node 18.20.8 / ABI108 in an explicitly identified ARM64 dependency-derived image, "
            "the official verifier reports true: three tests passed, F2P 1/1 and P2P 1/1, no missing targets. "
            "The 91 required offline deb packages add no upgrades or removals; original source, lock, patch, tests, parser and budgets remain unchanged. "
            "The drive suite passed; the unchanged run_script also reports No tests found in the mail workspace. "
            "Grading process exit 0 and outer controller exit 1 from a post-score cleanup assertion are both retained; "
            "separate release audit confirms API404, no containers or owned processes. No repeated grading or new Agent/model calls. "
            "Original missing-test and canvas-load records remain false with zero tests. "
            "Same 17/731 attempts: original records 2/17, after disclosed same-answer recovery 3/17; 714 unattempted. "
            "This is not an eighteenth attempt or a full-dataset/B0-B1 paired result."
        )
    return (
        "Protonmail复用原已提交答案，恢复不变的原测试，并在明确标识的ARM64依赖派生镜像中构建原canvas 2.11.2 "
        "与Node 18.20.8 / ABI108后，官方评分true：3项测试通过，F2P 1/1、P2P 1/1，缺失0。"
        "离线新增91项必需deb，0升级、0删除；原源码、锁文件、补丁、测试、解析器及预算不变。"
        "drive套件通过；原run_script另在mail路径记录No tests found，原输出保留。"
        "评分子进程退出0，外层控制器因评分后的清理断言退出1；独立释放审计确认API404、容器及所属进程消失，原异常保留，未重评分。"
        "新增Agent和模型调用均为0。原测试缺失和canvas加载失败的false、零测试记录保留。"
        "仍为17/731项尝试：原记录2/17，披露同一答案恢复后3/17，714项未执行；不是第18次尝试或全量、B0/B1配对通过率。"
    )


def trace_note(english=False):
    if english:
        return (
            "Cython original controls remain nop 1/11 and oracle 10/11, both reward 0. "
            "A single pinned-source/dependency reconstruction recovered KeyError('pos') in representation.py:338 "
            "after planarity.networkx_graph(pg): observed attributes are vertex_end, vertex_position and vertex_start. "
            "This identifies an attribute interface mismatch in this reconstruction, not the original terminated container traceback "
            "or proof of architecture causation/upgrade history. Original wheel/environment bytes were unavailable; "
            "local baseline Git identity and no-build-isolation are disclosed. Setup exit 0, diagnostic test exit 1; containers released. "
            "No test/oracle changes, new control score or model coverage. Terminal remains 2/89 attempted, 87 unattempted; "
            "this task's positive control still fails."
        )
    return (
        "Cython原控制仍为nop 1/11、oracle 10/11，reward均为0。"
        "一次固定源码与依赖重建取得单项完整trace：planarity.networkx_graph(pg)之后，"
        "representation.py:338读取pos触发KeyError；实际属性为vertex_end、vertex_position和vertex_start。"
        "该结果定位本次重建的属性接口不匹配，不冒充原终止容器的完整trace，不证明架构因果或版本升级历史。"
        "原wheel及环境字节不可恢复，本地Git基线与no-build-isolation差异已披露。"
        "setup退出0、诊断测试退出1，容器已释放；未改测试或oracle，不新增控制成绩或模型覆盖。"
        "Terminal仍为2/89项已尝试、87项未执行，本题正控制仍未通过。"
    )


def current_notes(english=False):
    old = previous.notes(english)
    start = old.index("qutebrowser")
    return recovery_note(english) + " " + old[start:]


def render_report(old):
    assert "报告版本：V0.94" in old
    text = old.replace(
        "报告版本：V0.94（SWE增量及评分依赖、Terminal控制）",
        "报告版本：V0.95（SWE原答案依赖恢复与Terminal诊断）",
        1,
    )
    assert previous.notes() in text and previous.cython_note() in text
    text = text.replace(previous.notes(), current_notes()).replace(
        previous.cython_note(), trace_note()
    )
    text = text.replace(
        "原测试依赖及部分补丁边界见本轮增量结果。",
        "原记录2/17；同一答案恢复后3/17，仍为17次尝试。",
    )
    text = text.replace("原测试缺失；恢复评分另列", "原false保留；同答案恢复true")
    marker = "#### 0.18 Native与已有优化方案同题质量"
    addition = (
        "#### 依赖恢复与诊断证据\n\n"
        "| 项目 | 原结果 | 本次结果 | 新增Agent |\n"
        "|---|---|---|---:|\n"
        "| Protonmail | false、测试0项 | 原答案恢复评分true；3项通过，F2P 1/1、P2P 1/1 | 0 |\n"
        "| Cython | oracle 10/11、reward 0 | 单项重建KeyError(pos)；原控制成绩不变 | 0 |\n\n"
        "证据：[Protonmail依赖恢复](../swe-protonmail-dependency-recovery-20261027/README.md)、"
        "[Cython单项诊断](../terminal-cython-single-trace-20261027/README.md)。\n\n"
    )
    assert marker in text
    return text.replace(marker, addition + marker, 1)


def build(revision, read=None):
    if read is None:

        def read(path):
            return subprocess.check_output(
                ["git", "show", f"{revision}:{path}"], cwd=ROOT
            )

    data = previous.build(revision, lambda p: read(OLD_REPORT if p == REPORT else p))
    report = read(REPORT)
    recovery, trace = json.loads(read(RECOVERY)), json.loads(read(TRACE))
    assert "报告版本：V0.95" in report.decode()
    a = recovery["aggregate_distinction"]
    assert a["attempted_tasks"] == a["recovery_denominator"] == 17
    assert (
        a["original_official_positives"] == 2
        and a["positives_after_disclosed_recovery"] == 3
    )
    assert a["unattempted_tasks"] == 714
    assert recovery["derived_dependency_recovery_result"]["official"] is True
    assert (
        recovery["Agent_attempts_added"]
        == recovery["model_requests_added"]
        == recovery["task_attempts_added"]
        == 0
    )
    assert recovery["controller_exit"] == 1 and recovery["grading_process_exit"] == 0
    assert (
        trace["new_model_calls"]
        == trace["new_Agent_attempts"]
        == trace["new_control_scores"]
        == 0
    )
    assert (
        trace["task_reward"] is None and trace["original_control"]["oracle_reward"] == 0
    )
    assert (
        not trace["original_trace_recovered"]
        and not trace["model_qualification_passed"]
    )
    data.update(
        version="V0.95",
        source_commit=revision,
        report_sha256=hashlib.sha256(report).hexdigest(),
        report_url=f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{revision}/{REPORT}",
        protonmail_dependency_recovery=recovery,
        terminal_cython_single_trace=trace,
        swe_same_answer_recovery_aggregate=a,
    )
    evidence = [
        {
            "path": RECOVERY,
            "kind": "blob",
            "label": "原答案依赖恢复 / Same-answer dependency recovery",
        },
        {
            "path": TRACE,
            "kind": "blob",
            "label": "Cython单项诊断 / Cython bounded diagnosis",
        },
    ]
    swe = next(s for s in data["sections"] if s["id"] == "swe-single-agent-observation")
    dep = next(s for s in data["sections"] if s["id"] == "swe-test-source-recovery")
    cython = next(s for s in data["sections"] if s["id"] == "terminal-cython-controls")
    remaining = next(s for s in data["sections"] if s["id"] == "v15-remaining")
    for key, english in (("scope", False), ("scope_en", True)):
        assert previous.notes(english) in swe[key]
        swe[key] = swe[key].replace(previous.notes(english), current_notes(english))
        dep[key] = current_notes(english)
        cython[key] = trace_note(english)
    for key, english in (("tables", False), ("tables_en", True)):
        old_rows = swe[key][0]["rows"]
        for entry, row in zip(
            data["swe_incremental_tasks"]["entries"], old_rows, strict=True
        ):
            if entry["run_id"] == "swe018-repo-08-r1":
                row[-1] = (
                    "Original false; same-answer recovery true"
                    if english
                    else "原false保留；同答案恢复true"
                )
        dep[key][0]["rows"].append(
            [
                "Native dependency recovered" if english else "原生依赖修复后恢复评分",
                "3 / 3",
                "true",
                "Same answer, zero new attempts" if english else "同一答案，新增尝试0",
            ]
        )
        for table in remaining[key]:
            for row in table["rows"]:
                if row[0] == "SWE-bench-Pro":
                    row[1] = current_notes(english)
                    if len(row) > 2:
                        row[2] = (
                            "714 tasks and B1 pairing remain."
                            if english
                            else "714题及B1同题配对未完成。"
                        )
                elif row[0] == "TerminalBench 2.1":
                    row[1] = (
                        previous.previous.scope(english) + " " + trace_note(english)
                    )
    swe["evidence"].insert(0, evidence[0])
    dep["evidence"].insert(0, evidence[0])
    cython["evidence"].insert(0, evidence[1])
    remaining["evidence"].extend(evidence)
    return data


def prepare():
    old = ROOT / OLD_REPORT
    if not old.exists():
        raw = (ROOT / REPORT).read_bytes()
        assert "报告版本：V0.94" in raw.decode()
        old.write_bytes(raw)
    (ROOT / REPORT).write_text(render_report(old.read_text()))
    index = ROOT / "reports/four-primary-datasets-20261025/README.md"
    lines = index.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("|") and line.split("|")[1].strip() == "SWE-bench-Pro":
            lines[i] = (
                "| SWE-bench-Pro | 17/731项已尝试；原官方2/17，同一Protonmail答案依赖恢复后3/17。1题应用证据受限、2题确定应用失败保留；未决0不代表所有预期测试齐全，不是全量或B0/B1配对 | 714题及B1同题配对未完成 |"
            )
        elif line.startswith("|") and line.split("|")[1].strip() == "TerminalBench 2.1":
            lines[i] = (
                "| TerminalBench 2.1 | "
                + previous.previous.scope()
                + " "
                + trace_note()
                + " | 87题及B1同题配对未完成 |"
            )
    index.write_text("\n".join(lines) + "\n")
    index = ROOT / "reports/vllm-hust-midterm-20261019/README.md"
    link = "- V0.95：[Protonmail同答案依赖恢复](../swe-protonmail-dependency-recovery-20261027/README.md)；[Cython单项诊断](../terminal-cython-single-trace-20261027/README.md)。"
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
