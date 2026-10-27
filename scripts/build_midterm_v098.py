"""Project three further SWE attempts and a failed Terminal positive control."""

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import build_midterm_v088 as common
import build_midterm_v096 as tables
import build_midterm_v097 as previous

ROOT = Path(__file__).resolve().parents[1]
REPORT = previous.REPORT
OLD_REPORT = "reports/vllm-hust-midterm-20261019/中期测试报告-v097.md"
FOLDER = "reports/swe-native018-wave2-followup-20261027"
SOURCE = FOLDER + "/catalog.json"
TERMINAL = "reports/terminal-break-filter-controls-20261027"
OUTPUT = ROOT / "website-publication/data/midterm-v098.json"


def note(english=False):
    if english:
        return (
            "Three further original-protocol attempts: Flipt 67 successful HTTP requests, submitted; "
            "the patch applied, but a field declaration outside a type caused compilation failure. "
            "Official false, parser empty, four F2P names missing: not four functional assertion failures. "
            "Vuls made 20 HTTP requests: 19 successful replies and one 64K context rejection. "
            "It did not submit; the empty patch failed git apply with exit 128, and affected paths and file states remain null. "
            "Tests nevertheless ran: F2P 0/6, P2P 1/1, no missing names, official false. "
            "This does not establish testing of an applied candidate patch. "
            "Teleport made 48 successful requests in 210.730896 seconds and submitted a 2266-byte patch. "
            "Apply succeeded on two paths and post-application/pre-test identities agree. "
            "The required target executed and failed its error-path string assertion: F2P 0/1, P2P 0 targets, missing 0, official false. "
            "This is an observed assertion failure, not a compilation failure. "
            "No repeated Agents, modified patches or model calls during grading; resources released."
        )
    return (
        "新增三次原协议尝试：Flipt 67次HTTP均成功、主动提交；补丁应用成功，但类型声明外的字段造成编译失败。"
        "官方false、解析结果为空、F2P缺失4项，不是4项功能断言失败。"
        "Vuls共20次HTTP：19次成功回复和1次64K上下文拒绝；未提交，空补丁git apply退出128，"
        "受影响路径及文件状态仍为null。原测试实际执行：F2P 0/6、P2P 1/1、缺失0、官方false；"
        "不能视为已应用候选补丁的功能验证。"
        "Teleport 48次请求均成功、210.730896秒、主动提交2266字节补丁；两路径应用成功，应用后与测试前身份一致。"
        "必要目标实际执行，错误路径字符串断言失败：F2P 0/1、P2P集合为空、缺失0、官方false；不是编译失败。"
        "未重复Agent、修改补丁或在评分阶段调用模型，资源已释放。"
    )


def current_notes(english=False):
    text = previous.current_notes(english)
    if english:
        old = "Including the three new wave2 tasks, current coverage is 21/731: original records 3/21, after disclosed same-answer recovery 4/21; 710 unattempted."
        new = "Current coverage is 24/731: original records 3/24, after disclosed same-answer recovery 4/24; 707 unattempted."
    else:
        old = "计入第二批前三项新任务后，当前覆盖21/731项：原记录3/21，披露同一答案恢复后4/21，710项未执行；"
        new = "当前覆盖24/731项：原记录3/24，披露同一答案恢复后4/24，707项未执行；"
    assert old in text
    return text.replace(old, new) + " " + note(english)


def control_note(english=False):
    if english:
        return (
            "break-filter-js-from-html ARM64 controls, not a model attempt: original r1 built successfully but its "
            "mount guard rejected the original Harbor log mounts; nop exit 130, reward null, no CTRF, oracle not run. "
            "The isolated r2 reused the same built image and precisely admitted three original RW trial-log binds plus one RO dependency bind. "
            "Fresh nop: 0/1, reward 0; fresh oracle: 0/1, reward 0. The positive control failed after the original five-second alert wait; "
            "the test did execute. Original task, oracle, 1200-second verifier budget and threshold were unchanged. "
            "Temporary filtered HTML and browser DOM were not retained, so parser, browser, timing or ARM64 causation is not established. "
            "Both containers released; the original preflight path-read failure is retained. No new Agent/model task; coverage stays 3/89, 86 unattempted."
        )
    return (
        "break-filter-js-from-html为ARM64环境控制，不是模型实测。原r1构建成功，但挂载检查误拒Harbor原日志挂载，"
        "nop退出130、reward=null、无CTRF，oracle未运行。隔离r2复用同一已建镜像，精确接纳三个原trial日志RW挂载和一个依赖RO挂载。"
        "两个新鲜控制：nop 0/1、reward=0；oracle 0/1、reward=0。正控制在原5秒alert等待后断言失败，测试确实执行；"
        "原任务、oracle、1200秒verifier预算及阈值不变。未保留临时过滤HTML和浏览器DOM，尚不能定性为parser、浏览器、时序或ARM64原因。"
        "两个容器均释放，原preflight路径读取失败保留。新增Agent和模型任务均为0，覆盖仍为3/89、86项未执行。"
    )


def rows(value, english=False):
    result = tables.rows(value, english)
    labels = {
        "swe018-wave2-03-r1": (
            "补丁已应用；编译失败，目标未执行",
            "Applied; compilation failed, targets not run",
        ),
        "swe018-wave2-04-r1": (
            "20 HTTP含1拒绝；空补丁应用128，状态null",
            "20 HTTP including 1 rejection; empty apply128, states null",
        ),
        "swe018-wave2-05-r1": (
            "补丁已应用；目标字符串断言失败",
            "Applied; target string assertion failed",
        ),
    }
    for entry, row in zip(value["entries"], result, strict=True):
        if entry["run_id"] in labels:
            row[-1] = labels[entry["run_id"]][int(english)]
    return result


def render_report(old, value, prior):
    assert "报告版本：V0.97" in old
    text = old.replace(
        "报告版本：V0.97（SWE第二批增量与Terminal Cython实测）",
        "报告版本：V0.98（SWE增量与Terminal正控制边界）",
        1,
    )
    start = text.index("| 任务序号 | 官方原记录 |")
    end = text.index("\n\n", start)
    table = [
        "| 任务序号 | 官方原记录 | 模型调用 | Agent秒 | 状态 | 证据范围 |",
        "|---|---:|---:|---:|---|---|",
    ]
    table += ["| " + " | ".join(row) + " |" for row in rows(value)]
    text = text[:start] + "\n".join(table) + text[end:]
    start = text.index(
        "Protonmail复用原已提交答案", text.index("#### SWE评分依赖与未提交部分补丁")
    )
    end = text.index("\n\n", start)
    assert " ".join(text[start:end].split()) == " ".join(
        previous.current_notes().split()
    )
    text = text[:start] + current_notes() + text[end:]
    text = text.replace(common.summary(prior), common.summary(value))
    lines = []
    for line in text.splitlines():
        if line.startswith("| SWE-bench-Pro "):
            line = (
                "| SWE-bench-Pro | "
                + common.summary(value)
                + " 原记录3/24；同答案恢复后4/24，共24次尝试。 | 707题及B1同题配对未完成 |"
            )
        elif line.startswith("| TerminalBench 2.1 "):
            line = (
                "| TerminalBench 2.1 | "
                + previous.terminal_scope()
                + " "
                + control_note()
                + " | 86题及B1同题配对未完成；break-filter正控制未通过 |"
            )
        lines.append(line)
    text = "\n".join(lines) + "\n"
    marker = "#### 0.18 Native与已有优化方案同题质量"
    addition = (
        "#### 后续三项与break-filter控制\n\n"
        + note()
        + "\n\n"
        + control_note()
        + "\n\n证据：[SWE后续三项](../swe-native018-wave2-followup-20261027/README.md)、[break-filter控制](../terminal-break-filter-controls-20261027/README.md)。\n\n"
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
    value = json.loads(read(SOURCE))
    prior = json.loads(read(previous.SOURCE))
    assert value["entries"][:21] == prior["entries"]
    assert (
        value["attempted_unique_tasks"],
        value["recorded_official_resolved"],
        value["unattempted_tasks"],
    ) == (24, 3, 707)
    report = read(REPORT)
    assert "报告版本：V0.98" in report.decode()
    controls = json.loads(read(TERMINAL + "/summary.json"))
    data.update(
        version="V0.98",
        source_commit=revision,
        report_sha256=hashlib.sha256(report).hexdigest(),
        report_url=f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{revision}/{REPORT}",
        swe_incremental_tasks=value,
        swe_same_answer_recovery_aggregate=json.loads(
            read(FOLDER + "/current-aggregate.json")
        ),
        terminal_break_filter_controls=controls,
    )
    section = next(
        s for s in data["sections"] if s["id"] == "swe-single-agent-observation"
    )
    dep = next(s for s in data["sections"] if s["id"] == "swe-test-source-recovery")
    remaining = next(s for s in data["sections"] if s["id"] == "v15-remaining")
    for key, english in (("scope", False), ("scope_en", True)):
        assert previous.current_notes(english) in section[key]
        section[key] = (
            section[key]
            .replace(previous.current_notes(english), current_notes(english))
            .replace(common.summary(prior, english), common.summary(value, english))
        )
        dep[key] = current_notes(english)
    for key, english in (("tables", False), ("tables_en", True)):
        section[key][0]["rows"] = rows(value, english)
        for table in remaining[key]:
            for row in table["rows"]:
                if row[0] == "SWE-bench-Pro":
                    row[1] = (
                        common.summary(value, english) + " " + current_notes(english)
                    )
                    row[2] = (
                        "707 tasks and B1 pairing remain."
                        if english
                        else "707题及B1同题配对未完成。"
                    )
                elif row[0] == "TerminalBench 2.1":
                    row[1] = (
                        previous.terminal_scope(english) + " " + control_note(english)
                    )
                    row[2] = (
                        "86 tasks and B1 pairing remain; break-filter positive control failed."
                        if english
                        else "86题及B1同题配对未完成；break-filter正控制未通过。"
                    )
    links = [
        {"path": SOURCE, "kind": "blob", "label": "24项累计结果 / 24-attempt catalog"},
        {
            "path": TERMINAL + "/summary.json",
            "kind": "blob",
            "label": "break-filter环境控制 / Environment controls",
        },
    ]
    section["evidence"].insert(0, copy.deepcopy(links[0]))
    remaining["evidence"] = copy.deepcopy(links) + remaining["evidence"]
    data["sections"].append(
        dict(
            id="terminal-break-filter-controls",
            title="Terminal break-filter环境控制 / Environment controls",
            scope=control_note(),
            scope_en=control_note(True),
            tables=[],
            tables_en=[],
            evidence=copy.deepcopy(links[1:]),
            source_heading="后续三项与break-filter控制",
        )
    )
    return data


def prepare():
    old = ROOT / OLD_REPORT
    if not old.exists():
        raw = (ROOT / REPORT).read_bytes()
        assert "报告版本：V0.97" in raw.decode()
        old.write_bytes(raw)
    (ROOT / REPORT).write_text(
        render_report(
            old.read_text(),
            json.loads((ROOT / SOURCE).read_text()),
            json.loads((ROOT / previous.SOURCE).read_text()),
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", "--source-commit", dest="revision")
    parser.add_argument(
        "--prepare-report", "--prepare", dest="prepare", action="store_true"
    )
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        assert args.revision
        OUTPUT.write_text(
            json.dumps(build(args.revision), ensure_ascii=False, indent=2) + "\n"
        )
