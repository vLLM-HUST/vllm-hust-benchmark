"""Project two SWE attempts and Circuit's original unscored trial."""

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import build_midterm_v088 as common
import build_midterm_v097 as terminal_previous
import build_midterm_v098 as previous

ROOT = Path(__file__).resolve().parents[1]
REPORT = previous.REPORT
OLD_REPORT = "reports/vllm-hust-midterm-20261019/中期测试报告-v098.md"
FOLDER = "reports/swe-native018-wave2-openlibrary-navidrome-20261027"
SOURCE = FOLDER + "/catalog.json"
CIRCUIT = "reports/terminal-circuit-context-20261027"
CONTROLS = "reports/terminal-circuit-controls-20261027"
DIAGNOSTIC = "reports/terminal-break-filter-byte-diagnostic-20261027"
OUTPUT = ROOT / "website-publication/data/midterm-v099.json"


def note(english=False):
    if english:
        return (
            "Two further original-protocol SWE attempts: Openlibrary 37 successful HTTP requests, "
            "submitted in 352.834124 seconds; the patch applied on three paths. "
            "Three list assertions failed: F2P 0/2, P2P 0/1, no missing names, official false. "
            "Nine internal pytest reruns came from the original image's reruns=3 setting, not repeated Agents or grading. "
            "The two original tracked fixtures and exact submodule admission were preserved. "
            "Navidrome made 62 successful requests, submitted in 533.363336 seconds, and applied its patch on ten paths. "
            "The patch introduced undefined imaging.Copy/DrawAt calls and a missing internal/storage/fs import. "
            "Compilation/setup failed; the original parser is empty and both F2P names are missing. "
            "Official false is not two observed functional assertion failures. "
            "For both tasks, post-application and pre-test identities agree; resources released, no grading model calls."
        )
    return (
        "新增两次原协议SWE尝试：Openlibrary 37次HTTP均成功、352.834124秒主动提交，补丁应用于3个路径。"
        "实际3个列表断言失败：F2P 0/2、P2P 0/1、缺失0、官方false。"
        "原镜像reruns=3造成9次pytest内部重试，不是重复Agent或评分；原两tracked fixture与精确子模块准入保持。"
        "Navidrome 62次请求均成功、533.363336秒主动提交，补丁应用于10个路径。"
        "原补丁新增未定义的imaging.Copy/DrawAt调用及缺失的internal/storage/fs导入，造成编译/setup失败；"
        "原解析器为空、F2P两项缺失，官方false不是2项功能断言阴性。"
        "两题应用后与测试前状态身份一致，资源均释放，评分阶段模型调用0。"
    )


def current_notes(english=False):
    text = previous.current_notes(english)
    old, new = (
        (
            "Current coverage is 24/731: original records 3/24, after disclosed same-answer recovery 4/24; 707 unattempted.",
            "Current coverage is 26/731: original records 3/26, after disclosed same-answer recovery 4/26; 705 unattempted.",
        )
        if english
        else (
            "当前覆盖24/731项：原记录3/24，披露同一答案恢复后4/24，707项未执行；",
            "当前覆盖26/731项：原记录3/26，披露同一答案恢复后4/26，705项未执行；",
        )
    )
    assert old in text
    return text.replace(old, new) + " " + note(english)


def circuit_note(english=False):
    if english:
        return (
            "circuit-fibsqrt ARM64 controls: fresh nop 2/3, reward 0; fresh oracle 3/3, reward 1, one each. "
            "The subsequent single 0.18/0.18 B0 Agent ended after 330.919770 seconds with ContextLengthExceededError; "
            "original verifier, reward and CTRF remain null, not 0/1. "
            "Actual HTTP: 11 requests, 10 successful and one rejected. The trajectory has nine Agent steps, "
            "ten episodes and nine latency entries. One successful request is the original output-length fallback's "
            "recursive query after truncation, not an extra Agent attempt. "
            "Trajectory tokens are 125594 input/10818 output; API metrics are 131043/14914, with a retained delta of 5449/4096. "
            "At least 28673 input plus the reserved 4096 output exceeded 32768; summarization was disabled. "
            "Neither 100 turns nor the original 3600-second Agent budget was exhausted; verifier budget also remains 3600 seconds. "
            "Bounded command/AST review found no answer-file write, and the artifact manifest is empty; "
            "the final answer-file hash is unknown. Printed counts cannot reconstruct an answer: no generator replay, "
            "answer completion or recovery scoring. Original failures and nulls remain; CPU and model resources released."
        )
    return (
        "circuit-fibsqrt的ARM64原控制各一次：新鲜nop 2/3、reward=0，新鲜oracle 3/3、reward=1。"
        "随后唯一0.18/0.18 B0 Agent运行330.919770秒后发生ContextLengthExceededError，"
        "原verifier、reward和CTRF仍为null，不是0/1。实际11次HTTP包括10次成功和1次拒绝；"
        "轨迹为9个Agent步骤、10个episode和9项延迟，另一次成功来自原输出截断处理的递归请求，不是第二次Agent。"
        "轨迹输入/输出token为125594/10818，API为131043/14914，差额5449/4096单列保留。"
        "至少28673输入加4096输出预留超过32768上下文，原summarization关闭；不是100turn或3600秒耗尽，"
        "原verifier预算同为3600秒。有限逐命令及AST审查未发现答案文件写入，artifact清单为空，终态答案文件SHA未知。"
        "不能从打印计数重建答案，未重跑或补完生成器，未恢复评分；原失败/null保留，CPU与模型资源已释放。"
    )


def terminal_scope(english=False):
    text = terminal_previous.terminal_scope(english)
    old, new = (
        (
            "3/89 unique tasks attempted; 86 unattempted",
            "4/89 unique tasks attempted; 85 unattempted",
        )
        if english
        else ("已尝试3/89题，86题未执行", "已尝试4/89题，85题未执行")
    )
    assert old in text
    return text.replace(old, new) + " " + circuit_note(english)


def historical_control_note(english=False):
    text = previous.control_note(english)
    return (
        text.replace(
            "coverage stays 3/89, 86 unattempted.",
            "coverage at that control snapshot was 3/89, 86 unattempted.",
        )
        if english
        else text.replace(
            "覆盖仍为3/89、86项未执行。", "该控制快照时覆盖为3/89、86项未执行。"
        )
    )


def diagnostic_note(english=False):
    if english:
        return (
            "A separate bounded break-filter byte diagnostic reused the same built image and original oracle/filter/test sources. "
            "The original oracle output was 41 bytes; filter exit 0 produced 20 bytes without the literal script/alert tokens. "
            "This supports the filter-path explanation in the new diagnostic environment, not reconstruction of the removed "
            "original browser DOM or proof of a particular version/ARM64 cause. No browser session, original control rerun, "
            "Agent or model call. The diagnostic's one pytest pass is not a successful positive control: original reward 0, "
            "five-second alert threshold and r1 null remain. Runtime/source identities and all 34 sealed files were verified; resources released."
        )
    return (
        "另行有界break-filter字节诊断复用同一已建镜像和原oracle/filter/test源码。"
        "原oracle输出41字节，经原filter退出0后得到20字节，literal script/alert标记消失。"
        "这支持新诊断环境中的过滤路径解释，不是已删除原浏览器DOM的重建，也未证明特定版本或ARM64根因。"
        "没有浏览器session、原控制重跑、Agent或模型调用；诊断pytest 1项通过不是正控制通过，"
        "原reward=0、5秒alert阈值和r1 null均保留。实际运行身份与34项封存核验，资源已释放。"
    )


def rows(value, english=False):
    result = previous.rows(value, english)
    labels = {
        "swe018-wave2-06-r1": (
            "3个列表断言失败；9次原pytest内部重试",
            "Three list assertions failed; nine internal pytest reruns",
        ),
        "swe018-wave2-07-r1": (
            "补丁编译/setup失败；两F2P缺失",
            "Patch compilation/setup failed; two F2P names missing",
        ),
    }
    for entry, row in zip(value["entries"], result, strict=True):
        if entry["run_id"] in labels:
            row[-1] = labels[entry["run_id"]][int(english)]
    return result


def render_report(old, value, prior):
    assert "报告版本：V0.98" in old
    text = old.replace(
        "报告版本：V0.98（SWE增量与Terminal正控制边界）",
        "报告版本：V0.99（SWE增量与Circuit原始上下文终止）",
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
    text = text.replace(terminal_previous.terminal_scope(), terminal_scope())
    text = text.replace(previous.control_note(), historical_control_note())
    lines = []
    for line in text.splitlines():
        if line.startswith("| SWE-bench-Pro "):
            line = (
                "| SWE-bench-Pro | "
                + common.summary(value)
                + " 原记录3/26；同答案恢复4/26，共26次尝试。 | 705题及B1同题配对未完成 |"
            )
        elif line.startswith("| TerminalBench 2.1 "):
            line = (
                "| TerminalBench 2.1 | "
                + terminal_scope()
                + " | 85题及B1同题配对未完成；Circuit原评分null，break-filter正控制未通过 |"
            )
        lines.append(line)
    text = "\n".join(lines) + "\n"
    marker = "#### 0.18 Native与已有优化方案同题质量"
    addition = (
        "#### 后续两项与Circuit原始结果\n\n"
        + note()
        + "\n\n"
        + circuit_note()
        + "\n\n"
        + diagnostic_note()
        + "\n\n证据：[SWE累计26项](../swe-native018-wave2-openlibrary-navidrome-20261027/README.md)、"
        "[Circuit原始结果](../terminal-circuit-context-20261027/README.md)、"
        "[Circuit控制](../terminal-circuit-controls-20261027/README.md)、"
        "[break-filter字节诊断](../terminal-break-filter-byte-diagnostic-20261027/README.md)。\n\n"
    )
    assert marker in text
    return text.replace(marker, addition + marker, 1)


def build(revision, read=None):
    if read is None:

        def read(path):
            return subprocess.check_output(
                ["git", "show", f"{revision}:{path}"], cwd=ROOT
            )

    if "报告版本：V1.01" in read(REPORT).decode():
        original_read = read

        def read(path):
            if path == REPORT:
                path = REPORT.replace(".md", "-v099.md")
            return original_read(path)

    data = previous.build(revision, lambda p: read(OLD_REPORT if p == REPORT else p))
    value = json.loads(read(SOURCE))
    prior = json.loads(read(previous.SOURCE))
    assert value["entries"][:24] == prior["entries"]
    assert (
        value["attempted_unique_tasks"],
        value["recorded_official_resolved"],
        value["unattempted_tasks"],
    ) == (26, 3, 705)
    report = read(REPORT)
    assert "报告版本：V0.99" in report.decode()
    data.update(
        version="V0.99",
        source_commit=revision,
        report_sha256=hashlib.sha256(report).hexdigest(),
        report_url=f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{revision}/{REPORT}",
        swe_incremental_tasks=value,
        swe_same_answer_recovery_aggregate=json.loads(
            read(FOLDER + "/current-aggregate.json")
        ),
        terminal_circuit_original_model=json.loads(read(CIRCUIT + "/summary.json")),
        terminal_circuit_request_accounting=json.loads(
            read(CIRCUIT + "/request-accounting.json")
        ),
        terminal_circuit_controls=json.loads(read(CONTROLS + "/summary.json")),
        terminal_break_filter_byte_diagnostic=json.loads(
            read(DIAGNOSTIC + "/summary.json")
        ),
    )
    for section in data["sections"]:
        for key, english in (("scope", False), ("scope_en", True)):
            if key not in section:
                continue
            section[key] = section[key].replace(
                terminal_previous.terminal_scope(english), terminal_scope(english)
            )
            section[key] = section[key].replace(
                previous.control_note(english), historical_control_note(english)
            )
            if section["id"] in {
                "swe-single-agent-observation",
                "swe-test-source-recovery",
            }:
                section[key] = section[key].replace(
                    previous.current_notes(english), current_notes(english)
                )
                section[key] = section[key].replace(
                    common.summary(prior, english), common.summary(value, english)
                )
    section = next(
        s for s in data["sections"] if s["id"] == "swe-single-agent-observation"
    )
    remaining = next(s for s in data["sections"] if s["id"] == "v15-remaining")
    for key, english in (("tables", False), ("tables_en", True)):
        section[key][0]["rows"] = rows(value, english)
        for table in remaining[key]:
            for row in table["rows"]:
                if row[0] == "SWE-bench-Pro":
                    row[1] = (
                        common.summary(value, english) + " " + current_notes(english)
                    )
                    row[2] = (
                        "705 tasks and B1 pairing remain."
                        if english
                        else "705题及B1同题配对未完成。"
                    )
                elif row[0] == "TerminalBench 2.1":
                    row[1] = terminal_scope(english)
                    row[2] = (
                        "85 tasks and B1 pairing remain; Circuit unscored, break-filter positive control failed."
                        if english
                        else "85题及B1同题配对未完成；Circuit未评分，break-filter正控制未通过。"
                    )
    links = [
        {"path": SOURCE, "kind": "blob", "label": "26项累计结果 / 26-attempt catalog"},
        {
            "path": CIRCUIT + "/summary.json",
            "kind": "blob",
            "label": "Circuit原始null / Original unscored trial",
        },
        {
            "path": CONTROLS + "/summary.json",
            "kind": "blob",
            "label": "Circuit原控制 / Original controls",
        },
        {
            "path": DIAGNOSTIC + "/summary.json",
            "kind": "blob",
            "label": "过滤字节诊断 / Filter-byte diagnostic",
        },
    ]
    section["evidence"].insert(0, copy.deepcopy(links[0]))
    remaining["evidence"] = copy.deepcopy(links) + remaining["evidence"]
    for section_id, title, describe, evidence in (
        (
            "terminal-circuit-model",
            "Terminal Circuit原始结果 / Original trial",
            circuit_note,
            links[1:3],
        ),
        (
            "terminal-break-filter-byte-diagnostic",
            "Terminal过滤字节诊断 / Filter-byte diagnostic",
            diagnostic_note,
            links[3:],
        ),
    ):
        data["sections"].append(
            dict(
                id=section_id,
                title=title,
                scope=describe(),
                scope_en=describe(True),
                tables=[],
                tables_en=[],
                evidence=copy.deepcopy(evidence),
                source_heading="后续两项与Circuit原始结果",
            )
        )
    return data


def prepare():
    old = ROOT / OLD_REPORT
    if not old.exists():
        raw = (ROOT / REPORT).read_bytes()
        assert "报告版本：V0.98" in raw.decode()
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
