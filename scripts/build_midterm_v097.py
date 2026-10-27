"""Project bounded wave2 tasks and Cython controls/original context failure."""

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import build_midterm_v088 as common
import build_midterm_v093 as terminal_previous
import build_midterm_v096 as previous

ROOT = Path(__file__).resolve().parents[1]
REPORT = previous.REPORT
OLD_REPORT = "reports/vllm-hust-midterm-20261019/中期测试报告-v096.md"
FOLDER = "reports/swe-native018-wave2-first-three-20261027"
SOURCE = FOLDER + "/catalog.json"
TERMINAL = "reports/terminal-cython-context-20261027"
OUTPUT = ROOT / "website-publication/data/midterm-v097.json"


def wave_note(english=False):
    if english:
        return (
            "Three further original-protocol attempts: NodeBB 54 calls / 265.495685 seconds, "
            "Ansible 36 / 361.464713 seconds, Element 96 / 511.628888 seconds; all submitted. "
            "All patches applied, with identical post-application and actual pre-test states. "
            "NodeBB official false: F2P 0/3, P2P 252/252; raw Mocha 262 pass / 3 fail versus parser 252 pass / 3 fail. "
            "Ansible official true: F2P 4/4, P2P 26/26, no missing targets. "
            "Element official false: raw Jest 1 pass / 2 snapshot failures; the original PASS-file-only parser "
            "omitted the FAIL file, returning no tests and F2P missing 3. Missing 3 is not three observed assertion failures or no execution. "
            "No parser changes, repeated Agents or grading model calls; resources released."
        )
    return (
        "新增三次原协议尝试：NodeBB 54次调用、265.495685秒，Ansible 36次、361.464713秒，"
        "Element 96次、511.628888秒，均主动提交。三份补丁均成功应用，应用后与测试前实际待评分状态一致。"
        "NodeBB官方false，F2P 0/3、P2P 252/252；原Mocha 262通过/3失败与原解析器252通过/3失败分列。"
        "Ansible官方true，F2P 4/4、P2P 26/26，目标缺失0。"
        "Element官方false：原Jest 1通过/2快照失败，原解析器仅收PASS文件块，略过FAIL文件后输出为空、F2P缺失3。"
        "缺失3不是三个实测断言失败，也不是测试未执行。不改解析器、不重复Agent，评分模型调用0；资源已释放。"
    )


def current_notes(english=False):
    text = previous.current_notes(english)
    if english:
        old = "With the separate new Tutanota attempt, current coverage is 18/731: original records 2/18, after disclosed same-answer recovery 3/18; 713 unattempted."
        new = "Including the three new wave2 tasks, current coverage is 21/731: original records 3/21, after disclosed same-answer recovery 4/21; 710 unattempted."
    else:
        old = "计入另行新增的Tutanota尝试后，当前覆盖18/731项：原记录2/18，披露同一答案恢复后3/18，713项未执行；"
        new = "计入第二批前三项新任务后，当前覆盖21/731项：原记录3/21，披露同一答案恢复后4/21，710项未执行；"
    assert old in text
    return text.replace(old, new) + " " + wave_note(english)


def cython_note(english=False):
    if english:
        return (
            "Original Cython oracle 10/11 and reward 0 are retained. The pinned ARM64 environment changes only "
            "planarity 1.0.0 to source-built 0.6, preserving the other 24 versions and the original eight task files, "
            "11 tests and oracle. Fresh nop: 1/11, reward 0; fresh oracle: 11/11, reward 1. Controls add no model coverage. "
            "One designated 0.18/0.18 B0 Agent ran 478.118260 seconds: 29 successful replies plus one context-limit rejection "
            "(30 HTTP requests and 30 episodes), not 100 turns or 900 seconds exhausted. Original 32K context, "
            "4096 output budget, seed 17 and retry policy remain unchanged. Verifier did not start; reward and CTRF are null. "
            "Resources released. Exact replay of 11 recorded edits over 144 verified Git blobs matches the captured diff in nine files; "
            "this establishes reusable tracked source, not original native binaries, installed state or a complete filesystem snapshot. "
            "Separately, nine necessary original build/install/copy commands reconstructed the same final source in 96.468339 seconds; "
            "the original verifier ran 9.089041 seconds, 10/11 passed, one failed, none missing, reward 0. "
            "test_ccomplexity reports missing NumPy int: the unchanged answer still uses np.int with frozen NumPy 2.3.0. "
            "The local baseline Git commit and new native/install artifacts are derived, not the original snapshot. "
            "Source hashes/modes and native hashes remained stable; resources released. No new Agent, model calls or controls; original null retained."
        )
    return (
        "Cython原oracle 10/11、reward=0保留。固定ARM64环境仅将planarity 1.0.0换为原源码构建的0.6，"
        "其余24版本及原8任务文件、11测试、oracle不变。两个新鲜控制分别为nop 1/11、reward=0和oracle 11/11、reward=1；"
        "控制不增加模型覆盖。唯一指定0.18/0.18 B0 Agent耗时478.118260秒，29次成功回复加1次上下文拒绝，共30次HTTP、episode计数30；"
        "不是100turn或900秒耗尽。原32K上下文、4096输出预算、seed=17及重试策略不变。verifier未启动，reward与CTRF均为null，资源已释放。"
        "核验144个Git blob并重放原11条编辑后，9文件差异精确等于原采集值；只证明受跟踪源码可复用，不是原native二进制、安装态或完整现场快照。"
        "另列派生重建：9条必要的原安装/构建/复制命令耗时96.468339秒，原verifier运行9.089041秒，"
        "11项完整执行、10通过1失败、缺失0，reward=0。test_ccomplexity实际报告NumPy无int属性，原答案在固定NumPy2.3.0下仍调用np.int。"
        "本地Git基线及新native/安装产物明确属于派生，并非原现场快照；源码SHA/模式和native前后稳定，资源已释放。"
        "新增Agent、模型调用及控制均为0，原context失败/null不覆盖。"
    )


def terminal_scope(english=False):
    old = terminal_previous.scope(english)
    if english:
        old = old.replace(
            "2/89 unique tasks attempted; 87 unattempted",
            "3/89 unique tasks attempted; 86 unattempted",
        )
    else:
        old = old.replace("已尝试2/89题，87题未执行", "已尝试3/89题，86题未执行")
    return old + " " + cython_note(english)


def render_report(old, value, prior):
    assert "报告版本：V0.96" in old
    text = old.replace(
        "报告版本：V0.96（SWE Tutanota增量实测）",
        "报告版本：V0.97（SWE第二批增量与Terminal Cython实测）",
        1,
    )
    start = text.index("| 任务序号 | 官方原记录 |")
    end = text.index("\n\n", start)
    text = (
        text[:start]
        + "\n".join(
            [
                "| 任务序号 | 官方原记录 | 模型调用 | Agent秒 | 状态 | 证据范围 |",
                "|---|---:|---:|---:|---|---|",
                *["| " + " | ".join(row) + " |" for row in previous.rows(value)],
            ]
        )
        + text[end:]
    )
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
                + " 原记录3/21；同答案恢复后4/21，共21次尝试。 | 710题及B1同题配对未完成 |"
            )
        elif line.startswith("| TerminalBench 2.1 "):
            line = (
                "| TerminalBench 2.1 | "
                + terminal_scope()
                + " | 86题及B1同题配对未完成；Cython原Agent评分null |"
            )
        lines.append(line)
    text = "\n".join(lines) + "\n"
    # Keep the previous bounded diagnosis explicitly historical rather than replacing its result.
    text = text.replace(
        "Terminal仍为2/89项已尝试、87项未执行，本题正控制仍未通过。",
        "该诊断时点Terminal为2/89项已尝试、87项未执行，当时本题正控制未通过。",
    )
    text = text.replace(terminal_previous.scope(), terminal_scope())
    marker = "#### 0.18 Native与已有优化方案同题质量"
    addition = (
        "#### 第二批增量与Cython当前证据\n\n"
        + wave_note()
        + "\n\n"
        + cython_note()
        + "\n\n证据：[SWE第二批前三题](../swe-native018-wave2-first-three-20261027/README.md)、"
        "[Cython控制与原Agent结果](../terminal-cython-context-20261027/README.md)。\n\n"
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
    assert value["entries"][:18] == prior["entries"]
    assert (
        value["attempted_unique_tasks"],
        value["recorded_official_resolved"],
        value["unattempted_tasks"],
    ) == (21, 3, 710)
    model = json.loads(read(TERMINAL + "/model-summary.json"))
    controls = json.loads(read(TERMINAL + "/controls-summary.json"))
    report = read(REPORT)
    assert "报告版本：V0.97" in report.decode()
    data.update(
        version="V0.97",
        source_commit=revision,
        report_sha256=hashlib.sha256(report).hexdigest(),
        report_url=f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{revision}/{REPORT}",
        swe_incremental_tasks=value,
        swe_same_answer_recovery_aggregate=json.loads(
            read(FOLDER + "/current-aggregate.json")
        ),
        terminal_cython_original_model=model,
        terminal_cython_repaired_controls=controls,
        terminal_cython_source_boundary=json.loads(
            read(TERMINAL + "/source-boundary.json")
        ),
        terminal_cython_reconstructed_recovery=json.loads(
            read(TERMINAL + "/recovery-summary.json")
        ),
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
        section[key][0]["rows"] = previous.rows(value, english)
        for table in remaining[key]:
            for row in table["rows"]:
                if row[0] == "SWE-bench-Pro":
                    row[1] = (
                        common.summary(value, english) + " " + current_notes(english)
                    )
                    row[2] = (
                        "710 tasks and B1 pairing remain."
                        if english
                        else "710题及B1同题配对未完成。"
                    )
                elif row[0] == "TerminalBench 2.1":
                    row[1] = terminal_scope(english)
                    row[2] = (
                        "86 tasks and B1 pairing remain; original Cython reward is null."
                        if english
                        else "86题及B1同题配对未完成；Cython原评分null。"
                    )
    for s in data["sections"]:
        for key, english in (("scope", False), ("scope_en", True)):
            if terminal_previous.scope(english) in s.get(key, ""):
                s[key] = s[key].replace(
                    terminal_previous.scope(english), terminal_scope(english)
                )
        if s["id"] == "terminal-cython-controls":
            s["scope"] = cython_note()
            s["scope_en"] = cython_note(True)
            s["evidence"].insert(
                0,
                {
                    "path": TERMINAL + "/controls-summary.json",
                    "kind": "blob",
                    "label": "新环境原控制 / Original controls in repaired environment",
                },
            )
            for key, english in (("tables", False), ("tables_en", True)):
                s[key][0]["rows"] += [
                    [
                        "planarity0.6 nop",
                        "1 / 11",
                        "0",
                        "No model coverage" if english else "模型覆盖新增0",
                    ],
                    [
                        "planarity0.6 oracle",
                        "11 / 11",
                        "1",
                        "Original tests unchanged" if english else "原测试不变",
                    ],
                ]
    links = [
        {"path": SOURCE, "kind": "blob", "label": "21项累计结果 / 21-attempt catalog"},
        {
            "path": TERMINAL + "/model-summary.json",
            "kind": "blob",
            "label": "Cython原Agent结果 / Original Cython Agent",
        },
        {
            "path": TERMINAL + "/source-boundary.json",
            "kind": "blob",
            "label": "源码复用边界 / Source reuse boundary",
        },
        {
            "path": TERMINAL + "/recovery-summary.json",
            "kind": "blob",
            "label": "派生重建评分 / Reconstructed-artifact scoring",
        },
    ]
    section["evidence"].insert(0, copy.deepcopy(links[0]))
    remaining["evidence"] = copy.deepcopy(links) + remaining["evidence"]
    data["sections"].append(
        {
            "id": "terminal-cython-model",
            "title": "Terminal Cython模型实测 / Cython model attempt",
            "scope": cython_note(),
            "scope_en": cython_note(True),
            "tables": [],
            "tables_en": [],
            "evidence": copy.deepcopy(links[1:]),
            "source_heading": "第二批增量与Cython当前证据",
        }
    )
    return data


def prepare():
    old = ROOT / OLD_REPORT
    if not old.exists():
        raw = (ROOT / REPORT).read_bytes()
        assert "报告版本：V0.96" in raw.decode()
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
