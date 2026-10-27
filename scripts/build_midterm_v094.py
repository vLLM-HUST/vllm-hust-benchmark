"""Project measured attempts, original-test recovery and failed environment controls."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import build_midterm_v088 as common
import build_midterm_v090 as swe
import build_midterm_v093 as previous

ROOT = Path(__file__).resolve().parents[1]
REPORT = previous.REPORT
OLD_REPORT = "reports/vllm-hust-midterm-20261019/中期测试报告-v093.md"
SOURCE = "reports/swe-native018-protonmail-qutebrowser-20261027/catalog.json"
PRIOR = "reports/swe-native018-navidrome-20261027/catalog.json"
CYTHON = "reports/terminal-build-cython-controls-20261027/summary.json"
RECOVERY = "reports/swe-protonmail-test-source-recovery-20261027/summary.json"
OUTPUT = ROOT / "website-publication/data/midterm-v094.json"


def notes(english=False):
    if english:
        return (
            "Protonmail: one submitted Agent attempt, 37 successful model calls, 194.631876 seconds. "
            "The original patch applied, but the fixed image lacked the requested test commit/file; "
            "the original official result is false with zero parsed tests, not a functional failure. "
            "Separate verifier-only recovery restored the unchanged 1033-byte test from the frozen dataset test_patch. "
            "Jest then failed loading canvas.node before any test body; the recovery still has zero tests and official false. "
            "The original patch, test assertions, run_script, parser and scoring sets were not changed; no new Agent or model calls. "
            "qutebrowser: 75 successful model responses and one context-limit rejection (76 HTTP attempts), "
            "744.708365 seconds, no submission. At least 61441 input tokens plus 4096 requested output exceeded 65536. "
            "The unchanged 740-byte partial patch applied and was graded separately: F2P 2/3, P2P 46/46, no missing targets, official false. "
            "Raw pytest: 96 passed, 2 failed, 5 skipped; the original parser retained 67 passed, 2 failed, 5 skipped, "
            "omitting 29 passed names containing whitespace. All required names were retained. "
            "The original pytest performed six internal reruns; no model-request automatic retry or second Agent. "
            "Neither the partial patch score nor process exit zero establishes a completed submission."
        )
    return (
        "Protonmail仅一次Agent，37次成功模型调用、194.631876秒，主动提交。原补丁应用成功，"
        "但固定镜像缺少指定测试提交及文件，原官方记录false、解析测试0项，不是功能断言失败。"
        "另行仅恢复评分，从冻结数据集test_patch恢复原1033字节测试文件，未改测试断言；"
        "Jest随后因缺少canvas.node在测试体执行前失败，恢复记录仍为测试0项、官方false。"
        "原补丁、run_script、解析器及评分集合不变，新增Agent和模型调用均为0。"
        "qutebrowser有75次成功模型响应及1次上下文拒绝，共76次HTTP尝试，耗时744.708365秒，未主动提交。"
        "被拒请求至少61441输入token加4096输出预算超过65536。原740字节部分补丁应用成功并单独评分："
        "F2P为2/3、P2P为46/46，预期测试名缺失0，官方false。"
        "原pytest记录96通过、2失败、5跳过；原解析器保留67通过、2失败、5跳过，漏收29个含空白的通过名称，"
        "全部必要评分名称仍在。原pytest内部重试6次，不是模型请求自动重试或第二次Agent。"
        "部分补丁评分及进程退出0均不代表已完成提交。"
    )


def cython_note(english=False):
    if english:
        return (
            "build-cython-ext ARM64 build succeeded. Fresh nop control: 1/11 passed, reward 0; "
            "fresh original oracle: 10/11 passed, reward 0. The failed outer test ran a nested suite with 17 passed and one failed; "
            "its full nested traceback was not retained. No root cause or architecture equivalence is inferred. "
            "Original 11 tests, oracle, thresholds and budgets unchanged: build 600 seconds, Agent/verifier 900 seconds, "
            "1 CPU, 2 GiB, MTU1400. Both containers released; zero model calls or added model-task coverage. "
            "Positive control failed; no model execution admitted for this task. Terminal model coverage remains 2/89, with 87 unattempted."
        )
    return (
        "build-cython-ext的ARM64构建成功；独立nop控制1/11通过、reward=0，原oracle控制10/11通过、reward=0。"
        "失败外层测试调用的内部测试为17通过、1失败，完整内部trace未保留，不能据此确定根因或宣称架构等价。"
        "原11项测试、oracle、数值阈值及预算不变：构建600秒、Agent及Verifier各900秒，1CPU、2GiB、MTU1400。"
        "两容器均已释放，新增模型调用和模型任务覆盖均为0。正控制未通过，本题尚未准入模型测试；"
        "Terminal模型覆盖仍为2/89，87题未执行。"
    )


def rows(value, english=False):
    result = swe.rows(value, english)
    for entry, row in zip(value["entries"], result, strict=True):
        if entry["run_id"] == "swe018-repo-08-r1":
            row[-1] = (
                "Tests unavailable; separate source recovery"
                if english
                else "原测试缺失；恢复评分另列"
            )
        elif entry["run_id"] == "swe018-repo-09-r1":
            row[2] = "75 success + 1 rejected" if english else "75成功 + 1拒绝"
            row[-1] = (
                "Unsubmitted partial patch graded"
                if english
                else "未提交部分补丁的独立评分"
            )
    return result


def render_report(old, value, prior):
    assert "报告版本：V0.93" in old
    text = old.replace(
        "报告版本：V0.93（Terminal部分产物独立恢复评分）",
        "报告版本：V0.94（SWE增量及评分依赖、Terminal控制）",
        1,
    )
    start = text.index("| 任务序号 | 官方原记录 |")
    end = text.index("\n\n", start)
    table = [
        "| 任务序号 | 官方原记录 | 模型调用 | Agent秒 | 状态 | 证据范围 |",
        "|---|---:|---:|---:|---|---|",
        *["| " + " | ".join(row) + " |" for row in rows(value)],
    ]
    text = text[:start] + "\n".join(table) + text[end:]
    assert common.summary(prior) in text
    text = text.replace(common.summary(prior), common.summary(value))
    marker = "#### 0.18 Native与已有优化方案同题质量"
    assert marker in text
    addition = (
        "#### SWE评分依赖与未提交部分补丁\n\n"
        + notes()
        + "\n\n证据：[增量规范结果](../swe-native018-protonmail-qutebrowser-20261027/README.md)、"
        "[原测试输入恢复](../swe-protonmail-test-source-recovery-20261027/README.md)。\n\n"
        "#### Terminal Cython构建与评分控制\n\n"
        + cython_note()
        + "\n\n证据：[规范结果及离线复核](../terminal-build-cython-controls-20261027/README.md)。\n\n"
    )
    text = text.replace(marker, addition + marker, 1)
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("|") and line.split("|")[1].strip() == "SWE-bench-Pro":
            lines[i] = (
                "| SWE-bench-Pro | "
                + common.summary(value)
                + " 原测试依赖及部分补丁边界见本轮增量结果。 | "
                + str(value["unattempted_tasks"])
                + "题及B1同题配对未完成 |"
            )
        elif line.startswith("|") and line.split("|")[1].strip() == "TerminalBench 2.1":
            lines[i] = (
                "| TerminalBench 2.1 | "
                + previous.scope()
                + " "
                + cython_note()
                + " | 87题及B1同题配对未完成；Cython正控制未通过 |"
            )
    return "\n".join(lines) + "\n"


def build(revision, read=None):
    if read is None:

        def read(path):
            return subprocess.check_output(
                ["git", "show", f"{revision}:{path}"], cwd=ROOT
            )

    data = previous.build(revision, lambda p: read(OLD_REPORT if p == REPORT else p))
    report = read(REPORT)
    value = json.loads(read(SOURCE))
    prior = json.loads(read(PRIOR))
    cython = json.loads(read(CYTHON))
    recovery = json.loads(read(RECOVERY))
    assert "报告版本：V0.94" in report.decode()
    assert (
        value["attempted_unique_tasks"] == 17
        and value["recorded_official_resolved"] == 2
    )
    assert recovery["new_Agent_attempts"] == recovery["new_model_calls"] == 0
    assert recovery["tests_executed"] == 0 and recovery["test_source_restored"]
    assert cython["model_calls"] == cython["new_model_task_attempts"] == 0
    data.update(
        version="V0.94",
        source_commit=revision,
        report_sha256=hashlib.sha256(report).hexdigest(),
        report_url=f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{revision}/{REPORT}",
        swe_incremental_tasks=value,
        protonmail_test_source_recovery=recovery,
        terminal_cython_controls=cython,
    )
    section = next(
        s for s in data["sections"] if s["id"] == "swe-single-agent-observation"
    )
    for key, english in (("scope", False), ("scope_en", True)):
        assert common.summary(prior, english) in section[key]
        section[key] = (
            section[key].replace(
                common.summary(prior, english), common.summary(value, english)
            )
            + " "
            + notes(english)
        )
    section["tables"][0]["rows"] = rows(value)
    section["tables_en"][0]["rows"] = rows(value, True)
    evidence = [
        {"path": SOURCE, "kind": "blob", "label": "SWE增量 / SWE increment"},
        {
            "path": RECOVERY,
            "kind": "blob",
            "label": "原测试恢复 / Original test recovery",
        },
        {"path": CYTHON, "kind": "blob", "label": "Cython控制 / Cython controls"},
    ]
    section["evidence"].insert(0, evidence[0])
    remaining = next(s for s in data["sections"] if s["id"] == "v15-remaining")
    remaining["evidence"].extend(evidence)
    for key, english in (("tables", False), ("tables_en", True)):
        for table in remaining[key]:
            for row in table["rows"]:
                if row[0] == "SWE-bench-Pro":
                    row[1] = common.summary(value, english) + " " + notes(english)
                    if len(row) > 2:
                        row[2] = (
                            "714 tasks and B1 pairing remain."
                            if english
                            else "714题及B1同题配对未完成。"
                        )
                elif row[0] == "TerminalBench 2.1":
                    row[1] = previous.scope(english) + " " + cython_note(english)
    for ident, title, heading, note, ev, zh, en in [
        (
            "swe-test-source-recovery",
            "SWE评分依赖 / SWE grading dependencies",
            "SWE评分依赖与未提交部分补丁",
            notes,
            evidence[1],
            [
                ["原Protonmail评分", "0", "false", "测试提交/文件缺失"],
                ["仅恢复原测试输入", "0", "false", "canvas.node缺失；未执行测试体"],
            ],
            [
                ["Original Protonmail grade", "0", "false", "Missing test commit/file"],
                [
                    "Original test source restored",
                    "0",
                    "false",
                    "canvas.node missing; no test bodies",
                ],
            ],
        ),
        (
            "terminal-cython-controls",
            "Terminal Cython控制 / Cython controls",
            "Terminal Cython构建与评分控制",
            cython_note,
            evidence[2],
            [
                ["nop", "1 / 11", "0", "负控制符合预期"],
                ["oracle", "10 / 11", "0", "正控制失败；模型新增0"],
            ],
            [
                ["nop", "1 / 11", "0", "Expected negative control"],
                [
                    "oracle",
                    "10 / 11",
                    "0",
                    "Positive control failed; zero new model tasks",
                ],
            ],
        ),
    ]:
        s = {
            "id": ident,
            "title": title,
            "source_heading": heading,
            "scope": note(),
            "scope_en": note(True),
            "evidence": [ev],
        }
        s["tables"] = [
            {"headers": ["阶段", "测试通过/执行", "原记录", "边界"], "rows": zh}
        ]
        s["tables_en"] = [
            {
                "headers": [
                    "Phase",
                    "Tests passed/executed",
                    "Recorded result",
                    "Scope",
                ],
                "rows": en,
            }
        ]
        data["sections"].append(s)
    return data


def prepare():
    old = ROOT / OLD_REPORT
    if not old.exists():
        raw = (ROOT / REPORT).read_bytes()
        assert "报告版本：V0.93" in raw.decode()
        old.write_bytes(raw)
    value = json.loads((ROOT / SOURCE).read_text())
    prior = json.loads((ROOT / PRIOR).read_text())
    (ROOT / REPORT).write_text(render_report(old.read_text(), value, prior))
    folder = (ROOT / SOURCE).parent
    lines = [
        "# SWE-bench-Pro跨仓库增量实测",
        "",
        "## 配置与方法",
        "",
        "Qwen3.5-35B-A3B，官方vLLM 0.18.0 / Ascend 0.18.0、TP2。每题一次Agent，100步、1200秒、max_tokens=4096、temperature=0、seed=0、thinking=false，模型请求零自动重试；原评分器和输入不变。",
        "",
        "## 结果",
        "",
        common.summary(value),
        "",
        notes(),
        "",
        "| 任务序号 | 官方原记录 | 模型调用 | Agent秒 | 状态 | 证据范围 |",
        "|---|---:|---:|---:|---|---|",
        *["| " + " | ".join(row) + " |" for row in rows(value)],
        "",
        "## 证据",
        "",
        "catalog.json为保留既有15题的完整17题目录。summary/review/release/custody仅含聚合与SHA，不公开题目、答案、补丁或请求标识。",
        "运行python3 verify.py离线复核公开聚合及哈希；derive_private_*.py.source复核原件需受控CPU证据。原失败、部分补丁和评分依赖限制均保留。",
        "",
    ]
    (folder / "README.md").write_text("\n".join(lines))
    prior_files = json.loads(((ROOT / PRIOR).parent / "public-files.json").read_text())
    paths = {v["path"] for v in prior_files}
    paths.add(str(((ROOT / PRIOR).parent / "public-files.json").relative_to(ROOT)))
    names = [
        "README.md",
        "catalog.json",
        "inputs.json",
        "derive_catalog.py",
        "verify.py",
    ]
    for item in json.loads((folder / "inputs.json").read_text())["observations"]:
        names.extend(
            item[key]
            for key in ("summary", "review", "release", "custody", "derivation_source")
        )
    paths.update(str((folder / name).relative_to(ROOT)) for name in names)
    payload = []
    for name in sorted(paths):
        raw = (ROOT / name).read_bytes()
        payload.append(
            {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        )
    (folder / "public-files.json").write_text(json.dumps(payload, indent=2) + "\n")
    index = ROOT / "reports/four-primary-datasets-20261025/README.md"
    lines = index.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("|") and line.split("|")[1].strip() == "SWE-bench-Pro":
            lines[i] = (
                "| SWE-bench-Pro | "
                + common.summary(value)
                + " | 714题及B1同题配对；Protonmail测试依赖待补齐 |"
            )
        elif line.startswith("|") and line.split("|")[1].strip() == "TerminalBench 2.1":
            lines[i] = (
                "| TerminalBench 2.1 | "
                + previous.scope()
                + " "
                + cython_note()
                + " | 87题及B1同题配对 |"
            )
    index.write_text("\n".join(lines) + "\n")
    index = ROOT / "reports/vllm-hust-midterm-20261019/README.md"
    link = "- SWE增量与评分依赖：[规范结果](../swe-native018-protonmail-qutebrowser-20261027/README.md)；[Terminal Cython控制](../terminal-build-cython-controls-20261027/README.md)。"
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
