"""Project new task outcomes and original ARM64 verifier controls."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import build_midterm_v089 as previous

ROOT = Path(__file__).resolve().parents[1]
REPORT = previous.REPORT
OLD_REPORT = "reports/vllm-hust-midterm-20261019/中期测试报告-v089.md"
SOURCE = "reports/swe-native018-continuation-20261026/catalog.json"
TERMINAL = "reports/terminal-openssl-arm64-controls-20261026/summary.json"
OUTPUT = ROOT / "website-publication/data/midterm-v090.json"


def coverage_note(entry, english=False):
    raw = entry.get("raw_test_coverage") or {}
    if raw.get("expected_F2P_and_P2P_union_equals_observed_tests"):
        if english:
            return (
                " Openlibrary: 18 raw tests passed, matching all 18 official parsed tests: F2P 3/3 and P2P 15/15, no missing targets. "
                "Four original warnings retained. The exact original submodule fixture was verified before the first model request; "
                "neither the submodule nor its symlink was included in the submitted patch. "
                "The patch applied successfully and the after/graded file identities match. No task or grading rerun."
            )
        return (
            " Openlibrary原始18项测试全部通过，与官方解析18项一致，F2P为3/3、P2P为15/15，无缺失目标，保留4条原warnings。"
            "首次模型请求前已核对原子模块fixture，提交补丁不含子模块或其符号链接；"
            "补丁应用成功，应用后与实际待评分文件身份一致，未重跑任务或评分。"
        )
    if not raw.get("target_test_build_failed"):
        return previous.coverage_note(entry, english)
    if english:
        return (
            " Teleport: the applied submission omitted Bytea/Timestamptz/UUID methods required by the original tests; "
            "the target package failed compilation before either target test body ran. "
            "All 1381 JSON records were package events, with no Test field; 188 package passes and 95 package failures are not test counts. "
            "The unchanged official parser returned zero test records and the original score remains false. "
            "Other missing-header/sqlite compilation errors and a grading-container OOM are retained separately. "
            "Grader exit 0 is not Go-test success; the independent numeric Go-test exit code was not captured."
        )
    return (
        " Teleport已应用提交缺少原测试调用的Bytea/Timestamptz/UUID方法，目标包编译失败，两个目标测试体均未执行。"
        "1381条JSON全部为包级事件且无Test字段，188包通过、95包失败不能作为测试通过数或失败数。"
        "冻结解析器返回0条测试记录，原官方false保留。其他缺失头文件、sqlite编译错误及评分容器OOM独立保留。"
        "评分进程退出0不等于Go测试成功；未独立采集Go测试的数值退出码，不填造。"
    )


def rows(value, english=False):
    result = previous.previous.rows(value, english)
    for row, entry in zip(result, value["entries"], strict=True):
        if (entry.get("raw_test_coverage") or {}).get("target_test_build_failed"):
            row[-1] = (
                "Target compilation failed; test bodies not run"
                if english
                else "目标编译失败，测试体未执行"
            )
    return result


def terminal_scope(english=False):
    if english:
        return (
            "Original openssl-selfsigned-cert task on a separately built ARM64 image: original nop 0/6, reward 0; "
            "original oracle 6/6, reward 1. One fresh container per control; no retries, zero model calls, both released. "
            "Harbor 0.24.0, one CPU, 2 GiB, MTU1400. Only image references changed; original instructions, tests and oracle are unchanged. "
            "One environment/verifier control pair, not model acceptance, 89-task coverage or equivalence to the original AMD64 image."
        )
    return (
        "原openssl-selfsigned-cert任务在独立构建的ARM64镜像上，原nop为0/6、奖励0，原oracle为6/6、奖励1。"
        "每个控制使用不同全新容器，各一次、无重试、零模型调用，均已释放。"
        "Harbor 0.24.0、1CPU、2GiB、MTU1400；仅镜像引用变化，原指令、测试和oracle保持不变。"
        "仅完成一题环境及评分控制，不是模型验收、89题覆盖或原AMD64镜像等价证明。"
    )


def render_report(old, value, prior):
    assert "报告版本：V0.89" in old
    text = old.replace(
        "报告版本：V0.89（SWE-bench-Pro跨仓库增量实测）",
        "报告版本：V0.90（SWE增量实测与Terminal ARM64评分控制）",
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
    old_summary = previous.previous.summary(prior)
    assert old_summary in text
    text = text.replace(old_summary, previous.previous.summary(value))
    position = text.index("证据：[本批规范结果与复算]")
    notes = []
    for entry in value["entries"][len(prior["entries"]) :]:
        g = entry["official_test_coverage"]
        f, p = g["fail_to_pass"], g["pass_to_pass"]
        notes.append(
            f"任务{value['entries'].index(entry) + 1}：{entry['model_calls']}次模型调用、{entry['elapsed_seconds']:.6f}秒，"
            f"官方原记录{entry['official_resolved']}/1；F2P {f['passed']}/{f['expected']}，P2P {p['passed']}/{p['expected']}，"
            f"预期测试名缺失分别为{f['missing']}、{p['missing']}。"
            + coverage_note(entry)
        )
    text = text[:position] + "\n\n".join(notes) + "\n\n" + text[position:]
    text = text.replace(
        "../swe-native018-next-repository-20261026/",
        "../swe-native018-continuation-20261026/",
    )
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("|") and line.split("|")[1].strip() == "TerminalBench 2.1":
            lines[i] = (
                "| TerminalBench 2.1 | "
                + terminal_scope()
                + " | 模型任务覆盖仍为0/89；其他任务的构建及评分控制待执行 |"
            )
    text = "\n".join(lines) + "\n"
    marker = "#### 0.18 Native与已有优化方案同题质量"
    section = (
        "#### TerminalBench2.1原任务ARM64评分控制\n\n"
        + terminal_scope()
        + "\n\n证据：[规范结果及复核](../terminal-openssl-arm64-controls-20261026/README.md)。\n\n"
    )
    assert marker in text
    return text.replace(marker, section + marker, 1)


def build(revision, read=None):
    if read is None:

        def read(path):
            return subprocess.check_output(
                ["git", "show", f"{revision}:{path}"], cwd=ROOT
            )

    data = previous.build(
        revision, lambda path: read(OLD_REPORT if path == REPORT else path)
    )
    report, value = read(REPORT), json.loads(read(SOURCE))
    terminal = json.loads(read(TERMINAL))
    assert "报告版本：V0.90" in report.decode()
    assert terminal["model_calls"] == terminal["official_89_task_model_coverage"] == 0
    data.update(
        version="V0.90",
        source_commit=revision,
        report_sha256=hashlib.sha256(report).hexdigest(),
        report_url=f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{revision}/{REPORT}",
        swe_incremental_tasks=value,
        terminal_arm64_controls=terminal,
    )
    section = next(
        s for s in data["sections"] if s["id"] == "swe-single-agent-observation"
    )
    section.update(
        scope=previous.previous.summary(value),
        scope_en=previous.previous.summary(value, True),
    )
    for entry in value["entries"]:
        section["scope"] += coverage_note(entry)
        section["scope_en"] += coverage_note(entry, True)
    section["tables"][0]["rows"] = rows(value)
    section["tables_en"][0]["rows"] = rows(value, True)
    section["evidence"].insert(
        0, {"path": SOURCE, "kind": "blob", "label": "后续任务结果 / Task continuation"}
    )
    data["sections"].append(
        {
            "id": "terminal-arm64-controls",
            "title": "Terminal ARM64评分控制 / Verifier controls",
            "scope": terminal_scope(),
            "scope_en": terminal_scope(True),
            "source_heading": "TerminalBench2.1原任务ARM64评分控制",
            "tables": [
                {
                    "headers": [
                        "控制",
                        "测试通过 / 分母",
                        "奖励",
                        "模型调用",
                        "资源释放",
                    ],
                    "rows": [
                        [
                            c["role"],
                            f"{c['passed']} / {c['tests']}",
                            str(c["reward"]),
                            "0",
                            "已释放",
                        ]
                        for c in terminal["controls"]
                    ],
                }
            ],
            "tables_en": [
                {
                    "headers": [
                        "Control",
                        "Passed / tests",
                        "Reward",
                        "Model calls",
                        "Resources",
                    ],
                    "rows": [
                        [
                            c["role"],
                            f"{c['passed']} / {c['tests']}",
                            str(c["reward"]),
                            "0",
                            "Released",
                        ]
                        for c in terminal["controls"]
                    ],
                }
            ],
            "evidence": [
                {
                    "path": TERMINAL,
                    "kind": "blob",
                    "label": "原任务控制结果 / Original-task controls",
                }
            ],
        }
    )
    remaining = next(s for s in data["sections"] if s["id"] == "v15-remaining")
    remaining["evidence"].extend(
        [section["evidence"][0].copy(), data["sections"][-1]["evidence"][0].copy()]
    )
    for key, english in (("tables", False), ("tables_en", True)):
        for table in remaining[key]:
            for row in table["rows"]:
                if row[0] == "SWE-bench-Pro":
                    row[1] = (
                        "Three existing 3,600-second runs retained; "
                        if english
                        else "三组各3600秒真实Agent记录保留；"
                    ) + previous.previous.summary(value, english)
                    row[1] += " ".join(
                        coverage_note(e, english)
                        for e in value["entries"]
                        if (e.get("raw_test_coverage") or {}).get(
                            "target_test_build_failed"
                        )
                    )
                elif row[0] == "TerminalBench 2.1":
                    row[1] = terminal_scope(english)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision")
    parser.add_argument("--prepare-report", action="store_true")
    args = parser.parse_args()
    if args.prepare_report:
        (ROOT / REPORT).write_text(
            render_report(
                (ROOT / OLD_REPORT).read_text(),
                json.loads((ROOT / SOURCE).read_text()),
                json.loads((ROOT / previous.SOURCE).read_text()),
            )
        )
    else:
        assert args.revision
        OUTPUT.write_text(
            json.dumps(build(args.revision), ensure_ascii=False, indent=2) + "\n"
        )
