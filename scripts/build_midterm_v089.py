"""Project independently graded repository tasks, preserving earlier evidence."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import build_midterm_v088 as previous

ROOT = Path(__file__).resolve().parents[1]
REPORT = previous.REPORT
OLD_REPORT = "reports/vllm-hust-midterm-20261019/中期测试报告-v088.md"
SOURCE = "reports/swe-native018-next-repository-20261026/catalog.json"
OUTPUT = ROOT / "website-publication/data/midterm-v089.json"


def coverage_note(entry, english=False):
    raw = entry.get("raw_test_coverage")
    if not raw:
        return ""
    passed, failed = raw["raw_top_level"]["PASSED"], raw["raw_top_level"]["FAILED"]
    records, setup = raw["official_parser_records"], raw["raw_package_setup_failures"]
    assert raw["missing_expected_F2P_raw_failed"] is True
    if english:
        return (
            f" Flipt raw top-level tests: {passed} passed, {failed} failed; {setup} package setup failures. "
            f"The frozen parser retained {records} PASS records only. The missing F2P target executed and failed; "
            "parser omissions are not unexecuted tests. Original scoring is unchanged."
        )
    return (
        f" Flipt原始顶层测试{passed}通过、{failed}失败，另有{setup}个包初始化失败；"
        f"冻结解析器仅保留{records}条PASS记录。缺失的F2P目标实际执行并失败，"
        "解析缺失不等于测试未执行；原评分保留。"
    )


def render_report(old, value, prior):
    assert "报告版本：V0.88" in old
    text = old.replace(
        "报告版本：V0.88（SWE-bench-Pro跨仓库实测及评分覆盖）",
        "报告版本：V0.89（SWE-bench-Pro跨仓库增量实测）",
        1,
    )
    start = text.index("| 任务序号 | 官方原记录 |")
    end = text.index("\n\n", start)
    table = [
        "| 任务序号 | 官方原记录 | 模型调用 | Agent秒 | 状态 | 证据范围 |",
        "|---|---:|---:|---:|---|---|",
        *["| " + " | ".join(row) + " |" for row in previous.rows(value)],
    ]
    text = text[:start] + "\n".join(table) + text[end:]
    old_summary = previous.summary(prior)
    assert old_summary in text
    text = text.replace(old_summary, previous.summary(value))
    marker = "证据：[本批规范结果与复算]"
    position = text.index(marker)
    paragraphs = []
    for entry in value["entries"][len(prior["entries"]) :]:
        grade = entry["official_test_coverage"]
        f, p = grade["fail_to_pass"], grade["pass_to_pass"]
        outcome = "true" if entry["official_resolved"] else "false"
        application = "成功" if entry["patch_applied"] else "失败"
        paragraphs.append(
            f"任务{value['entries'].index(entry) + 1}：{entry['model_calls']}次模型调用、"
            f"{entry['elapsed_seconds']:.6f}秒，补丁应用{application}，官方判{outcome}；"
            f"F2P为{f['passed']}/{f['expected']}、P2P为{p['passed']}/{p['expected']}，"
            f"预期测试名缺失分别为{f['missing']}、{p['missing']}。原预算、补丁及官方评分规则保留。"
            + coverage_note(entry)
        )
    text = text[:position] + "\n\n".join(paragraphs) + "\n\n" + text[position:]
    text = text.replace(
        "../swe-native018-repository-20261026/README.md",
        "../swe-native018-next-repository-20261026/README.md",
    )
    text = text.replace(
        "../swe-native018-repository-20261026/catalog.json",
        "../swe-native018-next-repository-20261026/catalog.json",
    )
    return text


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
    assert "报告版本：V0.89" in report.decode()
    data.update(
        version="V0.89",
        source_commit=revision,
        report_sha256=hashlib.sha256(report).hexdigest(),
        report_url=f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{revision}/{REPORT}",
        swe_incremental_tasks=value,
    )
    section = next(
        s for s in data["sections"] if s["id"] == "swe-single-agent-observation"
    )
    section.update(
        scope=previous.summary(value), scope_en=previous.summary(value, True)
    )
    for entry in value["entries"]:
        section["scope"] += coverage_note(entry)
        section["scope_en"] += coverage_note(entry, True)
    section["tables"][0]["rows"] = previous.rows(value)
    section["tables_en"][0]["rows"] = previous.rows(value, True)
    section["evidence"].insert(
        0,
        {
            "path": SOURCE,
            "kind": "blob",
            "label": "跨仓库增量结果 / Additional repository results",
        },
    )
    remaining = next(s for s in data["sections"] if s["id"] == "v15-remaining")
    remaining["evidence"].append(section["evidence"][0].copy())
    for key, english in (("tables", False), ("tables_en", True)):
        for table in remaining[key]:
            for row in table["rows"]:
                if row[0] == "SWE-bench-Pro":
                    prefix = (
                        "Three existing 3,600-second runs retained; "
                        if english
                        else "三组各3600秒真实Agent记录保留；"
                    )
                    row[1] = prefix + previous.summary(value, english)
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
