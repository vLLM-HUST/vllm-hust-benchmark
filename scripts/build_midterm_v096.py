"""Add one original-protocol SWE attempt without replacing prior recovery evidence."""

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import build_midterm_v088 as common
import build_midterm_v094 as tables
import build_midterm_v095 as previous

ROOT = Path(__file__).resolve().parents[1]
REPORT = previous.REPORT
OLD_REPORT = "reports/vllm-hust-midterm-20261019/中期测试报告-v095.md"
FOLDER = "reports/swe-native018-tutanota-20261027"
SOURCE = FOLDER + "/catalog.json"
PRIOR = tables.SOURCE
OUTPUT = ROOT / "website-publication/data/midterm-v096.json"


def note(english=False):
    if english:
        return (
            "Tutanota: one original-protocol Agent attempt, 100 model calls, 437.454803 seconds; "
            "the step budget was exhausted without submission. The unchanged 7862-byte partial patch "
            "applied successfully to four paths; after-application and actual pre-test states agree. "
            "Official grading is false: TypeScript compilation failed before the target test ran. "
            "The compiler reports a missing OperationProgressTracker module, a type-only import used "
            "as a value, and a constructor argument-count mismatch. The original parser records one "
            "ERROR; the required F2P target is missing, P2P has no required targets. "
            "This is not a failed functional assertion or a completed submission. "
            "No additional Agent, modified patch, scoring retry or model call during grading; resources released."
        )
    return (
        "Tutanota沿用原协议仅执行一次Agent，100次模型调用、437.454803秒，步数耗尽且未主动提交。"
        "原7862字节部分补丁成功应用于4个路径，应用后与测试前实际待评分状态一致。"
        "官方评分false：TypeScript编译失败，目标测试尚未执行。编译器记录OperationProgressTracker模块缺失、"
        "仅类型导入被用作运行时值及构造参数数量不匹配；原解析器产生1条ERROR，必要F2P目标缺失1项，P2P集合为空。"
        "该结果不是功能断言失败，也不代表完成提交。未追加Agent、改写补丁或重复评分，评分阶段模型调用0；资源已释放。"
    )


def aggregate(value, recovery):
    snapshot = recovery["aggregate_distinction"]
    assert snapshot["attempted_tasks"] == 17
    assert snapshot["original_official_positives"] == 2
    assert snapshot["positives_after_disclosed_recovery"] == 3
    assert recovery["derived_dependency_recovery_result"]["official"] is True
    proton = [e for e in value["entries"] if e["run_id"] == "swe018-repo-08-r1"]
    assert len(proton) == 1 and proton[0]["official_resolved"] == 0
    count = value["attempted_unique_tasks"]
    return {
        "attempted_tasks": count,
        "original_official_positives": value["recorded_official_resolved"],
        "positives_after_disclosed_recovery": value["recorded_official_resolved"] + 1,
        "recovery_denominator": count,
        "unattempted_tasks": value["unattempted_tasks"],
        "additional_Agent_attempts_for_recovery": 0,
        "recovery_source": previous.RECOVERY,
        "prior_recovery_snapshot": copy.deepcopy(snapshot),
    }


def current_notes(english=False):
    text = previous.current_notes(english)
    if english:
        old = "Same 17/731 attempts: original records 2/17, after disclosed same-answer recovery 3/17; 714 unattempted. This is not an eighteenth attempt or a full-dataset/B0-B1 paired result."
        new = "With the separate new Tutanota attempt, current coverage is 18/731: original records 2/18, after disclosed same-answer recovery 3/18; 713 unattempted. Recovery adds no attempt. Not full-dataset acceptance or a B0/B1 pair."
    else:
        old = "仍为17/731项尝试：原记录2/17，披露同一答案恢复后3/17，714项未执行；不是第18次尝试或全量、B0/B1配对通过率。"
        new = "计入另行新增的Tutanota尝试后，当前覆盖18/731项：原记录2/18，披露同一答案恢复后3/18，713项未执行；恢复评分本身不新增尝试，不是全量或B0/B1配对通过率。"
    assert old in text
    return text.replace(old, new) + " " + note(english)


def rows(value, english=False):
    result = tables.rows(value, english)
    for entry, row in zip(value["entries"], result, strict=True):
        if entry["run_id"] == "swe018-repo-08-r1":
            row[-1] = (
                "Original false; same-answer recovery true"
                if english
                else "原false保留；同答案恢复true"
            )
        elif entry["run_id"] == "swe018-repo-10-r1":
            row[-1] = (
                "Unsubmitted partial patch; compilation failed"
                if english
                else "未提交部分补丁；编译失败，功能未执行"
            )
    return result


def render_report(old, value, prior):
    assert "报告版本：V0.95" in old
    text = old.replace(
        "报告版本：V0.95（SWE原答案依赖恢复与Terminal诊断）",
        "报告版本：V0.96（SWE Tutanota增量实测）",
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
    note_start = text.index(
        "Protonmail复用原已提交答案", text.index("#### SWE评分依赖与未提交部分补丁")
    )
    note_end = text.index("\n\n", note_start)
    assert " ".join(text[note_start:note_end].split()) == " ".join(
        previous.current_notes().split()
    )
    text = text[:note_start] + current_notes() + text[note_end:]
    text = text.replace(common.summary(prior), common.summary(value))
    text = text.replace(
        "原记录2/17；同一答案恢复后3/17，仍为17次尝试。",
        "原记录2/18；同一答案恢复后3/18，共18次尝试。",
    )
    text = text.replace("714题及B1同题配对未完成", "713题及B1同题配对未完成")
    marker = "#### 0.18 Native与已有优化方案同题质量"
    assert marker in text
    return text.replace(
        marker,
        f"证据：[Tutanota规范结果及复算](../swe-native018-tutanota-20261027/README.md)。\n\n{marker}",
        1,
    )


def build(revision, read=None):
    if read is None:

        def read(path):
            return subprocess.check_output(
                ["git", "show", f"{revision}:{path}"], cwd=ROOT
            )

    data = previous.build(revision, lambda p: read(OLD_REPORT if p == REPORT else p))
    value = json.loads(read(SOURCE))
    prior = json.loads(read(PRIOR))
    report = read(REPORT)
    assert "报告版本：V0.96" in report.decode()
    assert value["entries"][:-1] == prior["entries"]
    assert value["attempted_unique_tasks"] == 18 and value["unattempted_tasks"] == 713
    assert value["recorded_official_resolved"] == 2
    data.update(
        version="V0.96",
        source_commit=revision,
        report_sha256=hashlib.sha256(report).hexdigest(),
        report_url=f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{revision}/{REPORT}",
        swe_incremental_tasks=value,
        swe_same_answer_recovery_aggregate=aggregate(
            value, json.loads(read(previous.RECOVERY))
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
        section[key][0]["rows"] = rows(value, english)
        for table in remaining[key]:
            for row in table["rows"]:
                if row[0] == "SWE-bench-Pro":
                    row[1] = (
                        common.summary(value, english) + " " + current_notes(english)
                    )
                    row[2] = (
                        "713 tasks and B1 pairing remain."
                        if english
                        else "713题及B1同题配对未完成。"
                    )
    link = {
        "path": SOURCE,
        "kind": "blob",
        "label": "18项累计规范结果 / 18-attempt canonical catalog",
    }
    section["evidence"].insert(0, link)
    remaining["evidence"].insert(0, link)
    return data


def prepare():
    old = ROOT / OLD_REPORT
    if not old.exists():
        raw = (ROOT / REPORT).read_bytes()
        assert "报告版本：V0.95" in raw.decode()
        old.write_bytes(raw)
    value = json.loads((ROOT / SOURCE).read_text())
    prior = json.loads((ROOT / PRIOR).read_text())
    (ROOT / REPORT).write_text(render_report(old.read_text(), value, prior))


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
