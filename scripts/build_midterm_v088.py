"""Project exact SWE attempts and patch-application evidence without changing scoring."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import build_midterm_v087 as previous

ROOT = Path(__file__).resolve().parents[1]
REPORT = previous.REPORT
OLD_REPORT = "reports/vllm-hust-midterm-20261019/中期测试报告-v087.md"
SOURCE = "reports/swe-native018-repository-20261026/catalog.json"
OUTPUT = ROOT / "website-publication/data/midterm-v088.json"


def rows(value, english=False):
    result = []
    for i, entry in enumerate(value["entries"]):
        if entry["patch_application_evidence_limited"]:
            scope = "Application unproven" if english else "补丁应用未获证明"
        elif entry["confirmed_patch_application_failure"]:
            scope = "Patch application failed" if english else "补丁应用失败"
        elif entry.get("patch_applied") is True:
            scope = (
                "Applied patch, official grade" if english else "已应用补丁，官方评分"
            )
        else:
            scope = "Official grade" if english else "官方评分"
        result.append(
            [
                str(i + 1),
                f"{entry['official_resolved']} / 1",
                str(entry["model_calls"]),
                f"{entry['elapsed_seconds']:.6f}",
                entry["status"],
                scope,
            ]
        )
    return result


def summary(value, english=False):
    count = value["attempted_unique_tasks"]
    resolved = value["recorded_official_resolved"]
    limited = value["application_evidence_limited_tasks"]
    failed = value["confirmed_patch_application_failure_tasks"]
    remaining = value["unattempted_tasks"]
    if english:
        return (
            f"Designated 0.18 baseline: {count}/731 unique tasks attempted; recorded official scores "
            f"{resolved}/{count}, zero indeterminate grades. Patch application is unproven for "
            f"{limited} task(s) and confirmed failed for {failed} task(s); neither establishes "
            f"testing of the applied candidate patch. {remaining} tasks remain unattempted. "
            "Zero indeterminate grades does not mean every expected test name was present. "
            "Not full-dataset acceptance or a B0/B1 pair."
        )
    return (
        f"指定0.18基线已执行{count}/731个不同任务，官方原记录{resolved}/{count}、未决0；"
        f"{limited}题补丁应用未获证明，{failed}题确定补丁应用失败，均不能视为已应用候选补丁的功能验证。"
        f"尚未执行{remaining}题，不是全量验收或B0/B1配对。"
        "评分未决0不代表所有预期测试名齐全。"
    )


def render_report(old, value):
    assert "报告版本：V0.87" in old
    old = old.replace(
        "报告版本：V0.87（SWE-bench-Pro批次实测及补丁应用取证）",
        "报告版本：V0.88（SWE-bench-Pro跨仓库实测及评分覆盖）",
        1,
    )
    start = old.index("#### SWE-bench-Pro指定0.18增量实测")
    end = old.index("#### 0.18 Native与已有优化方案同题质量", start)
    block = [
        "#### SWE-bench-Pro指定0.18增量实测",
        "",
        (
            "Qwen3.5-35B-A3B，vLLM 0.18.0 / vLLM-Ascend 0.18.0。各题一次Agent尝试，100步、1200秒、"
            "max_tokens=4096、temperature=0、seed=0、thinking=false、零自动重试；任务与独立官方评分使用不同新容器，MTU1400。"
            "已完成任务及同运行身份的五项资格直接复用。第五至第九题在同一独占模型窗口内串行执行，模型正常驻留、APC关闭；"
            "各题使用独立任务输出及客户端缓存，不作为逐题冷启动或吞吐配对。"
        ),
        "",
        "| 任务序号 | 官方原记录 | 模型调用 | Agent秒 | 状态 | 证据范围 |",
        "|---|---:|---:|---:|---|---|",
        *["| " + " | ".join(row) + " |" for row in rows(value)],
        "",
        summary(value),
        "",
        (
            "第二题保留原负分及补丁应用证据不足的限制：未采集git apply退出码，不能追补为已验证应用失败。"
            "第四题已记录git apply退出1，两个镜像原有Redis文本文件被补丁作为新增文件提交，与评分镜像已有路径冲突；"
            "全部受影响文件在应用前、应用后及测试前（实际待评分）保持相同。官方入口继续运行测试，F2P为0/3、P2P为288/288。"
            "其官方负分属于确定的补丁应用失败，不证明候选代码功能未通过。原补丁、二进制过滤规则及评分原件均保留，未修补或重新判分。"
        ),
        "",
        (
            "18c同样为补丁应用失败，F2P为0/7，其中4项解析为失败、3项预期测试名未出现；P2P为418/418。"
            "原官方仍明确判false、评分未决0，缺失测试名不改写为评分未决，也不忽略。"
            "Ansible题补丁应用成功，F2P为1/1、P2P为15/15，官方判true。"
            "评分阶段不调用模型。原始测试条目、官方按测试名集合计算的F2P/P2P和最终任务分数分别保存；重名条目不作为新增任务。"
            "主动提交、客户端退出0及模型服务正常释放均不等于任务判对。"
        ),
        "",
        "element-web题已应用补丁，28次调用、225.544925秒，官方判false。原始Jest记录437通过、1失败；冻结解析器产生437条通过记录、按名称去重418项，漏收目标失败项，F2P为0/1（解析缺失1）、P2P为385/385。原始日志确认该F2P实际执行并失败，不将解析缺失称为测试未执行。首个评分入口在测试前因原tag未登记而退出1；补齐同一冻结镜像ID的tag后，在新目录完成首个实际官方评分，原失败保留，未重跑Agent或修改补丁及评分规则。",
        "",
        (
            "证据：[本批规范结果与复算](../swe-native018-repository-20261026/README.md)、"
            "[完整任务目录](../swe-native018-repository-20261026/catalog.json)、"
            "[前三题历史归档](../swe-native018-incremental-20261026/README.md)。私有题目、回答、补丁及轨迹不公开。"
        ),
        "",
    ]
    old = old[:start] + "\n".join(block) + "\n" + old[end:]
    lines = old.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        cells = line.split("|")
        if cells[1].strip() == "SWE-bench-Pro":
            cells[2] = " 三组各3600秒真实Agent记录保留；" + summary(value) + " "
            lines[i] = "|".join(cells)
        elif cells[1].strip() == "TerminalBench 2.1":
            cells[2] = (
                " 原89题1035份任务文件核验通过；按国产ARM64准备派生镜像，尚无构建或任务通过率结果 "
            )
            lines[i] = "|".join(cells)
        elif cells[1].strip() == "数据集覆盖":
            cells[2] = (
                " MMLU-Pro、FrontierScience及HLE完整结果保留；" + summary(value) + " "
            )
            lines[i] = "|".join(cells)
    return "\n".join(lines) + "\n"


def build(revision, read=None):
    if read is None:

        def read(path):
            return subprocess.check_output(
                ["git", "show", f"{revision}:{path}"], cwd=ROOT
            )

    report = read(REPORT)
    assert "报告版本：V0.88" in report.decode()
    value = json.loads(read(SOURCE))
    data = json.loads(read("website-publication/data/midterm-v087.json"))
    data.update(
        version="V0.88",
        source_commit=revision,
        report_sha256=hashlib.sha256(report).hexdigest(),
        report_url=f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{revision}/{REPORT}",
        swe_incremental_tasks=value,
    )
    section = next(
        s for s in data["sections"] if s["id"] == "swe-single-agent-observation"
    )
    section.update(scope=summary(value), scope_en=summary(value, True))
    section["tables"][0]["rows"] = rows(value)
    section["tables_en"][0]["rows"] = rows(value, True)
    section["evidence"] = [
        {
            "path": SOURCE,
            "kind": "blob",
            "label": "任务结果及应用取证 / Task results and patch-application evidence",
        }
    ]
    section["scope"] += (
        " 新增element-web实际失败；冻结解析器漏收该失败项，原始438项测试与解析437条记录分开保留。"
    )
    section["scope_en"] += (
        " The added element-web task failed. The frozen parser omitted its failed test; 438 raw tests and 437 parsed records are retained separately."
    )
    remaining = next(s for s in data["sections"] if s["id"] == "v15-remaining")
    remaining["evidence"].append(section["evidence"][0].copy())
    for key, english in (("tables", False), ("tables_en", True)):
        for table in remaining[key]:
            for row in table["rows"]:
                if row[0] == "SWE-bench-Pro":
                    row[1] = (
                        "Three existing 3,600-second runs retained; "
                        if english
                        else "三组各3600秒真实Agent记录保留；"
                    ) + summary(value, english)
                elif row[0] == "TerminalBench 2.1":
                    row[1] = (
                        "89 tasks and 1,035 source files verified; derived images are being prepared for domestic ARM64. No build qualification or task pass rate yet."
                        if english
                        else "原89题1035份任务文件核验通过；按国产ARM64准备派生镜像，尚无构建或任务通过率结果"
                    )
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision")
    parser.add_argument("--prepare-report", action="store_true")
    args = parser.parse_args()
    if args.prepare_report:
        (ROOT / REPORT).write_text(
            render_report(
                (ROOT / OLD_REPORT).read_text(), json.loads((ROOT / SOURCE).read_text())
            )
        )
    else:
        assert args.revision
        OUTPUT.write_text(
            json.dumps(build(args.revision), indent=2, ensure_ascii=False) + "\n"
        )
