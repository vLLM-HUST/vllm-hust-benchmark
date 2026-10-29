"""Withdraw the unsupported capacity denominator; reuse and recalculate own F/T.

No experiments, throughput-to-F conversions, or target-derived denominators.
V1.00 is reserved for the independent, unmerged SWE/Terminal update (#540).
"""

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "reports/vllm-hust-midterm-20261019"
FOLDER = BASE + "/m01-reference-revision-20261029"
REPORT = BASE + "/中期测试报告.md"
OLD_REPORT = BASE + "/中期测试报告-v099.md"
OLD_METHOD = BASE + "/current-method-v099.json"
SOURCE = BASE + "/m01-recovery/wholechain-reference-20261025/summary.json"
OUTPUT = "website-publication/data/midterm-v101.json"
GAPS = [
    "Independently justified reference capacity with a whole-chain execution witness",
    "Stronger existing candidate: own necessary-F/geometry, synchronized T endpoints and loaded-weight continuity",
]


def note(english=False):
    if english:
        return (
            "M01 reference under revision. Same-run reanalysis: B0 3.8407, B1 7.1042 TFLOPS/chip, "
            "a descriptive gain of 84.9705%. No qualified replacement C_ref is available; "
            "current eta and both midterm 50% / final 90% verdicts are unresolved. "
            "Independent repeats of the shared implementation do not establish workload-capacity strength. "
            "Raw measurements and historical methods are retained; no new experiment. "
            "Same-run accounting costs remain; timing is not strictly uninstrumented."
        )
    return (
        "M01参考修订中。同轮复算：B0 3.8407、B1 7.1042 TFLOPS/芯片，描述性提升84.9705%。"
        "尚无经核验的替代C参；当前η及中期50%/最终90%判定均未确定。"
        "共享实现的独立重复不足以建立工况能力强度。保留原始测量与历史方法，本次未启动新实验。"
        "本轮保留同轮计数成本，并非严格无插桩。"
    )


def derive(raw):
    current = {
        key: raw[key]
        for key in ("model", "dtype", "participating_chips", "device", "dataset")
    }
    current.update(
        schema="m01-reference-revision/v1",
        status="reference_under_revision",
        reference_qualified=False,
        reference_C_TFLOPS_per_chip=None,
        all_frozen_applicability_checks_pass=False,
        stage_thresholds_percent={"midterm": 50, "final": 90},
        source_path=SOURCE,
        method_path=FOLDER + "/method.md",
        pending_evidence=GAPS,
        new_measurements_started=False,
    )
    for arm in ("B0", "B1"):
        old = raw[arm]
        value = old["ownF"] / (raw["participating_chips"] * old["own_T_seconds"] * 1e12)
        assert value == old["own_C"]
        current[arm] = dict(
            ownF=old["ownF"],
            own_T_seconds=old["own_T_seconds"],
            own_C=value,
            T_reference_seconds=None,
            eta_percent=None,
            eta_target_met=None,
            midterm_target_met=None,
            final_target_met=None,
        )
    current["paired_C_gain_percent"] = (
        current["B1"]["own_C"] / current["B0"]["own_C"] - 1
    ) * 100
    return current


def revise_pair(pair, current):
    pair = copy.deepcopy(pair)
    pair.update(
        status="own_F_T_C_recomputed_reference_under_revision",
        reference_method_status=current["status"],
        wholechain_reference=current,
        scope=note(True),
    )
    for target in (pair, *pair["rows"].values()):
        for key in (
            "reference_seconds",
            "reference_tflops_per_chip",
            "eta_percent",
            "attainment",
        ):
            target[key] = None
    return pair


def revise_method(method, current):
    method = copy.deepcopy(method)
    method.update(
        method_version="reference-revision-20261029",
        status=current["status"],
        pending_evidence=GAPS,
        wholechain_reference=current,
        reference_scope=note(True),
        threshold_percent=90,
        stage_thresholds_percent=current["stage_thresholds_percent"],
    )
    for key in (
        "reference_seconds",
        "reference_effective_tflops_per_chip",
        "efficiency_percent",
        "passed",
    ):
        method["current"][key] = None
    method["reference_closure"] = {"status": "historical_only", "path": OLD_METHOD}
    method["paired_own_compute"] = revise_pair(method["paired_own_compute"], current)
    method["requirements"]["reference"] = {
        "status": current["status"],
        "path": FOLDER + "/method.md",
    }
    method["whole_chain_method_revision"] = {
        "status": current["status"],
        "method_path": FOLDER + "/method.md",
        "external_acceptance_approval_record": None,
        "baseline_normalization_is_reference_attainment": False,
    }
    return method


def render_report(old):
    text = old.replace(
        "报告版本：V0.99（SWE增量与Circuit原始上下文终止）",
        "报告版本：V1.01（有效算力参考纠正；其他结果沿用V0.99）",
        1,
    )
    paragraphs = text.split("\n\n")
    for i, paragraph in enumerate(paragraphs):
        if paragraph.startswith("原合订V1.5的50%门槛"):
            paragraphs[i] = note() + " 中期门槛50%、最终门槛90%分别保留。"
        elif paragraph.startswith("**总体结论："):
            start = paragraph.index("独立三次C200校准")
            paragraphs[i] = paragraph[:start] + note() + "**"
        elif paragraph.startswith("**官方0.18/0.18 B0与"):
            paragraphs[i] = (
                "**"
                + note()
                + "**\n\n方法与机器可读复算：[参考修订](m01-reference-revision-20261029/method.md)、[summary.json](m01-reference-revision-20261029/summary.json)。"
            )
        elif paragraph.startswith("72条实际输入不少于16,384 token的探针"):
            paragraphs[i] = "72条实际输入不少于16,384 token的探针均回答正确。" + note()
    text = "\n\n".join(paragraphs)
    lines = []
    for line in text.splitlines():
        if line.startswith("| M01 "):
            line = "| M01 | 单芯片有效算力 | 中期η≥50%；最终η≥90% | B0 C=3.8407、B1 C=7.1042；描述性提升84.9705% | 参考修订中；C参、η及达标判定未确定 |"
        elif line.startswith("| 3.4.1 有效算力"):
            line = "| 3.4.1 有效算力 | C=F/(k×T×10¹²)；独立参考待核验 | B0 3.8407、B1 7.1042 TFLOPS/芯片；η修订中 |"
        elif line.startswith("| 有效算力     |"):
            line = "| 有效算力 | 同轮F/T复算完成，C提升84.9705%；原始记录保留 | 工况参考能力强度与整链见证；替代C参未确定 |"
        lines.append(line)
    text = "\n".join(lines) + "\n"
    text = text.replace(
        "#### 独立整链参考与适用性核验",
        "#### 历史整链经验参考（已退出当前判定）\n\n以下为2026年10月25日原方法和算术记录，不是当前参考或达标结论。当前状态见本章开头及参考修订文件。",
        1,
    )
    return text


def build(revision, read):
    data = json.loads(read("website-publication/data/midterm-v099.json"))
    current = derive(json.loads(read(SOURCE)))
    method = revise_method(json.loads(read(OLD_METHOD)), current)
    report = read(REPORT)
    data.update(
        version="V1.01",
        source_commit=revision,
        report_sha256=hashlib.sha256(report).hexdigest(),
        report_url=f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{revision}/{REPORT}",
        wholechain_reference=current,
        m01_method=method,
    )
    data["current_summary"]["reference"] = current
    data["paired_own_compute"] = revise_pair(data["paired_own_compute"], current)
    quality_reference = data["corrected_quality"]["reference"]
    for key in ("C_ref_TFLOPS_per_chip", "T_ref_seconds", "eta_percent", "attainment"):
        quality_reference[key] = None
    quality_reference.update(
        status=current["status"], current_result_path=FOLDER + "/summary.json"
    )
    data["milestone_contract"]["method"] = method
    data["milestone_contract"]["method_source"] = {
        "path": BASE + "/current-method.json",
        "sha256": hashlib.sha256(read(BASE + "/current-method.json")).hexdigest(),
    }
    data["method_history"] = json.loads(read(BASE + "/method-history-index.json"))
    data["milestone_contract"]["decision_relevant_gaps"]["m01"] = GAPS
    links = [
        dict(path=FOLDER + "/" + name, kind="blob", label=label)
        for name, label in (
            ("summary.json", "同轮复算 / Same-run reanalysis"),
            ("method.md", "参考修订 / Reference revision"),
        )
    ]
    for section in data["sections"]:
        prior_scope = {key: section.get(key, "") for key in ("scope", "scope_en")}
        if section["id"] in {
            "corrected-quality",
            "milestone-v15",
            "v15-remaining",
            "m01-verdict",
        }:
            section.update(scope=note(), scope_en=note(True))
            section["evidence"] = links + section["evidence"]
        if section["id"] == "v15-remaining":
            for key, marker in (("scope", " 成本采用"), ("scope_en", " Cost uses")):
                assert marker in prior_scope[key]
                section[key] += marker + prior_scope[key].split(marker, 1)[1]
        if section["id"] == "milestone-v15":
            for key, english in (("tables", False), ("tables_en", True)):
                for table in section[key]:
                    for row in table["rows"]:
                        if row[0].startswith("3.4.1"):
                            row[1:] = [note(english)] + ["—"] * (len(row) - 2)
        if section["id"] == "m01-verdict":
            for key, english in (("tables", False), ("tables_en", True)):
                section[key] = [
                    {
                        "headers": [
                            "Arm" if english else "方案",
                            "F (FLOPs)",
                            "T (s)",
                            "C (TFLOPS/chip)",
                            "C_ref / η",
                            "Status" if english else "判定",
                        ],
                        "rows": [
                            [
                                arm,
                                str(current[arm]["ownF"]),
                                str(current[arm]["own_T_seconds"]),
                                f"{current[arm]['own_C']:.4f}",
                                "—",
                                "Under revision" if english else "修订中",
                            ]
                            for arm in ("B0", "B1")
                        ],
                    }
                ]
    return data


def write(path, value):
    (ROOT / path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--revision", default="HEAD")
    args = parser.parse_args()
    if args.prepare:
        for original, archived in (
            (REPORT, OLD_REPORT),
            (BASE + "/current-method.json", OLD_METHOD),
            (
                BASE + "/method-history-index.json",
                BASE + "/method-history-index-v099.json",
            ),
        ):
            if not (ROOT / archived).exists():
                (ROOT / archived).write_bytes((ROOT / original).read_bytes())
        (ROOT / REPORT).write_text(render_report((ROOT / OLD_REPORT).read_text()))
        current = derive(json.loads((ROOT / SOURCE).read_text()))
        write(FOLDER + "/summary.json", current)
        write(
            BASE + "/current-method.json",
            revise_method(json.loads((ROOT / OLD_METHOD).read_text()), current),
        )
        history = json.loads(
            (ROOT / (BASE + "/method-history-index-v099.json")).read_text()
        )
        history["snapshots"][-1]["role"] = (
            "historical empirical wholechain reference; withdrawn from current acceptance on 2026-10-29"
        )
        history["snapshots"][-1]["current"] = False
        history["current_revision"] = "m01-reference-revision-20261029/summary.json"
        write(BASE + "/method-history-index.json", history)
        return
    revision = subprocess.check_output(
        ["git", "rev-parse", args.revision], cwd=ROOT, text=True
    ).strip()

    def read(path):
        return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)

    write(OUTPUT, build(revision, read))


if __name__ == "__main__":
    main()
