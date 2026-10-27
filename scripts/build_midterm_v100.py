"""Project frozen wave2 completion, Terminal nulls and reference-strength limits."""

import argparse
import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path

import build_midterm_v088 as common
import build_midterm_v099 as previous

ROOT = Path(__file__).resolve().parents[1]
REPORT = previous.REPORT
OLD_REPORT = "reports/vllm-hust-midterm-20261019/中期测试报告-v099.md"
FOLDER = "reports/swe-native018-wave2-final-20261027"
SOURCE = FOLDER + "/catalog.json"
COBOL = "reports/terminal-cobol-context-20261027"
CODEIMAGE = "reports/terminal-codeimage-context-20261027"
COBOL_CONTROLS = "reports/terminal-cobol-controls-20261027"
CODEIMAGE_CONTROLS = "reports/terminal-codeimage-controls-20261027"
COUNT = "reports/terminal-count-controls-20261027"
M01 = "reports/m01-reference-strength-20261027"
OUTPUT = ROOT / "website-publication/data/midterm-v100.json"
VERSION = "V1.00"


def note(english=False):
    if english:
        return (
            "Three further original SWE attempts complete this 29-task snapshot. Protonmail made 42 successful HTTP requests; "
            "its patch applied on five paths. ConversationView ran with ten runtime TypeError failures; the other hook target "
            "reported No tests found. The original parser's four FAILED values are file-level execution placeholders, not "
            "four named assertions or fourteen functional failures; official false is preserved. "
            "Qutebrowser made 28 successful requests, applied a one-path patch and passed all ten required tests "
            "(F2P 1/1, P2P 9/9, missing 0), official true. The trailing XIO after ten passes and grade exit 0 remain recorded. "
            "Tutanota made 38 successful requests and applied a two-path patch. Type building completed, then Esbuild lacked "
            "esbuild-linux-arm64: 107 F2P names are missing, not 107 failed assertions; official false remains. "
            "All three post-application identities match graded files; resources released, no repeated Agent or grading."
        )
    return (
        "本快照新增三次原SWE尝试。Protonmail 42次HTTP成功，补丁应用于5个路径；"
        "ConversationView实际发生10个运行时TypeError，另一hook目标为No tests found。"
        "原解析器4个FAILED值是文件级execution占位，不是4个逐名断言或14项功能阴性，保留官方false。"
        "Qutebrowser 28次请求成功、补丁应用于1个路径，原10项测试全部通过：F2P 1/1、P2P 9/9、missing0、官方true；"
        "10pass后的XIO及grade退出0均保留。Tutanota 38次请求成功、补丁应用于2个路径，类型构建已完成，"
        "随后Esbuild缺少esbuild-linux-arm64，107项F2P缺失不是107项断言失败，保留官方false。"
        "三题应用后与评分文件身份一致，资源已释放，无重复Agent或评分。"
    )


def current_notes(english=False):
    old, new = (
        (
            "Current coverage is 26/731: original records 3/26, after disclosed same-answer recovery 4/26; 705 unattempted.",
            "Current coverage is 29/731: original records 4/29, after disclosed same-answer recovery 5/29; 702 unattempted.",
        )
        if english
        else (
            "当前覆盖26/731项：原记录3/26，披露同一答案恢复后4/26，705项未执行；",
            "当前覆盖29/731项：原记录4/29，披露同一答案恢复后5/29，702项未执行；",
        )
    )
    text = previous.current_notes(english)
    assert old in text
    return text.replace(old, new) + " " + note(english)


def terminal_note(english=False):
    if english:
        return (
            "Cobol's single original Agent ended on the 32K context limit after 308.477447 seconds: "
            "23 HTTP requests = 22 successes + one rejection, 22 steps/23 episodes/22 latency entries, "
            "275180 input and 13626 output tokens, exactly matching API accounting. Code-from-image ended on the same "
            "context limit after 618.272890 seconds: 35 HTTP = 34 successes + one rejection, "
            "34 steps/35 episodes/34 latency entries, 504349/9697 tokens, also matching. "
            "Neither 100 turns nor their original 900/1200-second Agent budgets were exhausted. "
            "Both original verifier/reward/CTRF remain null, not 0/1; original nested retry caps stay 3+3. "
            "Cobol's 6098-byte final script is deterministically derivable, not an original filesystem snapshot; "
            "original file SHA/mode are unknown and no recovery has run. Code-from-image's original Agent autonomously "
            "attempted OCR through its shell; no operator injected OCR or an answer, and automatic pixel input is not proven. "
            "The recorded commands did not write the required answer, artifacts contain only an empty manifest, "
            "and no answer is reconstructed. The earlier zero-step observation used the wrong key and is corrected to 34. "
            "Original raw records and nulls remain; CPU and model resources released."
        )
    return (
        "Cobol唯一原Agent在308.477447秒因32K上下文终止：23次HTTP=22成功+1拒绝，"
        "22steps/23episodes/22latencies，输入/输出token275180/13626与API逐项相等。"
        "code-from-image在618.272890秒因同一上下文上限终止：35次HTTP=34成功+1拒绝，"
        "34steps/35episodes/34latencies，token504349/9697与API相等。"
        "两题均非100turn或原900/1200秒预算耗尽，原verifier/reward/CTRF保持null，不记0/1，原nested重试上限3+3不变。"
        "Cobol可确定性派生6098字节最终脚本，但不是原文件系统快照，原文件SHA/mode未知，尚无恢复评分。"
        "code-from-image原Agent通过shell自主尝试OCR，无操作者OCR或答案注入，也未证明自动像素输入；"
        "原记录未写要求答案，artifacts仅空manifest，不重建答案。先前零步骤观察因读错键，已明确纠正为34。"
        "原始记录/null保留，CPU及模型资源已释放。"
    )


def terminal_scope(english=False):
    old, new = (
        (
            "4/89 unique tasks attempted; 85 unattempted",
            "6/89 unique tasks attempted; 83 unattempted",
        )
        if english
        else ("已尝试4/89题，85题未执行", "已尝试6/89题，83题未执行")
    )
    text = previous.terminal_scope(english)
    assert old in text
    return text.replace(old, new) + " " + terminal_note(english)


def controls_note(english=False):
    if english:
        return (
            "Original ARM64 controls: Cobol nop 1/3 reward 0 and oracle 3/3 reward 1; "
            "code-from-image nop 0/2 reward 0 and oracle 2/2 reward 1. Controls alone add no model coverage or visual qualification. "
            "Count-dataset-tokens nop 0/1 and oracle 0/1 both reward 0. Its positive control failed during tokenizer-config "
            "transport: connection reset, then missing local cache, before dataset loading or counting. "
            "This is not an observed counting-algorithm or ARM64 defect. Original sources, versions, budgets and failures "
            "remain; no model attempt or dependency-repair score was added."
        )
    return (
        "原ARM64控制：Cobol nop 1/3、reward0，oracle 3/3、reward1；"
        "code-from-image nop 0/2、reward0，oracle 2/2、reward1。控制本身不增加模型覆盖或视觉资格。"
        "count-dataset-tokens nop与oracle均0/1、reward0；正控制在tokenizer config输送时连接reset，"
        "随后本地cache缺失，尚未加载dataset或计数，不是已观察的计数算法或ARM64错误。"
        "原源码、版本、预算和失败保留，没有新增模型尝试或依赖恢复成绩。"
    )


def m01_note(english=False):
    if english:
        return (
            "Historical frozen method: B1 attained 99.9397% of the specified shared-implementation empirical reference "
            "(7.108506511466548 TFLOPS/chip). Arithmetic and the original repeatability screens are verified; "
            "workload-reference strength remains unestablished and current strong-reference acceptance is null. "
            "This does not establish near-capacity workload or hardware performance. Same-run B0/B1 compute and the "
            "84.9705% gain remain unchanged. Faster historical E16/E24/FA observations remain valid within their identities; "
            "no eligible replacement denominator can be independently recomputed from the current records. "
            "E16's original 152/200 quality evidence is recovered, with no quality rerun required. The remaining gaps are "
            "same-window necessary-work F/geometry, device-synchronized own-T endpoints, and historical loaded-weight continuity. "
            "Repeating the shared implementation alone does not close reference strength. Shared code is allowed; neither "
            "a different algorithm nor a proof of global optimality is required. A conditional E16 3C+1H proposal is not "
            "execution authorization: exact wall budget remains unset, with zero new quality windows."
            " Original V1.5's 50% gate, independent derivation and uninstrumented requirement remain separate; "
            "the round's 90% criterion is not an externally approved formal indicator amendment. "
            "Same-run accounting costs remain. Quality remains B0 151/200, B1 156/200; the corrected KV "
            "143/200 failure remains, with no full-corpus or statistical noninferiority claim."
        )
    return (
        "历史冻结口径：B1相对既定共享实现经验参考7.108506511466548 TFLOPS/芯片的达成率为99.9397%，"
        "算术与原重复性门槛已核；工况参考强度尚未闭合，当前强参考验收为null，不能据此判定接近本工况可实现算力或硬件能力。"
        "原B0/B1同轮算力及84.9705%提升不变，更快的历史E16/E24/FA观测按原身份保留，现证据尚无可独立复算的合格替换分母。"
        "E16原152/200质量证据已恢复，不需重跑质量。真正缺口是同窗必要F及实际几何、设备同步自身T端点、历史加载权重连续性。"
        "重复同一共享实现不能自动闭合参考强度；允许共享代码，不新增不同算法或全局最优证明门槛。"
        "条件式E16三次C加一次H仅为方案，wall预算尚待exact，新增质量窗口0，未授权设备执行。"
        "原V1.5的50%、独立推导及无插桩要求另列；本轮90%不是外部批准的正式指标变更，同轮计量成本保留。"
        "质量仍为B0 151/200、B1 156/200，纠正KV 143/200失败保留，不作全集或统计非劣结论。"
    )


def rows(value, english=False):
    result = previous.rows(value, english)
    labels = {
        "swe018-wave2-08-r1": (
            "10项运行时失败；另一hook未找到测试，解析为文件占位",
            "Ten runtime failures; other hook absent; file-level parser placeholders",
        ),
        "swe018-wave2-09-r1": (
            "10项通过；末尾XIO保留",
            "Ten tests passed; trailing XIO retained",
        ),
        "swe018-wave2-10-r1": (
            "Esbuild ARM64依赖缺失；107项缺失",
            "Missing Esbuild ARM64 dependency; 107 names missing",
        ),
    }
    for entry, row in zip(value["entries"], result, strict=True):
        if entry["run_id"] in labels:
            row[-1] = labels[entry["run_id"]][int(english)]
    return result


def revise_reference(reference):
    value = copy.deepcopy(reference)
    value["historical_reference_qualified"] = value["reference_qualified"]
    value["reference_qualified"] = None
    value["status"] = "historical_empirical_reference_strength_unestablished"
    value["reference_strength"] = "not_established"
    value["current_strong_reference_acceptance"] = None
    for arm in ("B0", "B1"):
        value[arm]["historical_eta_target_met"] = value[arm]["eta_target_met"]
        value[arm]["eta_target_met"] = None
    value["interpretation"] = m01_note(True)
    return value


def revise_method(data, strength):
    data["m01_method_at_v099"] = copy.deepcopy(data["m01_method"])
    data["wholechain_reference_at_v099"] = copy.deepcopy(data["wholechain_reference"])
    method = data["m01_method"]
    method["status"] = strength["status"]
    method["reference_scope"] = m01_note(True)
    method["pending_evidence"] = strength["remaining_evidence_gaps"]
    method["current"]["historical_frozen_90_result"] = method["current"]["passed"]
    method["current"]["passed"] = None
    method["current"]["reference_strength"] = "not_established"
    method["wholechain_reference"] = revise_reference(method["wholechain_reference"])
    method["whole_chain_method_revision"]["status"] = strength["status"]
    paired = method["paired_own_compute"]
    paired["status"] = strength["status"]
    paired["historical_empirical_attainment"] = paired["attainment"]
    paired["attainment"] = None
    paired["reference_method_status"] = strength["status"]
    paired["scope"] = m01_note(True)
    for row in paired["rows"].values():
        row["historical_empirical_attainment"] = row["attainment"]
        row["attainment"] = None
    paired["wholechain_reference"] = revise_reference(paired["wholechain_reference"])
    data["wholechain_reference"] = revise_reference(data["wholechain_reference"])
    data["current_summary"]["reference"] = revise_reference(
        data["current_summary"]["reference"]
    )
    data["current_summary"]["effective_compute"]["reference_strength"] = (
        "not_established"
    )
    data["current_summary"]["effective_compute"][
        "current_strong_reference_acceptance"
    ] = None
    data["m01_reference_strength"] = strength


def render_report(old, value, prior):
    assert "报告版本：V0.99" in old
    text = old.replace(
        "报告版本：V0.99（SWE增量与Circuit原始上下文终止）",
        "报告版本：V1.00（SWE增量、Terminal原始结果与M01参考强度）",
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
    text = text.replace(previous.terminal_scope(), terminal_scope())
    replacements = {
        "M01通过，其他中期验收项仍按各自证据判定。": "M01历史经验判据通过，当前工况参考强度未闭合；其他中期验收项仍按各自证据判定。",
        "达到90%，M01通过。": "达到历史90%经验判据；当前工况参考强度未闭合。",
        "B1在指定工况下通过；不是芯片标称峰值利用率": "历史90%经验判据通过；当前工况参考强度未闭合",
        "独立整链参考与适用性核验": "历史整链经验参考与当前强度边界",
        "当前整链参考采用独立运行标定的工况能力": "历史经验参考采用独立运行标定的共享实现结果",
        "参考为指定工况下有实现见证的整链能力": "参考为指定共享实现的历史整链经验值",
        "B1通过；host计数成本保留": "B1历史经验判据通过、当前参考强度未闭合；host计数成本保留",
        "达到90%": "达到历史90%经验判据",
        "| 90%判定 |": "| 历史90%经验判定 |",
    }
    for before, after in replacements.items():
        text = text.replace(before, after)
    lines = []
    for line in text.splitlines():
        if line.startswith("| SWE-bench-Pro "):
            line = (
                "| SWE-bench-Pro | "
                + common.summary(value)
                + " 原记录4/29；同答案恢复5/29，共29次尝试。 | 702题及B1同题配对未完成 |"
            )
        elif line.startswith("| TerminalBench 2.1 "):
            line = (
                "| TerminalBench 2.1 | "
                + terminal_scope()
                + " | 83题及B1同题配对未完成；原上下文失败为null，控制不增加覆盖 |"
            )
        elif re.match(r"\| 有效算力\s*\|", line):
            line = "| 有效算力 | 原B0/B1同轮F/T/C与历史经验达成率保留；B1历史η=99.9397% | 工况参考强度未闭合：同窗F/几何、同步T端点、历史权重连续性；未授权新测 |"
        lines.append(line)
    text = "\n".join(lines) + "\n"
    heading = "#### 历史整链经验参考与当前强度边界"
    assert heading in text
    text = text.replace(
        heading,
        heading
        + "\n\n"
        + m01_note()
        + "\n\n证据：[M01参考强度](../m01-reference-strength-20261027/README.md)。",
        1,
    )
    marker = "#### 0.18 Native与已有优化方案同题质量"
    addition = (
        "#### Wave2末三题与Terminal新增原始结果\n\n"
        + note()
        + "\n\n"
        + terminal_note()
        + "\n\n"
        + controls_note()
        + "\n\n证据：[SWE累计29项](../swe-native018-wave2-final-20261027/README.md)、"
        "[Cobol原始结果](../terminal-cobol-context-20261027/README.md)、"
        "[code-from-image原始结果](../terminal-codeimage-context-20261027/README.md)、"
        "[count原控制](../terminal-count-controls-20261027/README.md)。\n\n"
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
    assert value["entries"][:26] == prior["entries"]
    assert (
        value["attempted_unique_tasks"],
        value["recorded_official_resolved"],
        value["unattempted_tasks"],
    ) == (29, 4, 702)
    report = read(REPORT)
    assert "报告版本：V1.00" in report.decode()
    data.update(
        version=VERSION,
        source_commit=revision,
        report_sha256=hashlib.sha256(report).hexdigest(),
        report_url=f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{revision}/{REPORT}",
        swe_incremental_tasks=value,
        swe_same_answer_recovery_aggregate=json.loads(
            read(FOLDER + "/current-aggregate.json")
        ),
    )
    for name, folder in (
        ("terminal_cobol_original_model", COBOL),
        ("terminal_codeimage_original_model", CODEIMAGE),
        ("terminal_cobol_controls", COBOL_CONTROLS),
        ("terminal_codeimage_controls", CODEIMAGE_CONTROLS),
        ("terminal_count_controls", COUNT),
    ):
        data[name] = json.loads(read(folder + "/summary.json"))
    strength = json.loads(read(M01 + "/summary.json"))
    revise_method(data, strength)
    old_m01 = next(s for s in data["sections"] if s["id"] == "corrected-quality")
    old_m01_scopes = {key: old_m01[key] for key in ("scope", "scope_en")}
    for section in data["sections"]:
        for key, english in (("scope", False), ("scope_en", True)):
            if key in section:
                section[key] = section[key].replace(
                    old_m01_scopes[key], m01_note(english)
                )
                section[key] = section[key].replace(
                    previous.terminal_scope(english), terminal_scope(english)
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
    swe = next(s for s in data["sections"] if s["id"] == "swe-single-agent-observation")
    remaining = next(s for s in data["sections"] if s["id"] == "v15-remaining")
    m01 = next(s for s in data["sections"] if s["id"] == "m01-verdict")
    m01["title"] = (
        "M01历史经验达成与参考强度 / Historical attainment and reference strength"
    )
    m01["scope"], m01["scope_en"] = m01_note(), m01_note(True)
    for key, english in (("tables", False), ("tables_en", True)):
        swe[key][0]["rows"] = rows(value, english)
        for table in m01[key]:
            table["headers"] = [
                h.replace("90%判定", "历史90%经验判定").replace(
                    "90% assessment", "Historical 90% empirical criterion"
                )
                for h in table["headers"]
            ]
        for table in remaining[key]:
            for row in table["rows"]:
                if row[0] == "SWE-bench-Pro":
                    row[1] = (
                        common.summary(value, english) + " " + current_notes(english)
                    )
                    row[2] = (
                        "702 tasks and B1 pairing remain."
                        if english
                        else "702题及B1同题配对未完成。"
                    )
                elif row[0] == "TerminalBench 2.1":
                    row[1] = terminal_scope(english)
                    row[2] = (
                        "83 tasks and B1 pairing remain; original context failures remain null."
                        if english
                        else "83题及B1同题配对未完成；原上下文失败保持null。"
                    )
                elif row[0] in ("有效算力", "Effective compute"):
                    row[1] = m01_note(english)
                    row[2] = (
                        "Reference strength remains unestablished; no new execution authorized."
                        if english
                        else "参考强度未闭合，未授权新增执行。"
                    )
    links = [
        {"path": folder + "/summary.json", "kind": "blob", "label": label}
        for folder, label in (
            (COBOL, "Cobol原始null / Original null"),
            (CODEIMAGE, "Code-from-image原始null / Original null"),
            (COBOL_CONTROLS, "Cobol原控制 / Controls"),
            (CODEIMAGE_CONTROLS, "Code-from-image原控制 / Controls"),
            (COUNT, "Count依赖输送失败 / Dependency transport failure"),
            (M01, "M01参考强度 / Reference strength"),
        )
    ]
    catalog_link = {
        "path": SOURCE,
        "kind": "blob",
        "label": "29项累计结果 / 29-attempt catalog",
    }
    swe["evidence"].insert(0, copy.deepcopy(catalog_link))
    remaining["evidence"] = (
        [copy.deepcopy(catalog_link)] + copy.deepcopy(links) + remaining["evidence"]
    )
    m01["evidence"].insert(0, copy.deepcopy(links[-1]))
    for entry in m01["evidence"][1:]:
        entry["label"] = "历史证据 / Historical: " + entry["label"]
    for section_id, title, describe, evidence in (
        (
            "terminal-cobol-codeimage-model",
            "Terminal两项原始null / Two original unscored trials",
            terminal_note,
            links[:2],
        ),
        (
            "terminal-controls-followup",
            "Terminal控制与依赖边界 / Controls and dependency limits",
            controls_note,
            links[2:5],
        ),
        (
            "m01-reference-strength",
            "M01参考强度边界 / Reference-strength limits",
            m01_note,
            links[-1:],
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
                source_heading="Wave2末三题与Terminal新增原始结果",
            )
        )
    return data


def prepare():
    old = ROOT / OLD_REPORT
    if not old.exists():
        raw = (ROOT / REPORT).read_bytes()
        assert "报告版本：V0.99" in raw.decode()
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
