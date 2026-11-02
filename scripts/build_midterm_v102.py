"""Publish actual acceptance-gap outcomes, including the failed B1 continuous run."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = "reports/vllm-hust-midterm-20261019"
NEW = "reports/acceptance-completion-20261029"
REPORT = BASE + "/中期测试报告.md"
OUTPUT = "website-publication/data/midterm-v102.json"
GAPS = [
    "B0-specific necessary-work/path reference binding",
    "Applicable uninstrumented synchronized timing matched to counted work",
    "Same-output-budget task quality; out256 scores21/200 and19/200 cannot borrow out8192 scores",
]


def note(en=False):
    if en:
        return "M01 formal reference acceptance remains under revision. Same-run C: B0 3.8407, B1 7.1042 TFLOPS/chip. The independently calibrated B1 DAG was reproduced: modeled reference19.0819 TFLOPS/chip and diagnostic37.2302% against retained host-counted timing, not uninstrumented acceptance. B0 reference binding, matched uninstrumented timing and output-budget quality remain open. V1.5 does not impose a separate executable whole-chain witness or a three-run implementation median."
    return "M01正式参考验收修订中。同轮复算：B0 3.8407、B1 7.1042 TFLOPS/芯片。独立标定B1依赖图已回验：模型参考19.0819 TFLOPS/芯片，对原计数轮诊断比值37.2302%，不是无插桩验收。仍缺B0参考绑定、适用无插桩计时及同预算质量。V1.5不另要求整链参考程序见证或三次实现运行中位数。"


def current_reference(read):
    value = json.loads(read(BASE + "/m01-reference-revision-20261029/summary.json"))
    value.update(
        method_path=NEW + "/reference-method.md",
        pending_evidence=GAPS,
        model_diagnostic=json.loads(
            read(NEW + "/reference-reassessment-20261029.json")
        ),
    )
    return value


def current_method(read, ref):
    method = json.loads(read(BASE + "/current-method-v101.json"))
    method.update(
        method_version="reference-model-reassessment-20261029",
        pending_evidence=GAPS,
        wholechain_reference=ref,
        reference_scope=note(True),
    )
    method["paired_own_compute"].update(wholechain_reference=ref, scope=note(True))
    method["requirements"]["reference"]["path"] = NEW + "/reference-method.md"
    method["whole_chain_method_revision"]["method_path"] = NEW + "/reference-method.md"
    method["reference_binding"] = {
        "method": "Independent calibration and dependency/resource model, V1.5 section3.4.1",
        "separate_by_arm_when_work_precision_or_path_differs": True,
        "additional_executable_witness_required": False,
        "three_run_implementation_median_required": False,
    }
    method["historical_identity_v101"] = method["identity"]
    method["identity"] = {
        "kind": "original_v15_reference_model_reassessment",
        "verification_date": "2026-10-29",
        "authority_scope": "Original V1.5 method; not a new acceptance amendment or external approval",
        "external_acceptance_approval_record": None,
        "formal_indicator_amendment": False,
        "original_v15": method["historical_identity_v101"]["original_v15"],
        "stage_thresholds_percent": {"midterm": 50, "final": 90},
    }
    return method


def completion(read):
    continuous = json.loads(read(NEW + "/continuous-summary.json"))
    quality = json.loads(read(NEW + "/out256-quality-audit.json"))
    cost = json.loads(
        read("reports/acceptance-gaps-20261029/recovered-serving-summary.json")
    )
    return dict(
        reference=json.loads(read(NEW + "/reference-reassessment-20261029.json")),
        continuous={k: v for k, v in continuous.items() if k != "b0"},
        b0_continuous={k: v for k, v in continuous["b0"].items() if k != "rows"},
        out256_quality={
            **quality,
            "arms": [
                {k: v for k, v in arm.items() if k != "rows"} for arm in quality["arms"]
            ],
        },
        recovered_serving=cost,
        five_task_stability=json.loads(read(NEW + "/five-task-stability.json")),
    )


def section(values):
    formal = [
        p for p in values["b0_continuous"]["phases"] if p["name"].endswith("formal")
    ]

    def table(en):
        rows = []
        for p in formal:
            rows.append(
                [
                    "B0 " + p["name"],
                    str(p["requests"]),
                    f"{p['cohort_output_tokens_per_second']:.4f}",
                    f"{p['timing']['ttft_ms']['p95']:.2f} / {p['timing']['ttft_ms']['p99']:.2f}",
                    f"{p['timing']['tpot_ms']['p95']:.2f} / {p['timing']['tpot_ms']['p99']:.2f}",
                    "0",
                ]
            )
        rows.append(["B1 4096-formal", "8", "—", "—", "—", "8"])
        return dict(
            headers=[
                "Arm / phase" if en else "方案/阶段",
                "Requests" if en else "请求数",
                "token/s",
                "TTFT P95/P99 (ms)",
                "TPOT P95/P99 (ms)",
                "Failures" if en else "失败",
            ],
            rows=rows,
        )

    return dict(
        id="acceptance-completion",
        title="当前证据与缺口清单 / Current evidence and remaining gaps",
        source_heading="V1.02 · 2026-10-29 · 固定B0补测 / Fixed-B0 supplement",
        scope="固定B0的4K/16K连续窗口已完成；B1状态卸载发生锁页主机内存分配失败，24个预热成功后8个正式请求失败，16K未执行。out256质量复算21/200对19/200，业务成本验收未闭合。"
        + note(),
        scope_en="Fixed B0 completed4K/16K continuous windows. B1 state offload failed pinned host allocation after24 successful warmup requests;8 formal requests failed and16K was not run. Out256 quality rescored21/200 versus19/200; business-cost acceptance remains open. "
        + note(True),
        tables=[table(False)],
        tables_en=[table(True)],
        evidence=[
            dict(path=NEW + "/" + name, kind="blob", label=label)
            for name, label in (
                ("README.md", "本轮结果 / Results"),
                ("continuous-raw.tar.gz", "连续补测原件 / Raw continuous runs"),
                (
                    "continuous-summary.json",
                    "逐请求和窗口复算 / Request and window metrics",
                ),
                ("out256-quality-audit.json", "同预算评分 / Same-budget scoring"),
                ("reference-method.md", "参考方法 / Reference method"),
                ("five-task-stability.json", "五任务核查 / Five-task review"),
            )
        ],
    )


def build(revision, read):
    data = json.loads(read("website-publication/data/midterm-v101.json"))
    ref = current_reference(read)
    method = current_method(read, ref)
    values = completion(read)
    data.update(
        version="V1.02",
        source_commit=revision,
        report_url=f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{revision}/{REPORT}",
        report_sha256=hashlib.sha256(read(REPORT)).hexdigest(),
        wholechain_reference=ref,
        m01_method=method,
        acceptance_completion=values,
    )
    data["current_summary"]["reference"] = ref
    data["paired_own_compute"].update(wholechain_reference=ref, scope=note(True))
    data["corrected_quality"]["reference"]["current_result_path"] = (
        NEW + "/reference-reassessment-20261029.json"
    )
    data["milestone_contract"]["method"] = method
    data["milestone_contract"]["method_source"]["sha256"] = hashlib.sha256(
        read(BASE + "/current-method.json")
    ).hexdigest()
    data["milestone_contract"]["decision_relevant_gaps"]["m01"] = GAPS
    data["method_history"] = json.loads(read(BASE + "/method-history-index.json"))
    added = section(values)
    data["sections"].insert(0, added)
    for s in data["sections"][1:]:
        if s["id"] in (
            "m01-verdict",
            "milestone-v15",
            "v15-remaining",
            "corrected-quality",
        ):
            prior_scopes = {key: s.get(key, "") for key in ("scope", "scope_en")}
            s.update(scope=note(), scope_en=note(True))
            if s["id"] == "v15-remaining":
                for key, marker in (("scope", " 成本采用"), ("scope_en", " Cost uses")):
                    if marker not in prior_scopes[key]:
                        raise ValueError("Existing cost/stability scope missing")
                    s[key] += marker + prior_scopes[key].split(marker, 1)[1]
            s["evidence"] = copy.deepcopy(added["evidence"]) + s["evidence"]
    return data


def prepare():
    def read(path):
        return (ROOT / path).read_bytes()

    for original, archived in (
        (REPORT, BASE + "/中期测试报告-v101.md"),
        (BASE + "/current-method.json", BASE + "/current-method-v101.json"),
        (BASE + "/method-history-index.json", BASE + "/method-history-index-v101.json"),
    ):
        if not (ROOT / archived).exists():
            (ROOT / archived).write_bytes(read(original))
    ref = current_reference(read)
    method = current_method(read, ref)
    (ROOT / (BASE + "/current-method.json")).write_text(
        json.dumps(method, ensure_ascii=False, indent=2) + "\n"
    )
    history = json.loads(read(BASE + "/method-history-index-v101.json"))
    history["current_revision"] = NEW + "/reference-reassessment-20261029.json"
    history["v101_snapshot"] = BASE + "/current-method-v101.json"
    (ROOT / (BASE + "/method-history-index.json")).write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n"
    )
    report = read(BASE + "/中期测试报告-v101.md").decode()
    report = report.replace(
        "报告版本：V1.01（有效算力参考纠正；其他结果沿用V0.99）",
        "报告版本：V1.02（固定B0连续补测、同预算质量与参考模型复核）",
    ).replace("编制日期：2026年10月26日", "编制日期：2026年10月29日")
    paragraphs = report.split("\n\n")
    for i, p in enumerate(paragraphs):
        if p.startswith("M01参考修订中") or p.startswith("**官方0.18/0.18 B0与"):
            paragraphs[i] = note()
        elif p.startswith("**总体结论："):
            paragraphs[i] = (
                "**固定B0的4K/16K连续窗口已完成；优化侧在4K正式阶段发生状态卸载锁页内存分配失败，配对稳定性未通过。原out256性能轮同预算评分为21/200对19/200，不能借用out8192质量证明业务成本验收。** "
                + note()
            )
        elif p.startswith("72条实际输入不少于16,384 token的探针"):
            paragraphs[i] = (
                "既有长输入探针保留；新固定B0/B1检索均6/6，连续补测的B1故障另列。"
                + note()
            )
    report = "\n\n".join(paragraphs)
    lines = []
    for line in report.splitlines():
        first_cell = line.split("|")[1].strip() if line.startswith("|") else ""
        if first_cell == "M01":
            line = "| M01 | 单芯片有效算力 | 中期η≥50%；最终η≥90% | B0 C=3.8407、B1 C=7.1042；B1模型参考19.0819、诊断37.2302% | 正式η仍未确定：B0参考、无插桩计时及质量未闭合 |"
        elif first_cell == "M05":
            line = "| M05 | 数据集运行与长上下文稳定性 | 五任务异常核查；实际≥16K连续区间及可比4K窗口 | B0 4K/16K完成153/73请求；B1正式4K的8请求失败，16K未执行 | 当前合成压力配置实测失败；五任务记录及缺失边界另列 |"
        elif first_cell == "M06":
            line = "| M06 | 单位成功输出成本及利用率 | 成本降低≥10%；高负载利用率明显改善 | 固定B0租赁估算66.2253→35.4849元/百万token，降低46.4178% | out256质量21/200对19/200；精确利用率与业务适用性未闭合 |"
        elif line.startswith("| 成本与利用率 "):
            line = "| 成本与利用率 | 固定B0原件和公开报价已恢复，租赁估算下降46.4178% | 同预算质量、适用SLO与精确配对利用率 |"
        lines.append(line)
    report = "\n".join(lines) + "\n"
    report = report.replace(
        "本次未启动新实验", "该参考计数复算未启动新实验；本版连续补测另列"
    )
    report = report.replace(
        "固定B0的综合月费依据、同一成功请求集合和同期时长加权利用率适用证据仍待核验",
        "固定B0的请求集合与类似机器报价已复核，见8.3节；同预算质量及精确配对利用率仍未闭合",
    )
    report = report.replace(
        "工况参考能力强度与整链见证；替代C参未确定",
        "B0参考绑定、无插桩计时与同预算质量；B1模型诊断不作正式通过",
    )
    report = report.replace(
        "本报告以卡秒/token衡量资源消耗，并在两组每卡综合费率相同的条件下计算成本变化率。由于缺少经确认的采购或计费依据，本版不折算人民币金额。",
        "下表保留历史开发栈资源系数。固定B0原件恢复后已按用户指定的公开类似机器报价另列租赁估算，见8.3节；不是本机账单或完整业务成本验收。",
    )
    addition = """### 当前证据与缺口清单

| 项目 | 当前结果 | 仍影响结论的条件 |
| --- | --- | --- |
| M01 | B1独立标定模型参考19.0819 TFLOPS/芯片；原计数轮诊断37.2302% | B0路径绑定、无插桩计时及同预算质量 |
| 吞吐/时延与成本 | 固定B0原200请求恢复；同费率租赁估算66.2253→35.4849元/百万输出token | out256评分21/200→19/200；质量/SLO未闭合 |
| 连续稳定性 | B0 4K/16K正式请求153/73、零失败；B1预热24成功后正式8失败 | 状态卸载锁页主机内存分配故障；B1 16K未执行 |
| 五任务 | 已复核MMLU、HLE及FrontierScience原记录，关联Agent既有上下文失败 | HLE原退出状态缺失；SWE/Terminal部分任务与失败保留 |

依据：[补齐结果](../acceptance-completion-20261029/README.md)、[原始连续运行](../acceptance-completion-20261029/continuous-raw.tar.gz)、[参考推导](../acceptance-completion-20261029/reference-method.md)、[同预算质量](../acceptance-completion-20261029/out256-quality-audit.json)。

"""
    report = report.replace(
        "## 二、大纲指标对照表", addition + "## 二、大纲指标对照表", 1
    )
    report = report.replace(
        "## 八、单位输出成本与设备利用率",
        """### 7.4 固定0.18连续补测

同权重、BF16/TP2、物理5/6两卡，实际输入4115/16407 token，C8闭环，输出256，60秒预热、300秒正式发起后完整排空。
B0两个长度分别153/73成功、零失败；B1预热24成功后，正式4K的8请求因状态卸载锁页内存分配失败而报错，16K未执行。原失败与后续释放回执保留。
本次是合成压力补充，不代替五任务质量；窗口统计及P95/P99见[逐请求复算](../acceptance-completion-20261029/continuous-summary.json)，原输入、SSE、采样与源码见[原件](../acceptance-completion-20261029/continuous-raw.tar.gz)。

## 八、单位输出成本与设备利用率""",
        1,
    )
    report = report.replace(
        "## 九、测试结论",
        """### 8.3 固定B0恢复复算与质量适用性

原同200请求、同物化输入、各保留两卡的同步窗口复算：吞吐215.0116→401.2743 token/s，平均TTFT154605.0625→86876.8906 ms，平均TPOT66.5920→33.0916 ms/token。
按天翼云类似双卡910B公开36908元/月、30天分析假设（云盘/公网另计），租赁估算66.2253→35.4849元/百万输出token，降低46.4178%；不作本机综合账单。
B1同区间时长加权利用率22.3946%；B0缺精确时钟桥，条件采样界限9.4245–16.7177%，不能写为精确测量点。
本次原out256回答按原评分器离线复算21/200对19/200，全部达到256输出预算，fallback解析不证明完整答案。该服务测量不能借用out8192质量；业务成本验收未闭合。
证据、公开报价来源及复算见[固定基线成本](../acceptance-gaps-20261029/README.md)和[同预算质量](../acceptance-completion-20261029/out256-quality-audit.json)。

## 九、测试结论""",
        1,
    )
    (ROOT / REPORT).write_text(report)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--revision", default="HEAD")
    a = p.parse_args()
    if a.prepare:
        prepare()
        return
    revision = subprocess.check_output(
        ["git", "rev-parse", a.revision], cwd=ROOT, text=True
    ).strip()

    def read(path):
        return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)

    (ROOT / OUTPUT).write_text(
        json.dumps(build(revision, read), ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
