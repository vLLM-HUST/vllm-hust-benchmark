"""Publish V1.5 interpretation without rewriting frozen V0.72 or experimental data."""

import hashlib
import json
import subprocess
from pathlib import Path

from build_midterm_publication import section, tables

ROOT = Path(__file__).resolve().parents[1]
REVISION = "77a12492dcd07c7eec85ea0a01de6a896f1cfe74"  # pragma: allowlist secret
REPORT_DIR = "reports/vllm-hust-midterm-20261019"
REPORT = REPORT_DIR + "/中期测试报告.md"
OUTPUT = ROOT / "website-publication/data/midterm-v074.json"


def pinned(path):
    return subprocess.check_output(["git", "show", f"{REVISION}:{path}"], cwd=ROOT)


def locator(path, label="核验原文与复算证据 / Verified source and reuse evidence"):
    return {"path": path, "kind": "blob", "label": label}


def build():
    raw = pinned(REPORT)
    text = raw.decode()
    assert "报告版本：V0.74" in text
    data = json.loads(pinned("website-publication/data/midterm-v072.json"))
    audit = json.loads(pinned(REPORT_DIR + "/milestone-v15/reuse-audit.json"))
    data.update(
        version="V0.74",
        source_commit=REVISION,
        report_url=f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{REVISION}/{REPORT}",
        report_sha256=hashlib.sha256(raw).hexdigest(),
        m01_method=audit["method"],
        milestone_contract=audit,
    )
    by_id = {s["id"]: s for s in data["sections"]}
    verdict = by_id["m01-verdict"]
    verdict["tables"] = tables(section(text, "#### M01 本轮验收判定"))
    for key in ("scope", "scope_en"):
        verdict[key] = verdict[key].replace("V5.27", "V1.5")
        by_id["m01-historical"][key] = by_id["m01-historical"][key].replace(
            "V5.27", "V1.5"
        )
    verdict["scope"] += (
        " 已取得200请求同步B1包络120.65347687713802秒；F与同工作量参考仍待关联，不以旧约62秒替换。"
    )
    verdict["scope_en"] += (
        " A 200-request synchronized B1 envelope of 120.65347687713802 seconds is now available; effective F and same-workload reference still need attribution. Do not substitute the older ~62-second run."
    )
    verdict["tables_en"][0]["rows"][1] = [
        "Uninstrumented synchronized T",
        "200/0 requests; envelope 120.65347687713802 s",
        "Measured; associate this workload with valid F and reference. Initial compiler-cache conditions differ from the old ~62 s run; no estimated deduction.",
    ]
    verdict["tables_en"][0]["rows"][2] = [
        "Independent reference and dependencies",
        "75 communication graph shapes measured; 10,800 verified samples per rank",
        "Model-level useful counts, dependencies, overlap and applicability remain open; do not add resource times",
    ]
    verdict["tables_en"][0]["rows"][3][1] = "V1.5 formulas; 50% threshold"
    verdict["evidence"] = [
        locator(REPORT_DIR + "/milestone-v15/method-status.json"),
        locator(REPORT_DIR + "/milestone-v15/project4-v1.5.txt"),
        locator(REPORT_DIR + "/m01-recovery/collective-graph-20261023/manifest.json"),
        locator(REPORT_DIR + "/m01-recovery/boundary-measured-20261023/manifest.json"),
    ]
    for ident in ("paired-lifecycle", "resources"):
        by_id[ident]["scope"] += (
            " V1.5按测量区间分摊费用；全生命周期结果仅作历史补充，不是当前成本否决条件。本组开发栈资源系数不构成固定B0货币成本验收。"
        )
        by_id[ident]["scope_en"] = (
            "Historical development-stack control and B1, not the fixed 0.18 B0. Request-window and full-lifecycle resource costs are separate. "
            "V1.5 allocates fees over the measurement interval: lifecycle increases are not a current cost veto. "
            "No monetary acceptance without a documented fee basis, matched successful-request population, quality/SLO and same-interval utilization. "
            "All original values, tail-latency regressions and shutdown limitations remain unchanged."
        )
    by_id["resources"]["scope_en"] += (
        " Historical request resource cost decreased 14.63%, lifecycle cost increased 2.68%, and P99 TPOT increased 57.07%; no RMB fee basis."
    )
    for ident in (
        "long-context",
        "current-b1-long-context",
        "long-context-parser",
        "long-diagnostic",
    ):
        by_id[ident]["scope"] += (
            " 本组为历史探针或合成诊断，不替代V1.5五项任务及实际≥16K连续稳定性判定。"
        )
        by_id[ident]["scope_en"] = (
            "Retained synthetic long-input probe or diagnostic, not five-task continuous stability acceptance. "
            "V1.5 requires actual >=16K continuous intervals, comparable 4K load, window throughput and P95/P99 TTFT/TPOT, "
            "plus task completion and anomaly checks. Compatible 32K records can be reused. Original scores and parser-repair confirmations remain separate."
        )
    by_id["long-context-parser"]["scope_en"] += (
        " Independent parser-repair confirmation: 6/6 strict body matches, zero failures; original 0/6 retained."
    )
    by_id["current-b1-long-context"]["scope_en"] += (
        " Original strict body score 0/6; supplemental final-answer parsing 6/6 does not replace the original score."
    )
    current = {
        "id": "milestone-v15",
        "title": "V1.5 当前依据与结论 / V1.5 current method and conclusions",
        "scope": "当前依据为项目里程碑大纲V1.5课题四，35B、固定B0 0.18/0.18，B1基座不限且标清实际版本。配置实现可不同，任务、资源、质量/SLO和统计条件保持可比。不削弱B0、不现场择优、不拼接最佳成绩。优先复用复算，不新增补测或完整生命周期成本要求。",
        "scope_en": "Project Milestone Outline V1.5, Project 4: 35B; fixed B0 0.18/0.18; B1 has no base-version restriction and actual versions are disclosed. Implementation choices may differ under comparable task, resource, quality/SLO and statistical conditions. No weakened B0, live tuning or splicing best metrics. Reuse/recalculate first; no new measurement or full-lifecycle cost requirement.",
        "tables": tables(section(text, "### V1.5 当前方法与复用范围")),
        "tables_en": [
            {
                "headers": ["Item", "V1.5 method", "Current boundary"],
                "rows": [
                    [
                        "3.4.1 Effective compute",
                        "C=F/(k*T*10^12); eta=T_reference/T*100%; >=50%",
                        "Synchronized B1 timing and graph communication samples measured; effective F and same-workload model reference still require attribution",
                    ],
                    [
                        "3.4.2 Performance",
                        "Throughput >=10%; mean TTFT and TPOT each >=5%; common quality/SLO",
                        "C16 fixed-B0 descriptive gain +258.30%; client/resource identity and acceptance applicability remain under review",
                    ],
                    [
                        "3.4.3 Stability",
                        "Five-task anomalies; actual >=16K continuous intervals and comparable 4K windows",
                        "Six probes are not full-task acceptance; compatible 32K records reusable",
                    ],
                    [
                        "3.4.4 Cost",
                        "Expense=P*k*T/(24*D*3600); cost=1e6*expense/N; >=10% reduction and improved utilization",
                        "All occupied cards, same successful-request population and time-weighted interval utilization; fee basis and fixed-B0 comparison still open",
                    ],
                ],
            }
        ],
        "source_heading": "V1.5 当前方法与复用范围 / Current method and reuse",
        "evidence": [
            locator(REPORT_DIR + "/milestone-v15/reuse-audit.json"),
            locator(REPORT_DIR + "/milestone-v15/project4-v1.5.txt"),
            locator(
                "reports/c16-baseline018-reuse-20261022/manifest.json",
                "C16 原始归档索引 / C16 original archive index",
            ),
        ],
    }
    data["sections"].insert(0, current)
    data["kv_audit"]["scope_en"] = (
        "Historical E16/E24 per-request recomputation controls: independent services, unique salts, no repeated salts across the two runs, "
        "and logged prefix hit rates all zero. This does not exclude Mamba/GDN state contamination or low-level faults, "
        "does not establish causal performance gains, and does not transfer to Microflow or other configurations. No new hardware test."
    )
    return data


if __name__ == "__main__":
    OUTPUT.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n")
