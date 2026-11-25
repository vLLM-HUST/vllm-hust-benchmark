"""Append sealed A01 evidence without rewriting fixed-five measurements or old reports."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ANALYSIS_SHA = "0e3947af17badfa7edb244a2ca397c5367480678eeb416bb1c1f886d02c08c5f"  # pragma: allowlist secret -- fixed public evidence digest

PAIRED_SHA = "eb7d3f6f5c62b4793e47a0b4d98ddb0ac80deb509ad4e12d6db4e2848524d327"  # pragma: allowlist secret -- fixed public source digest


def append(report, evidence, artifacts):
    if evidence["source_paired_sha256"] != PAIRED_SHA:
        raise ValueError("Foreign A01 capture")
    rows = evidence["comparisons"]
    if [r["phase"] for r in rows] != [
        "4096-formal",
        "16384-formal",
        "32768-formal",
        "multi-tenant-formal",
    ]:
        raise ValueError("Incomplete A01 phases")
    for row in rows:
        for role in ("B0", "B1"):
            arm = row["arms"][role]
            if (
                arm["metrics"]["failed"]
                or len(arm["windows"]) != 5
                or any(w["metrics"]["failed"] for w in arm["windows"])
            ):
                raise ValueError("A01 contains failed or missing windows")
        ratio = (
            row["arms"]["B1"]["metrics"]["output_tokens_per_second"]
            / row["arms"]["B0"]["metrics"]["output_tokens_per_second"]
        )
        if (
            abs(ratio - row["throughput_ratio"]) > 1e-10
            or abs(100 * (1 - 1 / ratio) - row["cost_reduction_percent"]) > 1e-10
        ):
            raise ValueError("A01 ratios do not recompute")
    d = deepcopy(report)
    assessment = evidence["qualitative_assessment"]
    d["acceptance_supplement"] = dict(
        schema="sealed-A01-context-tenant-publication/v1",
        report_date="2026-11-15",
        execution_complete=True,
        source_analysis_sha256=ANALYSIS_SHA,
        analysis_path="data/context-tenant-a01-20261115/analysis-v1.json",
        chart_path="data/context-tenant-a01-20261115/paired-overview.svg",
        scope=deepcopy(evidence["scope"]),
        comparisons=deepcopy(rows),
        tenant_summary=deepcopy(evidence["tenant_summary"]),
        context_summary=deepcopy(evidence["context_summary"]),
        individual_window_latency_regressions=deepcopy(
            evidence["individual_window_latency_regressions"]
        ),
        qualitative_assessment=deepcopy(assessment),
        artifacts=deepcopy(artifacts),
        native_tools_release="acceptance-tools-20261115-r4",
        fixed_five_datasets_rerun=False,
    )
    context = d["formal_targets"]["3.4.3"]
    context.update(
        accepted=assessment["milestone_context"]["accepted"],
        status="达成目标"
        if assessment["milestone_context"]["accepted"]
        else "未达成目标",
        display_value="16K/C8：142.54 token/s，4K的0.599倍；五个窗口无失败",
        display_value_en="16K/C8:142.54 token/s,0.599 of4K; all five windows complete",
        accepted_scope="A01: five original public task histories,16K–24K/C8,300s,output256",
        reason="固定五项结果保留。新增真实轨迹两臂4K/16K/32K配对：16K吞吐保持同量级，各档五个窗口的尾时延没有持续恶化。全部窗口及四条局部尾时延回归均披露；不同上下文长度的输出吞吐并不相同。",
        evidence_path="../context-tenant-a01-20261115/analysis-v1.json",
        source_analysis_sha256=ANALYSIS_SHA,
    )
    final = d["project_final_assessment"]["targets"]
    final["3.4.3"].update(
        attainment="demonstrated"
        if assessment["final_context"]["accepted"]
        else "not_demonstrated",
        evidence_summary="A01实测32–48K/C8，五个60秒提交窗口完成；吞吐与尾时延没有持续恶化。",
        evidence_summary_en="A01 tested32–48K/C8: all five60s submission windows complete without sustained throughput/tail deterioration.",
        reason=assessment["final_context"]["reason"],
        accepted_scope="A01: tested32K–48K original task input band,C8,300s,output256",
        source_analysis_sha256=ANALYSIS_SHA,
    )
    final["3.4.4"].update(
        attainment="demonstrated"
        if assessment["final_cost"]["accepted"]
        else "not_demonstrated",
        evidence_summary="四租户C24：同费率单位token成本降低62.89%，双卡平均AI Core利用率82.77%；四租户均完成请求。",
        evidence_summary_en="Four tenants/C24: equal-rate unit token cost−62.89%, mean AI Core utilization82.77%; all tenants complete requests.",
        reason=assessment["final_cost"]["reason"],
        accepted_scope="A01: four logical client tenants sharing one service,C24,300s,output256",
        source_analysis_sha256=ANALYSIS_SHA,
    )
    milestones = d["milestone_assessment"]
    milestones["scope"] = (
        "Project achievements with explicitly scoped evidence; current-version M01 requalification remains separate."
    )
    milestones["targets"]["3.4.1"] = deepcopy(
        d["formal_targets"]["3.4.1"]["historical_evidence"]
    )
    milestones["targets"]["3.4.3"] = deepcopy(context)
    milestones["all_met"] = all(r["accepted"] for r in milestones["targets"].values())
    d["report_revision_date"] = "2026-11-15"
    d["supplement_append_provenance"] = dict(
        source_analysis_sha256=ANALYSIS_SHA,
        original_fixed_five_report_unchanged=True,
        current_version_compute_efficiency_not_substituted=True,
    )
    for key in (
        "datasets",
        "source_provenance",
        "continuous",
        "stability_assessment",
        "price_basis",
        "compute_reference_statistics",
    ):
        if d[key] != report[key]:
            raise ValueError("Immutable prior measurements changed: " + key)
    if d["formal_targets"]["3.4.1"] != report["formal_targets"]["3.4.1"]:
        raise ValueError("Current-version compute assessment changed")
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--artifacts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if hashlib.sha256(a.analysis.read_bytes()).hexdigest() != ANALYSIS_SHA:
        raise ValueError("A01 analysis source differs")
    result = append(
        json.loads(a.report.read_bytes()),
        json.loads(a.analysis.read_bytes()),
        json.loads(a.artifacts.read_bytes()),
    )
    raw = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if a.out.exists() and a.out.read_text() != raw:
        raise ValueError("Preserve old report bytes; use a new report version")
    a.out.write_text(raw)


if __name__ == "__main__":
    main()
