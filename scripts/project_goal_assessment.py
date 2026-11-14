"""Assess project final goals using the user's confirmed reference/model interpretation."""

from decimal import Decimal
import copy
import json
from pathlib import Path

REFERENCE_SHA256 = "95c6bd465c4beb558cd25284e221a166e33634ea2f9b6d44364d2b1114863bd0"  # pragma: allowlist secret -- public evidence SHA256
REFERENCE_URL = (
    "https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/"
    "53541118b3461705f1e61532710ff0a759d59af0/"
    "reports/m01-qualified-pair-20261103/summary.json"
)
PROFILE_PATH = (
    Path(__file__).resolve().parents[1]
    / "reports/project-final-goals-20261113/final-goals-v1.json"
)


def unmeasured_current_compute_target(archived):
    """The five-profile driver supplies archived M01 evidence, not a new calibration."""
    historical = copy.deepcopy(archived)
    return {
        "name": "有效算力效率 ≥50%",
        "status": "current_workload_reference_not_measured",
        "accepted": False,
        "accepted_scope": None,
        "measured_value": None,
        "display_value": "本轮未验证；归档工况另列",
        "display_value_en": "Current workload unverified; archive separate",
        "reason": "本轮五数据集性能运行未采集完整必要算量、独立标定、整链验证和双卡同步计时，不能据此计算本轮3.4.1。历史MMLU200/out256的≥59.76%仍保留在historical_evidence，仅适用于其归档工况。现场采集包已接入主分支，但完整工况认证模块尚未齐备，不将分项算量或时间线冒充参考值。",
        "reason_en": "This five-profile campaign does not collect complete necessary-work accounting, independent calibration, whole-chain validation and synchronized timing for its current workload. The archived MMLU200/out256 lower bound remains separate; the live collection kit still requires complete workload certifiers.",
        "required_evidence": [
            "complete_necessary_work_ledger",
            "independent_calibration",
            "independent_whole_chain_validation",
            "synchronized_dual_card_timing",
            "workload_applicability_certification",
        ],
        "historical_evidence": historical,
        "reference_interpretation": "同工况独立参考",
        "kit_source_url": "https://github.com/vLLM-HUST/vllm-hust-engine/blob/9287f42708880de74e70145c5e8d4bf9d8fdca6a/third-party-test-kit/参考算力标定与复算.md",
    }


def assess(reference, datasets, profile=None):
    profile = profile or json.loads(PROFILE_PATH.read_bytes())
    criteria = profile["targets"]
    if (
        profile["confirmed_interpretation"]["operational_compute_denominator"]
        != "同工况独立参考"
    ):
        raise ValueError("Final goal reference interpretation changed")
    b1 = next(arm for arm in reference["arms"] if arm["role"] == "B1")
    lower = Decimal(b1["aggregate"]["eta_percent_lower"])
    threshold = criteria["3.4.1"]["efficiency_percent_min"]
    demonstrated = bool(reference["evidence_qualified"] and lower >= Decimal(threshold))
    targets = {
        "3.4.1": {
            "name": "单芯片有效算力",
            "requirement": "同工况参考效率 ≥90%",
            "requirement_en": "Workload-reference efficiency ≥90%",
            "threshold_percent": threshold,
            "formula": profile["confirmed_interpretation"]["formula"],
            "milestone_threshold_percent": reference["milestone_percent"],
            "evidence_lower_bound_percent": str(lower),
            "evidence_summary": "历史out256保守下界 ≥59.76%",
            "evidence_summary_en": "Historical out256 lower bound ≥59.76%",
            "lower_bound_distance_to_target_pp": str(Decimal(threshold) - lower),
            "source_url": REFERENCE_URL,
            "source_sha256": REFERENCE_SHA256,
            "criterion_source_verified": True,
            "attainment": "demonstrated" if demonstrated else "not_demonstrated",
            "reason": "原文“理论峰值”按用户确认解释为同工况独立参考，模型沿用已确认35B。历史out256的B1保守下界约59.76%，未证明90%；下界距离不是实际效率差距，不反推真实效率必低90%，不套用于本轮完整回答或全五项。",
        }
    }
    performance = []
    costs = []
    for dataset in datasets:
        change = dataset["changes_percent"]
        ratio = (
            dataset["arms"]["B1"]["output_tokens_per_second"]
            / dataset["arms"]["B0"]["output_tokens_per_second"]
        )
        ttft_reduction = -change["mean_ttft_ms"]
        tpot_reduction = -change["mean_tpot_ms"]
        performance.append(
            {
                "dataset": dataset["id"],
                "throughput_ratio": ratio,
                "mean_ttft_reduction_percent": ttft_reduction,
                "mean_tpot_reduction_percent": tpot_reduction,
                "numeric_threshold_met": ratio
                >= criteria["3.4.2"]["throughput_ratio_min"]
                and ttft_reduction
                >= criteria["3.4.2"]["mean_ttft_reduction_percent_min"]
                and tpot_reduction
                >= criteria["3.4.2"]["mean_tpot_reduction_percent_min"],
            }
        )
        reduction = -change["unit_cost_change_percent"]
        costs.append(
            {
                "dataset": dataset["id"],
                "equal_rate_unit_cost_reduction_percent": reduction,
                "numeric_threshold_met": reduction
                >= criteria["3.4.4"]["unit_token_cost_reduction_percent_min"],
            }
        )
    misses = [p["dataset"] for p in performance if not p["numeric_threshold_met"]]
    targets["3.4.2"] = {
        **criteria["3.4.2"],
        "requirement": "吞吐 ≥2×；平均TTFT/TPOT分别降低 ≥20%",
        "requirement_en": "Throughput ≥2×; mean TTFT/TPOT each reduced ≥20%",
        "evidence_summary": "按五项逐一核算；未达数值门槛：" + ", ".join(misses)
        if misses
        else "五项吞吐/时延数值门槛均达到",
        "evidence_summary_en": "Profiles missing numeric targets: " + ", ".join(misses)
        if misses
        else "All five profiles meet throughput/latency numeric targets",
        "criterion_source_verified": True,
        "attainment": "not_demonstrated" if misses else "demonstrated",
        "numeric_results_by_profile": performance,
        "all_profile_numeric_thresholds_met": all(
            p["numeric_threshold_met"] for p in performance
        ),
        "reason": "沿用本轮可比B0/B1、固定任务和服务协议，按各项实际吞吐、TTFT/TPOT分别判定，不用某项最好值覆盖其他负载。质量与裁判未决数独立保留；数值达标不构成统计质量等价证明。",
    }
    targets["3.4.3"] = {
        **criteria["3.4.3"],
        "requirement": "≥32K：吞吐平稳，TTFT/TPOT P95/P99稳定、无明显退化",
        "requirement_en": "≥32K: stable throughput and TTFT/TPOT P95/P99, no obvious regression",
        "evidence_summary": "本轮独立C8仅4K/16K；无可验收32K连续窗口",
        "evidence_summary_en": "Current C8 covers4K/16K; no qualified32K continuous window",
        "criterion_source_verified": True,
        "attainment": "not_demonstrated",
        "observed_input_at_least32768_cases": sum(
            a.get("input_at_least32768_cases", 0)
            for d in datasets
            for a in d["arms"].values()
        ),
        "observed_input_plus_output_at_least32768_cases": sum(
            a.get("input_plus_output_at_least32768_cases", 0)
            for d in datasets
            for a in d["arms"].values()
        ),
        "continuous_32k_stability_verified": False,
        "reason": "16K里程碑窗口不证明32K最终目标；偶发总上下文长度也不能代替连续吞吐曲线与P95/P99尾延迟稳定性证据。保留原文定性要求，不自行增加尾延迟阈值。",
    }
    targets["3.4.4"] = {
        **criteria["3.4.4"],
        "requirement": "单位token成本降低 ≥30%；多租户场景维持较高利用率",
        "requirement_en": "Unit token cost reduced ≥30%; high utilization with multiple tenants",
        "evidence_summary": "各项同费率估算降幅单列；多租户证据未完成",
        "evidence_summary_en": "Current equal-rate cost reductions are listed per profile; multi-tenant evidence incomplete",
        "criterion_source_verified": True,
        "attainment": "not_demonstrated",
        "numeric_cost_results_by_profile": costs,
        "multi_tenant_utilization_verified": False,
        "reason": "保留已确认国际主流开源框架B0与可比条件，费用为历史公开报价同费率估算，不是本机账单。单服务高负载利用率不能自动证明多租户要求；未发明“较高利用率”的数值门槛。",
    }
    return {
        "schema": "project-final-goals-assessment/v1",
        "profile": "project-final",
        "goal_contract_id": profile["id"],
        "fully_specified": True,
        "acceptance_established": all(
            t["attainment"] == "demonstrated" for t in targets.values()
        ),
        "confirmed_interpretation": profile["confirmed_interpretation"],
        "criterion_source": profile["source"],
        "targets": targets,
        "limits": "已确认同工况参考和35B模型解释；最终目标单独展示，里程碑通过不等于最终验收通过。",
    }


def merge_requirements(review=None):
    """Report/source readiness is separate from achieving every project goal."""
    review = review or {}
    required = (
        "execution_complete",
        "raw_sources_verified",
        "quality_denominators_accounted",
    )
    blockers = [
        label
        for key, label in zip(
            required,
            (
                "本轮五数据集执行未完整结束",
                "本轮原始流和来源清单尚未核验",
                "本轮质量分母、失败与裁判未决数尚未完整列出",
            ),
        )
        if review.get(key) is not True
    ]
    return {
        "milestone_and_final_sections_required": True,
        "both_sections_present": True,
        "final_goal_sources_complete": True,
        "reference_interpretation_recorded": True,
        "model_scope_explanation_recorded": True,
        "runtime_and_quality_review_resolved": not blockers,
        "ready_to_merge": not blockers,
        "merge_scope": "结果报告及来源核验就绪，不代表项目最终目标全部达到",
        "project_final_attainment_is_independent": True,
        "blockers": blockers,
    }
