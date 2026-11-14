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
ACCEPTANCE_POLICY = {
    "id": "single-fixed-profile-20261114-v2",
    "source": "user_confirmed_interpretation_after_partial_execution",
    "source_quote": "只要一个数据集达成目标就行。",
    "minimum_qualifying_profiles": 1,
    "joint_metrics_must_share_one_profile": True,
    "all_five_numeric_passes_required": False,
    "all_five_execution_and_result_disclosure_unchanged": True,
    "compute_reference_profile": "mmlu-fixed200-v1",
    "context_and_multi_tenant_requirements_unchanged": True,
}


def milestone_representative_targets(original, datasets):
    """Each numeric goal needs one whole qualifying profile, never mixed metrics."""
    targets = copy.deepcopy(original)
    performance, costs = [], []
    for d in datasets:
        change = d["changes_percent"]
        ratio = (
            d["arms"]["B1"]["output_tokens_per_second"]
            / d["arms"]["B0"]["output_tokens_per_second"]
        )
        performance.append(
            {
                "dataset": d["id"],
                "throughput_ratio": ratio,
                "mean_ttft_reduction_percent": -change["mean_ttft_ms"],
                "mean_tpot_reduction_percent": -change["mean_tpot_ms"],
                "numeric_threshold_met": ratio >= 1.10
                and -change["mean_ttft_ms"] >= 5
                and -change["mean_tpot_ms"] >= 5,
            }
        )
        gain = change.get("ai_core_utilization_gain_pp")
        costs.append(
            {
                "dataset": d["id"],
                "equal_rate_unit_cost_reduction_percent": -change[
                    "unit_cost_change_percent"
                ],
                "ai_core_utilization_gain_pp": gain,
                "numeric_threshold_met": -change["unit_cost_change_percent"] >= 10,
                "observed_utilization_increased": gain is not None and gain > 0,
            }
        )
    good_performance = [p["dataset"] for p in performance if p["numeric_threshold_met"]]
    good_cost = [
        p["dataset"]
        for p in costs
        if p["numeric_threshold_met"] and p["observed_utilization_increased"]
    ]
    for key, rows, qualifying in [
        ("3.4.2", performance, good_performance),
        ("3.4.4", costs, good_cost),
    ]:
        targets[key].update(
            accepted=bool(qualifying),
            numeric_threshold_met=bool(qualifying)
            if key == "3.4.2"
            else any(p["numeric_threshold_met"] for p in costs),
            status="单一固定数据集达标"
            if qualifying
            else "尚无单一固定数据集满足该项条件",
            accepted_scope="One complete fixed profile per target; all profile results disclosed"
            if qualifying
            else None,
            qualifying_profiles=qualifying,
            numeric_results_by_profile=rows,
            acceptance_policy_id=ACCEPTANCE_POLICY["id"],
            measured_value=None,
            display_value="达标：" + ", ".join(qualifying) if qualifying else "未达标",
            display_value_en="Met: " + ", ".join(qualifying)
            if qualifying
            else "Not met",
            reason="同一个固定数据集须同时满足该项目标的全部条件；符合条件："
            + (", ".join(qualifying) or "无")
            + "。五项结果与负收益均保留，不跨数据集拼接指标。成本为已披露的同费率估算，利用率增量单列，不另造‘明显改善’阈值。质量与专项稳定性另行报告。",
            reason_en="One fixed profile must jointly satisfy this target; qualifying profiles: "
            + (", ".join(qualifying) or "none")
            + ". All results and regressions remain visible. Never mix metrics across profiles. Cost uses the disclosed equal-rate estimate; utilization changes are reported without inventing a significance threshold. Quality and context stability remain separate.",
        )
        targets[key].pop("source_url", None)
    return targets


def unmeasured_current_compute_target(archived):
    """M01 uses one representative MMLU200 workload, independently of five-profile tests."""
    historical = copy.deepcopy(archived)
    return {
        "name": "有效算力效率 ≥50%",
        "status": "current_workload_reference_not_measured",
        "accepted": False,
        "accepted_scope": None,
        "measured_value": None,
        "display_value": "本轮未验证；归档工况另列",
        "display_value_en": "Current workload unverified; archive separate",
        "reason": "3.4.1按用户确认仅选固定MMLU200为代表工况，无需对HLE、Frontier、SWE或Terminal另做参考标定。当前MMLU200仍缺完整必要算量、独立标定、整链验证和双卡同步计时，不能据吞吐配对直接计算参考效率。历史MMLU200/out256的≥59.76%单列保留，不套用于当前完整回答配置。现场采集包的完整工况认证模块尚未齐备，不将分项算量或时间线冒充参考值。",
        "reason_en": "Target3.4.1 uses only fixed MMLU200 as the user-confirmed representative workload. HLE, Frontier, SWE and Terminal do not require separate reference calibration. Current MMLU200 still lacks complete necessary-work accounting, independent calibration, whole-chain validation and synchronized timing; throughput comparisons alone cannot establish reference efficiency. Archived MMLU200/out256 evidence remains separate from the current full-answer configuration; complete live workload certifiers are still required.",
        "calibration_scope": {
            "dataset": "mmlu",
            "profile": "fixed200-v1",
            "tasks": 200,
            "selection_basis": "user_confirmed_20261114",
            "current_workload": "frozen_mmlu_pair",
            "other_profiles_require_calibration": False,
            "generation_policy": "Retain the frozen MMLU pair configuration; historical out256 is a separate workload.",
        },
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
    qualifying = [p["dataset"] for p in performance if p["numeric_threshold_met"]]
    targets["3.4.2"] = {
        **criteria["3.4.2"],
        "requirement": "单一固定数据集：吞吐 ≥2×；平均TTFT/TPOT分别降低 ≥20%",
        "requirement_en": "One fixed profile: throughput ≥2×; mean TTFT/TPOT each reduced ≥20%",
        "evidence_summary": "符合全部数值门槛的固定数据集："
        + (", ".join(qualifying) or "无"),
        "evidence_summary_en": "Fixed profiles meeting all numeric targets: "
        + (", ".join(qualifying) or "none"),
        "criterion_source_verified": True,
        "attainment": "demonstrated" if qualifying else "not_demonstrated",
        "qualifying_profiles": qualifying,
        "nonqualifying_profiles": misses,
        "any_profile_numeric_threshold_met": bool(qualifying),
        "numeric_results_by_profile": performance,
        "all_profile_numeric_thresholds_met": all(
            p["numeric_threshold_met"] for p in performance
        ),
        "reason": "按用户确认，任一固定数据集在同一可比B0/B1工况同时满足吞吐、TTFT和TPOT全部数值门槛即可，不要求五项均达标。各项结果、负收益和未决数完整披露，不跨数据集拼接最好指标，不改写此前报告。数值达标不构成统计质量等价证明。",
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
        "qualifying_cost_profiles": [
            p["dataset"] for p in costs if p["numeric_threshold_met"]
        ],
        "any_profile_cost_threshold_met": any(
            p["numeric_threshold_met"] for p in costs
        ),
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
        "acceptance_policy": copy.deepcopy(ACCEPTANCE_POLICY),
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
