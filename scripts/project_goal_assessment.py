"""Assess project final goals using the user's confirmed reference/model interpretation."""

from decimal import Decimal
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
    targets["3.4.2"] = {
        **criteria["3.4.2"],
        "requirement": "吞吐 ≥2×；平均TTFT/TPOT分别降低 ≥20%",
        "requirement_en": "Throughput ≥2×; mean TTFT/TPOT each reduced ≥20%",
        "evidence_summary": "按五项逐一核算；HLE吞吐/TTFT不满足，质量仍在复核",
        "evidence_summary_en": "Per-profile numbers; HLE throughput/TTFT miss; quality under review",
        "criterion_source_verified": True,
        "attainment": "not_demonstrated",
        "numeric_results_by_profile": performance,
        "all_profile_numeric_thresholds_met": all(
            p["numeric_threshold_met"] for p in performance
        ),
        "reason": "沿用既定可比B0/B1与统一质量/服务要求。MMLU和两项trace等部分负载数值达到最终吞吐/时延门槛，但不能用这些通过覆盖HLE退化或未解决的质量/Schema问题；不拼接不同负载的最优值。",
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
        "evidence_summary": "MMLU同费率估算约降低50.24%；多租户证据未完成",
        "evidence_summary_en": "MMLU equal-rate estimate about−50.24%; multi-tenant evidence incomplete",
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


def merge_requirements():
    return {
        "milestone_and_final_sections_required": True,
        "both_sections_present": True,
        "final_goal_sources_complete": True,
        "reference_interpretation_recorded": True,
        "model_scope_explanation_recorded": True,
        "runtime_and_quality_review_resolved": False,
        "ready_to_merge": False,
        "blockers": [
            "MMLU质量优势未复现、HLE首批延迟和裁判未决问题仍在复核；4例开发诊断不替代正式验收。"
        ],
    }
