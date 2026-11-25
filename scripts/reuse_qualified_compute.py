"""Append the qualified representative MMLU200/out256 reference to a sealed report."""

import argparse
from copy import deepcopy
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path

SOURCE_SHA256 = "95c6bd465c4beb558cd25284e221a166e33634ea2f9b6d44364d2b1114863bd0"  # pragma: allowlist secret -- fixed public source digest
SOURCE_COMMIT = "53541118b3461705f1e61532710ff0a759d59af0"  # pragma: allowlist secret -- fixed public revision


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def correct(report, qualified):
    if not qualified["evidence_qualified"] or qualified["chips"] != 2:
        raise ValueError("Unqualified reference")
    b1 = next(a for a in qualified["arms"] if a["role"] == "B1")
    repeats = b1["repetitions"]
    if len(repeats) != 3 or any(
        r["requests"] != 200
        or r["output_tokens"] != 51200
        or r["synchronized_ranks"] != 2
        or not r["uninstrumented"]
        for r in repeats
    ):
        raise ValueError("Reference differs from fixed200/out256 repeated timing")
    with localcontext() as ctx:
        ctx.prec = 40
        eta = (
            100
            * sum(Decimal(r["Tref_lower_seconds"]) for r in repeats)
            / sum(Decimal(r["T_seconds"]) for r in repeats)
        )
        if abs(eta - Decimal(b1["aggregate"]["eta_percent_lower"])) > Decimal("1e-30"):
            raise ValueError("Reference efficiency recomputation differs")
    d = deepcopy(report)
    historical = deepcopy(report["formal_targets"]["3.4.1"]["historical_evidence"])
    with localcontext() as ctx:
        ctx.prec = 40
        reference_seconds = sum(Decimal(r["Tref_lower_seconds"]) for r in repeats)
        reference_upper = Decimal(sum(r["F_interval"][1] for r in repeats)) / (
            2 * reference_seconds * 10**12
        )
    target = {
        "name": "有效算力效率 ≥50%",
        "accepted": False,
        "status": "reference_available_current_numerator_not_measured",
        "display_value": "参考分母已复用；本轮分子待测",
        "display_value_en": "Reference denominator reused; current numerator pending",
        "measured_value": None,
        "accepted_scope": None,
        "reference_tflops_per_card_upper": str(reference_upper),
        "reference_statistics_path": "reference-denominator-v1.json",
        "reference_interpretation": "同工况独立参考",
        "reference_available": True,
        "reference_statistics_reused": True,
        "historical_evidence": historical,
        "source_url": historical["source_url"],
        "source_summary_sha256": SOURCE_SHA256,
        "reason": "已有参考分母统计可复用；本轮效率由匹配工作域的必要算量和同步计时重新计算。旧59.7573%作为历史结果保留。",
        "reason_en": "Reuse the existing reference denominator statistics. Recompute efficiency from current matched-workload arithmetic and synchronized timing; retain59.7573% only as a historical result.",
        "calibration_scope": {
            "dataset": "mmlu",
            "profile": "fixed200-v1",
            "tasks": 200,
            "max_completion_tokens": 256,
            "repetitions_per_arm": 3,
            "frontend": "native_inproc",
        },
    }
    d["formal_targets"]["3.4.1"] = target
    d["milestone_assessment"]["targets"]["3.4.1"] = deepcopy(target)
    final = d["project_final_assessment"]["targets"]["3.4.1"]
    final.update(
        evidence_summary="原MMLU200/out256效率下界≥59.76%；本轮效率需重算",
        evidence_summary_en="Historical MMLU200/out256 lower bound≥59.76%; current efficiency requires recomputation",
    )
    d["report_revision_date"] = "2026-11-15"
    d["reference_reuse_correction"] = {
        "schema": "qualified-reference-reuse-correction/v1",
        "source_benchmark_commit": SOURCE_COMMIT,
        "source_summary_sha256": SOURCE_SHA256,
        "new_inference_executed": False,
        "reason": "Reuse denominator statistics rather than an old efficiency score; distinguish available calibration from pending current numerator measurements.",
    }
    return d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--qualified", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--denominator", type=Path)
    a = parser.parse_args()
    if digest(a.qualified) != SOURCE_SHA256:
        raise ValueError("Qualified reference source SHA256 changed")
    if a.out.exists():
        raise FileExistsError(a.out)
    d = correct(json.loads(a.report.read_bytes()), json.loads(a.qualified.read_bytes()))
    denominator = a.denominator or a.report.parent / "reference-denominator-v1.json"
    cache = json.loads(denominator.read_bytes())
    if cache["benchmark_commit"] != SOURCE_COMMIT:
        raise ValueError("Reference denominator source differs")
    qualified = json.loads(a.qualified.read_bytes())
    with localcontext() as ctx:
        ctx.prec = 40
        for arm in qualified["arms"]:
            expected = Decimal(sum(r["F_interval"][1] for r in arm["repetitions"])) / (
                2
                * sum(Decimal(r["Tref_lower_seconds"]) for r in arm["repetitions"])
                * 10**12
            )
            if (
                Decimal(cache["arms"][arm["role"]]["reference_tflops_per_card_upper"])
                != expected
            ):
                raise ValueError("Reference denominator does not recompute")
    d["compute_reference_statistics"] = cache
    d["reference_reuse_correction"]["denominator_statistics_sha256"] = digest(
        denominator
    )
    d["reference_reuse_correction"]["prior_summary_sha256"] = digest(a.report)
    a.out.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
