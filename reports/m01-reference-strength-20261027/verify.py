"""Keep empirical arithmetic separate from reference-strength acceptance."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def verify():
    s = json.loads((HERE / "summary.json").read_text())
    r = json.loads((HERE / "review.json").read_text())
    d = s["current_disposition"]
    assert d["historical_Cref"] == 7.108506511466548
    assert d["historical_B1_empirical_attainment_percent"] == 99.93970959931444
    assert d["historical_frozen_90_result"] is True
    assert (
        abs(
            d["same_window_B1_C"] / d["historical_Cref"] * 100
            - d["historical_B1_empirical_attainment_percent"]
        )
        < 1e-10
    )
    assert (
        d["arithmetic"] == "verified" and d["reference_strength"] == "not_established"
    )
    assert (
        d["current_strong_reference_acceptance"]
        is d["recommended_replacement_Cref"]
        is None
    )
    assert len(s["remaining_evidence_gaps"]) == 3
    assert s["remaining_evidence_gaps"] == r["three_remaining_gaps"]
    assert (
        s["shared_implementation_allowed"]
        and not s["different_algorithm_or_global_optimality_proof_required"]
    )
    assert (
        s["eligible_replacement_reference"] is None
        and not s["new_measurements_authorized"]
    )
    assert s["new_quality_windows"] == 0
    assert s["prospective_candidate_budget"] == r["budget"]
    assert not r["execution_authorized"]
    assert r["budget"]["wall_limit_seconds"] is None
    assert r["budget"]["new_measurements_authorized"] is False
    assert r["proposal_sha256"] == s["restricted_source_sha256"]["proposal"]
    return dict(
        status="passed",
        historical_empirical_attainment_percent=d[
            "historical_B1_empirical_attainment_percent"
        ],
        current_reference_strength_acceptance=None,
        new_measurements=0,
    )


if __name__ == "__main__":
    print(json.dumps(verify()))
