import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location(
        "remaining", ROOT / "scripts/audit_v15_remaining.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reanalysis_matches_originals_and_does_not_promote_acceptance():
    module = load()
    result = module.build()
    assert result == json.loads(module.OUT.read_text())
    assert not result["new_measurements_started"]
    assert result["m01"]["effective_flops"] is None
    assert result["m01"]["passed"] is None
    assert result["stability_passed"] is None
    arms = result["cost"]["arms"]
    assert 22.60 < result["cost"]["equal_fee_basis_resource_reduction_percent"] < 22.61
    for row in arms.values():
        assert row["successful_requests"] == 200
        assert row["output_tokens"] == 51200
        assert row["reserved_cards"] == 4
        assert row["monthly_fee_P"] is None
        assert row["cached_input_tokens"] is None
        assert set(row["utilization"]["per_card_percent"]) == {"4", "5", "6", "7"}
        for seconds in row["utilization"]["covered_seconds"].values():
            assert seconds == pytest.approx(row["interval_seconds"])
    for card in ("4", "5"):
        assert (
            arms["b1"]["utilization"]["per_card_percent"][card]
            < arms["b0"]["utilization"]["per_card_percent"][card]
        )
    full = [
        r
        for r in result["swe_continuous_records"]["windows"]
        if r["window_seconds"] == [0, 3600]
    ]
    assert [r["actual_context_ge16k_returns"] for r in full] == [2593, 1380, 2232]
    assert all(r["ttft_p95_seconds"] is None for r in full)
    assert result["mmlu_fixed_b0_completion"]["counts"]["truncated"] == 18
    assert len(result["quality_inventory_all_solutions"]) > 10


def test_nearest_rank_empty_and_tail():
    module = load()
    assert module.quantile([], 0.99) is None
    assert module.quantile([4, 1, 3, 2], 0.95) == 4


def test_v075_publication(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import build_midterm_v075 as builder

    result = json.loads(builder.OUTPUT.read_text())
    assert result["version"] == "V0.75"
    assert result["m01_method"]["current"]["passed"] is None
    section = next(s for s in result["sections"] if s["id"] == "v15-remaining")
    assert len(section["tables"][0]["rows"]) == 5
    assert "does not improve" in section["scope_en"]
    assert "22.601951%" in section["scope"]
    assert len(section["tables_en"]) == 3
    assert result["source_commit"] == builder.REVISION
