"""V5 custody must retain failed quality and increasing native launch counts."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

DATA = (
    Path(__file__).resolve().parents[1]
    / "reports/muxpd-followup-20261025/quality-reference-v5"
)
SPEC = importlib.util.spec_from_file_location("muxpd_v5", DATA / "verify_v5.py")
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def test_complete_saved_publication_custody():
    result = VERIFY.verify(DATA)
    assert result["preserved_prior_files"] == 400
    assert (result["terminal_runs"], result["capsules"]) == (7, 8)
    assert not result["quality_accepted"]
    assert not result["native_kernel_launch_reduction"]


def test_higher_total_score_cannot_relabel_failed_quality():
    claims = json.loads((DATA / "archive-index-v5.json").read_text())["claims"]
    claims["supplementary_quality_gate_passed"] = True
    with pytest.raises(ValueError, match="Unsupported acceptance"):
        VERIFY.validate_claims(claims)


def test_duplicate_rank_cannot_supply_four_rank_profile():
    data = json.loads((DATA / "analysis/mixed-profile/comparison-v1.json").read_text())
    data["comparisons"][3] = copy.deepcopy(data["comparisons"][2])
    with pytest.raises(ValueError, match="duplicated profile rank"):
        VERIFY.validate_profile(data)
