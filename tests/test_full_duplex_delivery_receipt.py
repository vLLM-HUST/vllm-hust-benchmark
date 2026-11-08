"""Ledger/receipt consistency only; never substitutes for server native evidence."""

from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_fixed_duplex_scope_preserves_categories_pairs_and_original_ids():
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/full-duplex-fixed27-native-delivery-20261107.json"
        ).read_text()
    )
    selected = receipt["selected_samples"]
    assert len(selected) == len(set(selected)) == 27
    categories = Counter(sample.rsplit("/", 1)[0] for sample in selected)
    assert len(categories) == 9 and set(categories.values()) == {3}
    assert "v1.0/icc_backchannel/0" in selected
    assert (
        sum(2 if sample.startswith("v1.5/") else 1 for sample in selected)
        == receipt["native_sessions"]
        == 39
    )
    assert receipt["input_handoff"]["selected_native_cases_read_back"] == 27
    assert receipt["scoring"]["local_judge_is_non_official"] is True
    assert receipt["scoring"]["official_gpt_judge_executed"] is False
    assert receipt["scoring"]["performance_claim"] is False
    assert receipt["evidence_reuse"]["fixtures_are_not_new_neural_execution"] is True
    assert receipt["deployment"]["whole_environment_pip_check_clean"] is False


def test_current_source_completion_tracks_delivery_audit_not_historical_preparation():
    path = (
        ROOT
        / "docs/workload-data/validation/full-duplex-fixed27-native-delivery-20261107.json"
    )
    receipt = json.loads(path.read_text())
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text()
    )
    item = ledger["delivery_inventory"]["FULL_DUPLEX_BENCH"]
    assert (
        item["source_delivery_audit_20261107"]["receipt"]
        == path.relative_to(ROOT).as_posix()
    )
    if receipt["source_checkbox_eligible"]:
        assert receipt["status"] == "END_TO_END_EXECUTED_AND_SCORED"
        assert receipt["software_gate"]["status"] == "PASSED"
        assert item["checked"] and item["status"] == "AVAILABLE"
    else:
        assert receipt["status"] == "AWAITING_FINAL_SOFTWARE_CHECK"
        assert not item["checked"] and item["status"] == "IN_PROGRESS"
