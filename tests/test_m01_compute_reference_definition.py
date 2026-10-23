"""Check the dimensional link without creating experimental acceptance results."""

import json
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_reference_compute_definition_and_pending_results():
    method = json.loads(
        (ROOT / "reports/vllm-hust-midterm-20261019/m01-method-v527.json").read_text()
    )
    assert method["effective_compute_formula"] == "C = F / (k * T * 10^12)"
    assert method["reference_effective_compute_formula"] == (
        "C_reference = F / (k * T_reference * 10^12)"
    )
    assert method["efficiency_formula"] == (
        "eta = C / C_reference * 100% = T_reference / T * 100%"
    )
    assert method["threshold_percent"] == 50
    assert method["equivalent_threshold"] == "C >= 0.5 * C_reference"
    assert all(value is None for value in method["current"].values())


def test_same_work_and_card_count_cancel_exactly():
    # Synthetic arithmetic examples, not measurements or reference calibration.
    for work in (10**12, 3 * 10**15):
        for cards in (1, 2, 4):
            for elapsed in (40, 50, 100):
                reference_time = 25
                compute = Fraction(work, cards * elapsed * 10**12)
                reference = Fraction(work, cards * reference_time * 10**12)
                efficiency = compute / reference * 100
                assert efficiency == Fraction(reference_time, elapsed) * 100
                assert (efficiency >= 50) == (compute >= reference / 2)
