"""Validate the real published projection's scope, denominators and arithmetic offline."""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_all_five_paired_results_keep_original_denominators_and_scope():
    original = (
        ROOT / "reports/five-datasets-standard-20261113/summary-public-v2.json"
    ).read_bytes()
    assert (
        original
        == (
            ROOT
            / "website-publication/data/five-datasets-standard-20261113/summary-public-v2.json"
        ).read_bytes()
    )
    data = json.loads(original)
    assert data["execution_complete"] and not data["formal_acceptance_complete"]
    assert data["reused_stages"] == 15 and data["new_inference_stages"] == 0
    assert [d["id"] for d in data["datasets"]] == [
        "mmlu",
        "hle",
        "frontier",
        "swe",
        "terminal",
    ]
    for dataset, n in zip(data["datasets"], [200, 200, 160, 64, 48], strict=True):
        for arm in dataset["arms"].values():
            assert arm["planned"] == arm["recorded"] == arm["successful"] == n
            assert arm["failed"] == arm["unreturned"] == 0
            assert math.isclose(
                arm["successful_output_tokens"] / arm["duration_seconds"],
                arm["output_tokens_per_second"],
                rel_tol=1e-12,
            )
            assert arm["ttft_p50_ms"] <= arm["ttft_p95_ms"] <= arm["ttft_p99_ms"]
            assert arm["tpot_p50_ms"] <= arm["tpot_p95_ms"] <= arm["tpot_p99_ms"]
        if dataset["mode"] == "inference_only":
            assert dataset["quality"] is None
            assert all(
                a["input_at_least16384_cases"] == 0 for a in dataset["arms"].values()
            )
    assert data["formal_targets"]["3.4.3"]["accepted"] is False
    assert data["datasets"][1]["changes_percent"]["mean_ttft_ms"] > 0
