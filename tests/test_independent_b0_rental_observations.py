"""Check historical B0 unit costs against the sealed inputs used for publication."""

from hashlib import sha256
import json
from pathlib import Path

import pytest


PUBLICATION = Path(__file__).resolve().parents[1] / "website-publication"


def read(path):
    return json.loads((PUBLICATION / path).read_bytes())


def verify(descriptor):
    raw = (PUBLICATION / descriptor["path"]).read_bytes()
    assert len(raw) == descriptor["bytes"]
    assert sha256(raw).hexdigest() == descriptor["sha256"]
    return json.loads(raw)


def test_independent_b0_cost_and_sampling_use_the_same_frozen_windows():
    data = read("data/b0-independent-rental-observations-v1.json")
    assert data["inference_reruns"] == 0
    replay = verify(data["sse_replay"])
    official = verify(data["rental_quote_source"])
    assert data["rental_quote"] == official["targets"]["4.4"]["rental_quote"]
    rate = float(data["rental_quote"]["cny_per_card_month"])
    assert set(data["baselines"]) == set(replay["baselines"])

    for name, evidence in data["baselines"].items():
        baseline = verify(evidence["observations"])
        assert baseline["baseline_id"] == name and baseline["pairs"] is None
        rows = {row["id"]: row for row in baseline["profiles"]}
        workloads = evidence["workloads"]
        assert set(workloads) == {"mmlu", "real-context-multi-tenant-formal"}

        for workload, sample in workloads.items():
            row = rows[workload]
            arm = row["arm"]
            assert sample["measurement_seal_sha256"] == arm["source_seal_sha256"]
            assert sample["source_metric"] == row["source_metric"]
            assert sample["duration_seconds"] == arm["duration_seconds"]
            assert sample["successful_output_tokens"] == arm["output_tokens"]
            utilization = sample["utilization"]
            assert utilization["missing_seconds"] == 0
            assert utilization["covered_seconds"] == pytest.approx(
                arm["duration_seconds"]
            )
            assert len(sample["physical_cards"]) == 2
            assert (
                sorted(map(int, utilization["per_card_percent"]))
                == sample["physical_cards"]
            )
            assert utilization["mean_percent"] == pytest.approx(
                sum(utilization["per_card_percent"].values()) / 2
            )
            expected = (
                1e6
                * 2
                * rate
                * arm["duration_seconds"]
                / (30 * 86400 * arm["output_tokens"])
            )
            assert sample[
                "estimated_cny_per_million_successful_output_tokens"
            ] == pytest.approx(expected)

        mmlu = replay["baselines"][name]["stages"]["mmlu"]
        assert (
            workloads["mmlu"]["device_samples_sha256"] == mmlu["device_samples_sha256"]
        )
        assert workloads["mmlu"]["physical_cards"] == mmlu["physical_cards"]
        tenant = "real-context-multi-tenant-formal"
        assert workloads[tenant]["utilization"] == {
            key: rows[tenant]["utilization"][key]
            for key in workloads[tenant]["utilization"]
        }
