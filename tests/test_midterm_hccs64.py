import hashlib
import io
import json
from pathlib import Path
import statistics
import struct
import tarfile

import pytest


ROOT = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019/m01-recovery/hccs64-r1"
)


def read_archive(directory):
    metadata = json.loads((directory / "archive.json").read_text())
    chunks = []
    for part in metadata["parts"]:
        payload = (directory / part["file"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == part["sha256"]
        chunks.append(payload)
    payload = b"".join(chunks)
    assert hashlib.sha256(payload).hexdigest() == metadata["concatenated_sha256"]
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        members = {
            name: archive.extractfile(name).read() for name in metadata["members"]
        }
    for name, digest in metadata["members"].items():
        assert hashlib.sha256(members[name]).hexdigest() == digest
    return members


@pytest.fixture(scope="module")
def evidence():
    return read_archive(ROOT)


@pytest.fixture(scope="module")
def capacity():
    return read_archive(ROOT.parent / "hccs-capacity-r1")


def test_qualification_is_flits_not_an_invented_byte_conversion(evidence):
    qualification = json.loads(
        evidence["hccs64-known-payload-control-r2/qualification.json"]
    )
    assert qualification["status"] == "raw_flit_counter_qualified"
    assert qualification["unit"] == "raw_flits"
    assert qualification["byte_conversion"] is None
    for phase in qualification["phases"]:
        assert phase["peer_exact"]
        assert all(card["stable_boundaries"] for card in phase["cards"])


def test_model_deltas_recompute_from_raw_counters(evidence):
    summary = json.loads(evidence["microflow-hccs64-control-r1/summary.json"])
    rows = [
        json.loads(line)
        for line in evidence["microflow-hccs64-control-r1/counters.jsonl"].splitlines()
    ]
    for device in summary["devices"]:
        samples = [row for row in rows if row["card"] == device["card"]]

        def timestamp(row):
            return (row["before"]["wall_ns"] + row["after"]["wall_ns"]) // 2 / 1e9

        before = next(r for r in samples if timestamp(r) == device["start_wall"])
        after = next(r for r in samples if timestamp(r) == device["end_wall"])
        for direction in ("tx", "rx"):
            measured = [
                b - a
                for a, b in zip(
                    before[f"{direction}_counts"],
                    after[f"{direction}_counts"],
                    strict=True,
                )
            ]
            assert measured == device[f"{direction}_flits"]
    assert summary["physical_bytes"] is None
    assert summary["effective_flops"] is None
    assert summary["attainment_percent"] is None


def test_workload_finished_and_released_all_cards(evidence):
    receipt = json.loads(evidence["microflow-hccs64-r1/receipt.json"])
    result = json.loads(evidence["microflow-hccs64-r1/mmlu-pro.json"])
    assert receipt["status"] == "measured"
    assert receipt["all_devices_released"]
    assert result["completed"] == 200
    assert result["failed"] == 0
    assert result["total_input_tokens"] == 282265
    assert result["total_output_tokens"] == 51200


def test_recovered_gold_is_not_claimed_as_new_quality_measurement(evidence):
    gold = json.loads(evidence["source/recovered-mmlu-gold.json"])
    cases = gold["cases"]
    assert len(cases) == len({case["case_id"] for case in cases}) == 200
    assert all(case["gold"] in tuple("ABCDEFGHIJ") for case in cases)
    assert "not new inference" in gold["scope"]


def test_capacity_rates_recompute_from_counter_boundaries(capacity):
    summary = json.loads(capacity["hccs-capacity-control-r1/summary.json"])
    samples = [
        json.loads(line)
        for line in capacity["hccs-capacity-control-r1/counters.jsonl"].splitlines()
    ]
    ranks = [
        json.loads(capacity[f"hccs-capacity-r1/rank-{r}.json"])["phases"]
        for r in (0, 1)
    ]
    for index, phase in enumerate(summary["phases"]):
        duration = max(
            rank[index]["end_monotonic"] - rank[index]["start_monotonic"]
            for rank in ranks
        )
        assert duration == phase["duration_seconds"]
        assert all(rank[index]["numerics_passed"] for rank in ranks)
        for device in phase["devices"]:
            rows = [row for row in samples if row["card"] == device["card"]]

            def midpoint(row):
                return (row["before"]["wall_ns"] + row["after"]["wall_ns"]) / 2e9

            before = next(
                r for r in rows if midpoint(r) == device["counter_start_wall"]
            )
            after = next(r for r in rows if midpoint(r) == device["counter_end_wall"])
            for direction in ("tx", "rx"):
                delta = [
                    y - x
                    for x, y in zip(
                        before[f"{direction}_counts"],
                        after[f"{direction}_counts"],
                        strict=True,
                    )
                ]
                assert delta == device[f"{direction}_flits"]
                assert sum(delta) / duration == device[f"{direction}_flits_per_second"]
        assert phase["peer_exact"]
    for kind, median in summary["median_aggregate_tx_flits_per_second"].items():
        assert median == statistics.median(
            p["aggregate_tx_flits_per_second"]
            for p in summary["phases"]
            if p["kind"] == kind
        )
    receipt = json.loads(capacity["hccs-capacity-control-r1/receipt.json"])
    assert receipt["probe_exit_code"] == 0 and receipt["all_devices_released"]
    assert not receipt["sampler_errors"]


def test_driver_unit_constants_are_binary_gib_not_decimal_gb(capacity):
    constants = capacity[
        "hccs-capacity-control-r1/dcmi-bandwidth-constants.txt"
    ].decode()
    assert "00000000 00408f40 00000000 0000103e" in constants
    assert struct.unpack("<dd", bytes.fromhex("0000000000408f40000000000000103e")) == (
        1000,
        2**-30,
    )
    disassembly = capacity[
        "hccs-capacity-control-r1/dcmi-bandwidth-disassembly.txt"
    ].decode()
    assert "fmov\td5, #2.000000000000000000e+01" in disassembly
    assert "fmul\td1, d1, d5" in disassembly
    assert "fmul\td1, d1, d6" in disassembly


def test_driver_byte_projection_never_double_counts_peer_rx(evidence, capacity):
    raw = json.loads(evidence["microflow-hccs64-control-r1/summary.json"])
    projected = json.loads(
        capacity["hccs-capacity-control-r1/model-link-accounting.json"]
    )
    assert projected["aggregate_driver_accounted_tx_bytes"] == sum(
        sum(device["tx_flits"]) * 20 for device in raw["devices"]
    )
    assert projected["aggregate_driver_accounted_tx_bytes"] == 414523195180
    assert raw["physical_bytes"] is None
    assert projected["model_effective_flops"] is None
    assert projected["model_attainment_percent"] is None


def test_sampled_getter_crosscheck_recomputes_without_filling_gaps(capacity):
    summary = json.loads(capacity["hccs-capacity-control-r1/summary.json"])
    samples = [
        json.loads(line)
        for line in capacity["hccs-capacity-control-r1/bandwidth.jsonl"].splitlines()
    ]
    ranks = [
        json.loads(capacity[f"hccs-capacity-r1/rank-{r}.json"])["phases"]
        for r in (0, 1)
    ]
    for index, phase in enumerate(summary["phases"]):
        start = min(rank[index]["start_wall"] for rank in ranks)
        end = max(rank[index]["end_wall"] for rank in ranks)
        assert phase["aggregate_tx_flits_per_second"] == sum(
            device["tx_flits_per_second"] for device in phase["devices"]
        )
        for device in phase["devices"]:
            windows = [
                row
                for row in samples
                if row["card"] == device["card"]
                and row["before"]["wall_ns"] / 1e9 >= start + 0.5
                and row["after"]["wall_ns"] / 1e9 <= end - 0.5
            ]
            durations = [
                (row["after"]["monotonic_ns"] - row["before"]["monotonic_ns"]) / 1e9
                for row in windows
            ]
            assert sum(durations) == pytest.approx(
                device["observed_seconds"], rel=0, abs=1e-12
            )
            assert 0 < sum(durations) < device["interior_seconds"]
            for direction in ("tx", "rx"):
                mean = sum(
                    row[f"total_{direction}_gbs"] * duration
                    for row, duration in zip(windows, durations, strict=True)
                ) / sum(durations)
                assert mean == pytest.approx(
                    device["sampled_time_weighted_reported_bandwidth"][direction]
                )
                ratio = (
                    mean
                    * 2**30
                    / (
                        sum(device[f"{direction}_flits"])
                        * 20
                        / phase["duration_seconds"]
                    )
                )
                assert ratio == pytest.approx(
                    device["getter_gibps_to_flit20_rate_ratio"][direction]
                )
