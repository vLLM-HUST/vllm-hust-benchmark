import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest


ROOT = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019/m01-recovery/hccs64-r1"
)


@pytest.fixture(scope="module")
def evidence():
    metadata = json.loads((ROOT / "archive.json").read_text())
    chunks = []
    for part in metadata["parts"]:
        payload = (ROOT / part["file"]).read_bytes()
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
