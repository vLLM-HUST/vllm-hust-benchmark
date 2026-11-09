"""Original publisher metadata coverage only; no downloaded conversations or images."""

import re

from vllm_hust_workload_data.registry import adapter_spec, source_spec


def test_complete_original_shard_metadata_is_available_without_data():
    spec = source_spec("VISIONARENA")
    assert len(spec["files"]) == 43
    assert sum(item["bytes"] for item in spec["files"]) == 84565530886
    assert {item["path"] for item in spec["files"]} == {
        f"data/train-{number:05d}-of-00043.parquet" for number in range(43)
    }
    for item in spec["files"]:
        assert item["role"] == "data" and re.fullmatch("[a-f0-9]{64}", item["sha256"])
        assert "/resolve/" + spec["revision"] + "/" + item["path"] in item["url"]


def test_original_custom_terms_not_misrepresented_as_open_redistribution():
    spec = source_spec("VISIONARENA")
    assert spec["access"]["agreement_acceptance_required"] is True
    assert spec["access"]["third_party_transfer_permitted"] is False
    assert "publisher_metadata_acquisition" not in spec
    assert "historical_acquisition_reuse" not in spec
    assert (
        adapter_spec("VISIONARENA")["status"]
        == "CONTRACT_MAPPED_ASSET_VALIDATION_PENDING"
    )
