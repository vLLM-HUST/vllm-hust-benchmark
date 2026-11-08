"""Original release metadata only; no native retrieval or generated answer claims."""

import re

from vllm_hust_workload_data.registry import source_spec


def test_original_four_way_same_tag_release_not_replaced_with_questions_only():
    spec = source_spec("DRAGON_PUBLIC")
    assert spec["publisher_metadata_acquisition"]["release_tag"] == "1.15.0"
    assert len(spec["files"]) == 4
    assert sum(row["bytes"] for row in spec["files"]) == 2277123
    assert {row["path"].split("/")[0] for row in spec["files"]} == {
        "public_questions",
        "public_texts",
        "reference_qa",
        "reference_texts",
    }
    assert sum(row["role"] == "data" for row in spec["files"]) == 1
    for row in spec["files"]:
        assert re.fullmatch("[a-f0-9]{64}", row["sha256"])
        assert row["publisher_repo"] in row["url"]
        assert row["publisher_revision"] in row["url"]
        assert row["original_path"] in row["url"]
    assert spec["access"]["gated"] is False
    assert "not per-document update/delete" in spec["boundary"]
