"""Pinned asset inventory only; no raw capsule, Agent or native claim."""

from vllm_hust_workload_data.catalog import download_plan
from vllm_hust_workload_data.registry import source_spec


def test_original_metadata_and_capsules_are_distinct_fixed_assets():
    source = source_spec("BIXBENCH")
    assert (
        source["revision"]
        == "f8cc3bdcc6357c88b8c3648306522b9c422dc95a"  # pragma: allowlist secret - public dataset revision
    )
    assert source["repo_id"] == "futurehouse/BixBench"
    assert len(source["files"]) == 65
    assert len({item["path"] for item in source["files"]}) == 65
    metadata = [item for item in source["files"] if item["role"] == "data"]
    capsules = [
        item for item in source["files"] if item["role"] == "native_environment"
    ]
    assert len(metadata) == 1 and metadata[0]["path"] == "BixBench.jsonl"
    assert metadata[0]["bytes"] == 285842
    assert len(capsules) == 64
    assert sum(item["bytes"] for item in source["files"]) == 5909995227
    assert all(item["path"].endswith(".zip") for item in capsules)
    assert all(len(item["sha256"]) == 64 for item in source["files"])
    assert all(
        "/resolve/" + source["revision"] + "/" in item["url"]
        for item in source["files"]
    )


def test_download_plan_does_not_establish_native_execution():
    plan = download_plan("BIXBENCH")
    assert len(plan["files"]) == 65
    assert plan["access"]["gated"] is False
    assert "not" in source_spec("BIXBENCH")["boundary"].lower()
