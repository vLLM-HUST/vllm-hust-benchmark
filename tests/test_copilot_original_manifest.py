"""Small invented integrity fixtures, not workload data or native validation."""

from copy import deepcopy

import pytest

from vllm_hust_workload_data.copilot import qualify_day_manifest


def fixture():
    manifest = {
        "complete": True,
        "date": "2026-06-01",
        "n_sessions": 3,
        "n_shards": 2,
        "total_bytes": 30,
        "shards": [
            {"shard_idx": 0, "n_sessions": 2, "size_bytes": 10},
            {"shard_idx": 1, "n_sessions": 1, "size_bytes": 20},
        ],
    }
    members = [
        {"path": "date=2026-06-01/shard-0000.jsonl.gz", "bytes": 10, "rows": 2},
        {"path": "date=2026-06-01/shard-0001.jsonl.gz", "bytes": 20, "rows": 1},
    ]
    return manifest, members


def test_complete_day_order_independent_and_no_mutation():
    manifest, members = fixture()
    before = deepcopy((manifest, members))
    result = qualify_day_manifest(manifest, members[::-1])
    assert result == {
        "date": "2026-06-01",
        "shards": 2,
        "sessions": 3,
        "compressed_shard_bytes": 30,
    }
    assert (manifest, members) == before


@pytest.mark.parametrize(
    "change", ["missing", "extra", "duplicate", "rows", "bytes", "bool"]
)
def test_reject_incomplete_or_modified_original_members(change):
    manifest, members = fixture()
    if change == "missing":
        members.pop()
    elif change == "extra":
        members.append({"path": "../shard-0002.jsonl.gz", "bytes": 0, "rows": 0})
    elif change == "duplicate":
        members.append(deepcopy(members[0]))
    elif change == "bool":
        members[0]["rows"] = True
    else:
        members[0][change] += 1
    with pytest.raises(ValueError):
        qualify_day_manifest(manifest, members)


@pytest.mark.parametrize("change", ["incomplete", "date", "total", "duplicate", "bool"])
def test_reject_inconsistent_publisher_manifest(change):
    manifest, members = fixture()
    if change == "incomplete":
        manifest["complete"] = False
    elif change == "date":
        manifest["date"] = "2026-6-1"
    elif change == "total":
        manifest["total_bytes"] += 1
    elif change == "duplicate":
        manifest["shards"][1] = deepcopy(manifest["shards"][0])
    else:
        manifest["shards"][0]["shard_idx"] = False
    with pytest.raises(ValueError):
        qualify_day_manifest(manifest, members)
