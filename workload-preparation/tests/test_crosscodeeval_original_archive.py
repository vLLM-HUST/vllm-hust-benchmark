"""Inert archive fixtures only, not original data/native code-completion validation."""

from io import BytesIO
import json
import tarfile

import pytest

from vllm_hust_workload_data.adapters import adapt_row
from vllm_hust_workload_data.crosscodeeval import original_rows
from vllm_hust_workload_data.registry import adapter_spec


def archive(tmp_path, rows):
    path = tmp_path / "fixture.tar.xz"
    with tarfile.open(path, "w:xz") as bundle:
        for name, row in rows:
            raw = (json.dumps(row) + "\n").encode()
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            bundle.addfile(member, BytesIO(raw))
    return path


def row():
    return {
        "prompt": "inert prefix",
        "crossfile_context": {
            "text": "inert retrieved context",
            "list": [{"chunk": "inert", "score": 1.0}],
        },
        "groundtruth": "inert reference",
        "right_context": "inert future context",
        "metadata": {
            "task_id": "original-task-id",
            "repository": "inert repository",
            "groundtruth_start_lineno": 8,
        },
    }


def test_all_original_fields_preserved_without_reference_leak(tmp_path):
    original = row()
    name = "python/line_completion_rg1_unixcoder_cosine_sim.jsonl"
    (loaded,) = original_rows(archive(tmp_path, [(name, original)]))
    record = adapt_row(
        "CROSSCODEEVAL",
        loaded,
        source={"source_id": "CROSSCODEEVAL", "file_sha256": "fixture"},
        spec=adapter_spec("CROSSCODEEVAL"),
        index=0,
    )
    assert record.task["input"] == {
        key: original[key] for key in ("prompt", "crossfile_context")
    }
    assert record.outcome == {
        key: original[key] for key in ("groundtruth", "right_context")
    }
    assert record.task["task_id"] == original["metadata"]["task_id"]
    assert record.replay["annotations"]["metadata"] == original["metadata"]
    assert record.replay["annotations"]["source_archive_member"] == name
    assert not record.replay["unmapped_field_names"]


def test_original_shared_task_id_and_distinct_profiles_are_retained(tmp_path):
    names = [
        "python/line_completion_rg1_bm25.jsonl",
        "python/line_completion_oracle_bm25.jsonl",
    ]
    loaded = list(original_rows(archive(tmp_path, [(name, row()) for name in names])))
    assert [item["source_archive_member"] for item in loaded] == names
    assert loaded[0]["source_task_id"] == loaded[1]["source_task_id"]
    assert loaded[0]["source_profile"] != loaded[1]["source_profile"]
    assert [item["source_member_row_index"] for item in loaded] == [0, 0]


def test_baseline_has_no_invented_crossfile_context(tmp_path):
    original = row()
    del original["crossfile_context"]
    (loaded,) = original_rows(
        archive(tmp_path, [("java/line_completion.jsonl", original)])
    )
    assert "crossfile_context" not in loaded


@pytest.mark.parametrize(
    "name",
    [
        "../escape.jsonl",
        "/escape.jsonl",
        "rust/line_completion.jsonl",
        "python/unknown.jsonl",
    ],
)
def test_invalid_original_member_is_rejected_not_rewritten(tmp_path, name):
    with pytest.raises(ValueError):
        list(original_rows(archive(tmp_path, [(name, row())])))


def test_provenance_field_collision_not_silently_overwritten(tmp_path):
    original = {**row(), "source_profile": "unexpected input field"}
    with pytest.raises(ValueError, match="collision"):
        list(
            original_rows(
                archive(tmp_path, [("python/line_completion.jsonl", original)])
            )
        )


def test_symlink_member_rejected_without_following_target(tmp_path):
    path = tmp_path / "symlink.tar.xz"
    with tarfile.open(path, "w:xz") as bundle:
        member = tarfile.TarInfo("python/line_completion.jsonl")
        member.type = tarfile.SYMTYPE
        member.linkname = "/unrelated/private/path"
        bundle.addfile(member)
    with pytest.raises(ValueError, match="nonregular"):
        list(original_rows(path))
