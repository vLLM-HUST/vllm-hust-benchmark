"""Tiny archive fixtures, not original speech or native synthesis evidence."""

import io
import json
import tarfile

import pytest

from vllm_hust_workload_data.adapters import adapt_row, iter_source_rows
from vllm_hust_workload_data.registry import adapter_spec, source_spec
from vllm_hust_workload_data.bindings import bind_source
from vllm_hust_workload_data.consumer import resolve_input
from vllm_hust_workload_data.pipeline import prepared_rows, run


def archive_fixture(tmp_path, *, metadata=None, extra=None):
    path = tmp_path / "LJSpeech-1.1.tar.bz2"
    members = [
        (
            "LJSpeech-1.1/metadata.csv",
            metadata
            or b"LJ001-0002|raw 2|normalized two\nLJ001-0001|raw 1|normalized one\n",
        ),
        ("LJSpeech-1.1/wavs/LJ001-0001.wav", b"fixture not real WAV"),
        ("LJSpeech-1.1/wavs/LJ001-0002.wav", b"fixture not real WAV"),
    ]
    if extra:
        members.append(extra)
    with tarfile.open(path, "w:bz2") as archive:
        for name, value in members:
            member = tarfile.TarInfo(name)
            member.size = len(value)
            archive.addfile(member, io.BytesIO(value))
    return path


def test_original_csv_order_text_and_reference_separation(tmp_path):
    path = archive_fixture(tmp_path)
    spec = adapter_spec("LJSPEECH")
    rows = list(iter_source_rows(path, spec))
    assert [row["id"] for row in rows] == ["LJ001-0002", "LJ001-0001"]
    record = adapt_row(
        "LJSPEECH", rows[0], source={"source_id": "LJSPEECH"}, spec=spec, index=0
    )
    assert record.task["input"] == {
        "text": "raw 2",
        "normalized_text": "normalized two",
    }
    assert record.outcome["audio"]["archive_member"].endswith("LJ001-0002.wav")
    assert record.runtime["execution"] == "not_executed"
    assert source_spec("LJSPEECH")["files"][0]["rows"] == 13100


@pytest.mark.parametrize(
    "metadata",
    [
        b"LJ001-0001|text|normalized\n",  # unpaired audio
        b"LJ001-0001|text|normalized\nLJ001-0001|text|normalized\n",
        b"LJ001-0003|text|normalized\n",  # missing audio
        b"LJ001-0001|text\n",  # missing original column
    ],
)
def test_broken_original_relations_are_not_filtered(tmp_path, metadata):
    path = archive_fixture(tmp_path, metadata=metadata)
    with pytest.raises(ValueError):
        list(iter_source_rows(path, adapter_spec("LJSPEECH")))


@pytest.mark.parametrize(
    "name", ["../escape", "/escape", "LJSpeech-1.1/wavs/LJ001-0001.wav"]
)
def test_unsafe_or_duplicate_members_are_rejected(tmp_path, name):
    path = archive_fixture(tmp_path, extra=(name, b"fixture"))
    with pytest.raises(ValueError):
        list(iter_source_rows(path, adapter_spec("LJSPEECH")))


def test_binding_index_and_consumer_keep_original_text_and_reference(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    archive_fixture(root)
    manifest = bind_source(
        "LJSPEECH",
        data_root=root,
        files=["LJSpeech-1.1.tar.bz2"],
        revision="structural-fixture-not-publisher-data",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "prepared"
    report = run(
        {
            "sources": [
                {
                    "source_id": "LJSPEECH",
                    "data_root": str(root),
                    "manifest": str(manifest_path),
                }
            ]
        },
        output=output,
    )
    assert report["native_tasks_run"] == 0
    indexed = list(prepared_rows(output / "workload.sqlite"))
    assert len(indexed) == 2
    restored = resolve_input(
        output / "workload.sqlite", indexed[0]["record_id"], data_root=root
    )
    assert restored.task["task_id"] == "LJ001-0002"
    assert restored.task["input"] == {
        "text": "raw 2",
        "normalized_text": "normalized two",
    }
    assert "audio" not in restored.task["input"]
    assert restored.outcome["audio"]["archive_member"].endswith("LJ001-0002.wav")
