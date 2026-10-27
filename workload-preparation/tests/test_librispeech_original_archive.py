import io
import tarfile

import pytest

from vllm_hust_workload_data.librispeech import (
    original_archive_rows,
    prepare_original_archives,
    prepared_audio_path,
    read_original_audio,
)
from vllm_hust_workload_data.assets import sha256
from vllm_hust_workload_data.adapters import adapt_row, iter_source_rows
from vllm_hust_workload_data.registry import adapter_spec


def archive_fixture(
    tmp_path,
    *,
    transcript="1-2-0001  ORIGINAL TEXT",
    audio=True,
    escape=False,
    split="test-clean",
):
    path = tmp_path / f"{split}.tar.gz"
    files = {f"LibriSpeech/{split}/1/2/1-2.trans.txt": (transcript + "\n").encode()}
    if audio:
        files[f"LibriSpeech/{split}/1/2/1-2-0001.flac"] = b"fixture-not-real-audio"
    if escape:
        files["../escape"] = b"x"
    with tarfile.open(path, "w:gz") as archive:
        for name, raw in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    return path


def test_original_transcript_and_audio_locator_not_a_chat_request(tmp_path):
    (row,) = original_archive_rows(archive_fixture(tmp_path))
    assert row["text"] == " ORIGINAL TEXT"
    assert row["id"] == "1-2-0001"
    assert row["audio"] == {
        "archive_file": "test-clean.tar.gz",
        "archive_member": "LibriSpeech/test-clean/1/2/1-2-0001.flac",
        "format": "flac",
    }
    assert row["transcript_line_index"] == 0
    assert "prompt" not in row and "messages" not in row


def test_archive_uses_existing_native_fields_without_reference_leak(tmp_path):
    spec = adapter_spec("LIBRISPEECH")
    (row,) = iter_source_rows(archive_fixture(tmp_path), spec)
    record = adapt_row(
        "LIBRISPEECH",
        row,
        source={"source_id": "LIBRISPEECH", "file_sha256": "fixture"},
        spec=spec,
        index=0,
    )
    assert record.task["input"] == {"audio": row["audio"]}
    assert record.outcome_record() == {"text": " ORIGINAL TEXT"}
    assert record.replay["unmapped_field_names"] == []
    assert record.replay["annotations"]["transcript_member"].endswith("1-2.trans.txt")
    assert "not_request_arrival" in record.replay["order"]
    with pytest.raises(ValueError, match="not a ready model request"):
        record.request_body()


def test_existing_hf_native_field_mapping_is_preserved():
    row = {
        "id": "original",
        "audio": {"path": "original.flac"},
        "text": "REFERENCE",
        "speaker_id": 1,
        "chapter_id": 2,
        "file": "original.flac",
    }
    record = adapt_row(
        "LIBRISPEECH",
        row,
        source={"source_id": "LIBRISPEECH", "file_sha256": "fixture"},
        spec=adapter_spec("LIBRISPEECH"),
        index=0,
    )
    assert record.task["input"]["audio"] == row["audio"]
    assert record.outcome_record() == {"text": "REFERENCE"}
    assert record.replay["unmapped_field_names"] == []


def test_original_audio_accessor_returns_exact_member_bytes(tmp_path):
    path = archive_fixture(tmp_path)
    spec = adapter_spec("LIBRISPEECH")
    (row,) = iter_source_rows(path, spec)
    record = adapt_row(
        "LIBRISPEECH",
        row,
        source={
            "source_id": "LIBRISPEECH",
            "file": path.name,
            "file_sha256": sha256(path),
        },
        spec=spec,
        index=0,
    )
    assert read_original_audio(record, data_root=tmp_path) == b"fixture-not-real-audio"
    path.write_bytes(b"changed original archive")
    with pytest.raises(ValueError, match="archive identity mismatch"):
        read_original_audio(record, data_root=tmp_path)


def test_unmatched_original_transcript_is_not_dropped(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        list(original_archive_rows(archive_fixture(tmp_path, audio=False)))


def test_prepare_preserves_bytes_and_checks_prepared_audio(tmp_path, monkeypatch):
    archives = tmp_path / "archives"
    archives.mkdir()
    paths = [
        archive_fixture(archives, split=split) for split in ("test-clean", "test-other")
    ]
    files = [
        dict(
            role="data",
            path=path.name,
            sha256=sha256(path),
            bytes=path.stat().st_size,
            rows=1,
        )
        for path in paths
    ]
    monkeypatch.setattr(
        "vllm_hust_workload_data.librispeech.source_spec", lambda _: {"files": files}
    )
    output = tmp_path / "prepared"
    result = prepare_original_archives(data_root=archives, output=output)
    assert result["original_utterances"] == 2
    assert result["native_asr_or_scoring_verified"] is False
    for path in paths:
        (row,) = original_archive_rows(path)
        record = adapt_row(
            "LIBRISPEECH",
            row,
            source={
                "source_id": "LIBRISPEECH",
                "file": path.name,
                "file_sha256": sha256(path),
            },
            spec=adapter_spec("LIBRISPEECH"),
            index=0,
        )
        prepared = prepared_audio_path(record, prepared_root=output)
        assert prepared.read_bytes() == b"fixture-not-real-audio"
        prepared.write_bytes(b"modified")
        with pytest.raises(ValueError, match="audio bytes changed"):
            prepared_audio_path(record, prepared_root=output)
    with pytest.raises(ValueError, match="fresh output"):
        prepare_original_archives(data_root=archives, output=output)
    with pytest.raises(ValueError, match="fresh output"):
        prepare_original_archives(data_root=archives, output=archives / "nested")


def test_archive_cannot_escape_or_invent_utterance(tmp_path):
    with pytest.raises(ValueError, match="escapes corpus root"):
        list(original_archive_rows(archive_fixture(tmp_path, escape=True)))
    with pytest.raises(ValueError, match="identity or content"):
        list(
            original_archive_rows(
                archive_fixture(tmp_path, transcript="3-4-0001 WRONG FOLDER")
            )
        )
