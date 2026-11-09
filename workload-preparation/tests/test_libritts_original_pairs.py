"""Tiny self-created format fixtures; not original speech or native evidence."""

import io
import tarfile
import wave

import pytest

from vllm_hust_workload_data.libritts import original_rows
from vllm_hust_workload_data.adapters import adapt_row, iter_source_rows
from vllm_hust_workload_data.registry import adapter_spec, source_spec


def archive(tmp_path, *, missing=None, bad_path=None, sample_rate=24000):
    identity = "123_456_000001_000002"
    base = "LibriTTS/test-other/123/456/" + identity
    audio = io.BytesIO()
    with wave.open(audio, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * 12)
    files = {
        base + ".wav": audio.getvalue(),
        base + ".original.txt": b"Dr. Test.\n",
        base + ".normalized.txt": b"Doctor Test.\n",
        "LibriTTS/test-other/123/456/123_456.trans.tsv": b"inert metadata fixture\n",
    }
    if missing:
        del files[base + missing]
    if bad_path:
        files[bad_path] = b"not executable"
    path = tmp_path / "test-other.tar.gz"
    with tarfile.open(path, "w:gz") as out:
        for name, data in files.items():
            item = tarfile.TarInfo(name)
            item.size = len(data)
            out.addfile(item, io.BytesIO(data))
    return path


def test_original_pair_fields_and_chapter_references_preserved(tmp_path):
    (row,) = original_rows(archive(tmp_path))
    assert row["text"] == "Dr. Test.\n"
    assert row["normalized_text"] == "Doctor Test.\n"
    assert row["id"] == "123_456_000001_000002"
    assert row["segment_id"] == "000001" and row["utterance_id"] == "000002"
    assert row["audio"]["archive_member"].endswith(row["id"] + ".wav")
    assert row["context_metadata_members"] == [
        "LibriTTS/test-other/123/456/123_456.trans.tsv"
    ]
    assert set(tmp_path.iterdir()) == {tmp_path / "test-other.tar.gz"}


@pytest.mark.parametrize("suffix", [".wav", ".original.txt", ".normalized.txt"])
def test_missing_original_pair_fails_without_partial_success(tmp_path, suffix):
    with pytest.raises(ValueError, match="unmatched"):
        list(original_rows(archive(tmp_path, missing=suffix)))


def test_unsafe_archive_member_rejected_without_extraction(tmp_path):
    with pytest.raises(ValueError, match="unsafe"):
        list(original_rows(archive(tmp_path, bad_path="LibriTTS/../outside")))


def test_wrong_sample_rate_not_silently_converted(tmp_path):
    with pytest.raises(ValueError, match="24kHz"):
        list(original_rows(archive(tmp_path, sample_rate=22050)))


def test_registered_fields_keep_original_audio_reference_side():
    spec = source_spec("LIBRITTS")
    assert spec["files"][0]["path"] == "test-other.tar.gz"
    adapter = adapter_spec("LIBRITTS")
    assert adapter["input_fields"] == ["text", "normalized_text"]
    assert adapter["outcome_fields"] == ["audio"]
    assert "context_metadata_members" in adapter["annotation_fields"]


def test_public_dispatch_preserves_task_reference_and_identity_roles(tmp_path):
    spec = adapter_spec("LIBRITTS")
    (row,) = iter_source_rows(archive(tmp_path), spec)
    record = adapt_row(
        "LIBRITTS", row, source={"source_id": "LIBRITTS"}, spec=spec, index=0
    )
    assert record.task["input"] == {
        "text": "Dr. Test.\n",
        "normalized_text": "Doctor Test.\n",
    }
    assert record.outcome["audio"]["archive_member"].endswith(row["id"] + ".wav")
    assert record.runtime["execution"] == "not_executed"
