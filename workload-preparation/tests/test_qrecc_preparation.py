"""Synthetic archive fixtures validate preparation only, not native QReCC tasks."""

import hashlib
import json
import zipfile

import pytest

from vllm_hust_workload_data import provider


def test_reject_unknown_archive_without_creating_output(tmp_path):
    archive = tmp_path / "unknown.zip"
    archive.write_bytes(b"not original data")
    with pytest.raises(ValueError, match="archive identity"):
        provider.prepare_qrecc_test(data_archive=archive, output=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_prepare_preserves_member_and_never_overwrites(tmp_path, monkeypatch):
    raw = json.dumps([{"fixture": True}] * 16451).encode()
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("qrecc_test.json", raw)
        bundle.writestr("unselected.json", "[]")
    monkeypatch.setattr(
        provider,
        "QRECC_ARCHIVE_SHA256",
        hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(provider, "QRECC_TEST_SHA256", hashlib.sha256(raw).hexdigest())
    output = tmp_path / "prepared"
    result = provider.prepare_qrecc_test(data_archive=archive, output=output)
    assert (output / "qrecc_test.json").read_bytes() == raw
    assert not (output / "unselected.json").exists()
    assert result["model_calls"] == result["scorer_runs"] == 0
    assert result["retrieval_corpus_prepared"] is False
    with pytest.raises(FileExistsError):
        provider.prepare_qrecc_test(data_archive=archive, output=output)
