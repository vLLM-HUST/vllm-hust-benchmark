"""Synthetic archive fixtures validate preparation only, not native QReCC tasks."""

import hashlib
import json
import zipfile

import pytest

from vllm_hust_workload_data import provider


def test_download_plan_separates_archive_and_original_member():
    from vllm_hust_workload_data.catalog import download_plan

    plan = download_plan("QRECC")
    archives = [row for row in plan["files"] if row["role"] == "data_archive"]
    members = [row for row in plan["files"] if row["role"] == "data"]
    assert len(archives) == len(members) == 1
    assert archives[0]["sha256"] == provider.QRECC_ARCHIVE_SHA256
    assert "/1d174464290790c0198b474ad022952ecd83b76b/" in archives[0]["url"]
    assert members[0]["sha256"] == provider.QRECC_TEST_SHA256
    assert members[0]["path"] == members[0]["archive_member"] == "qrecc_test.json"
    assert members[0]["rows"] == 16451
    assert "url" not in members[0]  # No zip URL advertised as the JSON member bytes.
    assert plan["preparation_command"].startswith("workload-data prepare-qrecc-test ")


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
