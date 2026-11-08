"""Invented archive fixtures test custody, not real task execution."""

import hashlib
import zipfile

import pytest

from vllm_hust_workload_data.miniwob import (
    PREFIX,
    adapt_case,
    original_rows,
    read_native_definition,
)


def fixture(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(
            PREFIX + "click-dialog.html", b"<html>FIXTURE_REWARD_CODE</html>"
        )
        bundle.writestr(PREFIX + "multi-layouts.html", b"<html>second fixture</html>")
        bundle.writestr("shared/core.js", b"fixture core")
    source = {
        "source_id": "MINIWOB_PLUSPLUS",
        "file": archive.name,
        "file_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    return archive, source


def test_definitions_are_references_not_fabricated_instructions(tmp_path):
    archive, source = fixture(tmp_path)
    rows = list(original_rows(archive))
    assert [r["task_name"] for r in rows] == [
        "miniwob.click-dialog",
        "miniwob.multi-layouts",
    ]
    record = adapt_case(rows[0], source=source, index=0)
    assert record.task["input"] == {"task_name": "miniwob.click-dialog"}
    assert record.runtime["kind"] == "native_environment"
    assert "FIXTURE_REWARD_CODE" not in str(record.input_record())
    assert (
        read_native_definition(record, data_root=tmp_path)
        == b"<html>FIXTURE_REWARD_CODE</html>"
    )
    assert record.outcome is None


def test_changed_archive_fails_before_native_definition_read(tmp_path):
    archive, source = fixture(tmp_path)
    record = adapt_case(next(original_rows(archive)), source=source, index=0)
    with archive.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError):
        read_native_definition(record, data_root=tmp_path)


@pytest.mark.parametrize("unsafe", ["../escape.html", "/absolute.html", "bad\\name"])
def test_unsafe_member_references_fail(tmp_path, unsafe):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(unsafe, "fixture")
    if "\\" in unsafe:
        # Windows ZipInfo normalizes separators on write; forge the actual header.
        archive.write_bytes(archive.read_bytes().replace(b"bad/name", b"bad\\name"))
    with pytest.raises(ValueError, match="unsafe"):
        list(original_rows(archive))
