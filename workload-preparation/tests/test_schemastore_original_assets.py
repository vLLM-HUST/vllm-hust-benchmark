"""Fabricated archive fixtures, not original SchemaStore validation evidence."""

import hashlib
import json
import sqlite3
from zipfile import ZipFile

import pytest

from vllm_hust_workload_data import iter_workload
from vllm_hust_workload_data.consumer import resolve_input
from vllm_hust_workload_data.pipeline import run
from vllm_hust_workload_data.schemastore import (
    original_schema_rows,
    read_original_fixture,
)


def fixture(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    archive = data / "fixture.zip"
    original = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {"value": {"type": "integer"}},
    }
    with ZipFile(archive, "w") as stream:
        stream.writestr("original/src/schemas/json/a.json", json.dumps(original))
        stream.writestr("original/src/schemas/json/b.json", "true")
        stream.writestr("original/src/test/a/example.yaml", "value: 1\n")
        stream.writestr(
            "original/src/negative_test/a/example.json", '{"value": "wrong"}'
        )
    raw = archive.read_bytes()
    source = {
        "source_id": "SCHEMASTORE",
        "revision": "software-fixture",
        "unit": "schema_assets",
        "provenance_kind": "schema_collection",
        "files": [
            {
                "path": archive.name,
                "role": "data",
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "rows": 2,
            }
        ],
    }
    return data, source, original


def test_original_schema_fixture_bytes_and_index_resolve_without_prompts(tmp_path):
    data, source, original = fixture(tmp_path)
    records = list(iter_workload("SCHEMASTORE", data_root=data, manifest=source))
    assert len(records) == 2 and records[0].task["input"] == {"schema": original}
    assert records[1].task["input"] == {"schema": True}
    assert records[0].runtime["asset_role"] == "schema_collection_not_model_request"
    assert "positive_examples" not in json.dumps(records[0].input_record())
    outcomes = records[0].outcome_record()
    assert (
        read_original_fixture(data, records[0].source, outcomes["positive_examples"][0])
        == b"value: 1\n"
    )
    assert (
        read_original_fixture(data, records[0].source, outcomes["negative_examples"][0])
        == b'{"value": "wrong"}'
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(source), "utf-8")
    output = tmp_path / "index"
    run(
        {
            "sources": [
                {
                    "source_id": "SCHEMASTORE",
                    "data_root": str(data),
                    "manifest": str(manifest),
                }
            ]
        },
        output=output,
    )
    with sqlite3.connect(output / "workload.sqlite") as connection:
        ids = connection.execute(
            "SELECT record_id FROM records ORDER BY rowid"
        ).fetchall()
    for (identity,), expected in zip(ids, records, strict=True):
        actual = resolve_input(output / "workload.sqlite", identity, data_root=data)
        assert actual.input_record() == expected.input_record()
        assert actual.outcome_record() == expected.outcome_record()
    with pytest.raises(ValueError, match="not a ready model request"):
        records[0].request_body()


def test_fixture_checksum_tamper_is_rejected(tmp_path):
    data, source, _ = fixture(tmp_path)
    record = next(iter_workload("SCHEMASTORE", data_root=data, manifest=source))
    reference = record.outcome_record()["positive_examples"][0]
    reference["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fixture identity"):
        read_original_fixture(data, record.source, reference)


@pytest.mark.parametrize("member", ["original/../escape.json", "/outside.json"])
def test_archive_escape_is_rejected(tmp_path, member):
    archive = tmp_path / "invalid.zip"
    with ZipFile(archive, "w") as stream:
        stream.writestr(member, "{}")
    with pytest.raises(ValueError):
        list(original_schema_rows(archive))
