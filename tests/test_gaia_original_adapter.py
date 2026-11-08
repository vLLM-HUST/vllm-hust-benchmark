"""Invented task/attachment controls; no gated GAIA content is bundled."""

from copy import deepcopy
import hashlib
import json
import sqlite3

import pytest

from vllm_hust_workload_data import gaia
from vllm_hust_workload_data.adapters import adapt_row, iter_source_rows
from vllm_hust_workload_data.registry import adapter_spec, source_spec
from vllm_hust_workload_data.consumer import resolve_input


@pytest.fixture
def data(tmp_path, monkeypatch):
    row = {
        "task_id": "invented-fixture-id",
        "Question": "Read the fixture attachment, then use tools as needed.",
        "Level": "2",
        "Final answer": "fixture-only-oracle-answer",
        "file_name": "fixture.txt",
        "file_path": "2023/validation/fixture.txt",
        "Annotator Metadata": {
            "Steps": "fixture-only-oracle-plan",
            "Tools": "fixture-only-tool-label",
        },
    }
    files = []
    for name in sorted(gaia.METADATA | {row["file_path"]}):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"invented fixture, not original data")
        files.append(
            {
                "path": name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "role": "data" if name in gaia.METADATA else "attachment",
            }
        )
    manifest = {"source_id": "GAIA", "revision": "invented-fixture", "files": files}
    canonical = deepcopy(manifest)
    monkeypatch.setattr(gaia, "source_spec", lambda sid: deepcopy(canonical))
    spec = gaia.relation_spec(manifest, adapter_spec("GAIA"))
    monkeypatch.setattr(gaia, "iter_rows", lambda p: iter([deepcopy(row)]))
    return tmp_path, row, manifest, spec


def record(data):
    root, _, _, spec = data
    original = next(iter_source_rows(root / "2023/validation/metadata.parquet", spec))
    return adapt_row(
        "GAIA",
        original,
        source={"source_id": "GAIA", "file": "2023/validation/metadata.parquet"},
        spec=spec,
        index=0,
    )


def test_full_original_fields_roundtrip_and_oracle_separation(data):
    value = record(data)
    assert gaia.restore_original_case(value) == data[1]
    inputs = json.dumps(value.input_record())
    assert (
        "fixture-only-oracle" not in inputs and "fixture-only-tool-label" not in inputs
    )
    assert value.task["input"]["question"] == data[1]["Question"]
    assert value.replay["annotations"] == {"level": "2", "split": "validation"}
    with pytest.raises(ValueError, match="not a ready model request"):
        value.request_body()


def test_no_attachment_is_not_filtered_or_replaced(data):
    data[1].update(file_name="", file_path="")
    value = record(data)
    assert value.task["input"]["attachment_asset"] is None
    assert gaia.restore_original_case(value) == data[1]


@pytest.mark.parametrize(
    "change", ["changed-bytes", "missing", "unbound", "path-traversal", "backslash"]
)
def test_changed_missing_or_redirected_original_attachment_is_rejected(data, change):
    root, row, _, _ = data
    if change == "changed-bytes":
        (root / row["file_path"]).write_bytes(b"changed fixture")
    elif change == "missing":
        (root / row["file_path"]).unlink()
    elif change == "unbound":
        row["file_path"] = "2023/validation/another-file.txt"
    elif change == "path-traversal":
        row["file_name"] = "../fixture.txt"
    else:
        row["file_name"] = "..\\fixture.txt"
    with pytest.raises((ValueError, FileNotFoundError)):
        record(data)


def test_test_references_are_not_claimed_as_locally_scoreable(data):
    root, row, manifest, _ = data
    row.update(file_name="", file_path="", **{"Final answer": "?"})
    spec = gaia.relation_spec(manifest, adapter_spec("GAIA"))
    original = next(gaia.original_rows(root / "2023/test/metadata.parquet", spec))
    value = gaia.adapt_case(original, source={"source_id": "GAIA"}, index=0)
    assert value.outcome["reference_availability"] == "withheld_test"
    assert gaia.restore_original_case(value) == row


def test_duplicate_ids_and_unknown_fields_do_not_silently_drop_tasks(data, monkeypatch):
    root, row, _, spec = data
    monkeypatch.setattr(
        gaia, "iter_rows", lambda p: iter([deepcopy(row), deepcopy(row)])
    )
    with pytest.raises(ValueError, match="unique ID"):
        list(gaia.original_rows(root / "2023/validation/metadata.parquet", spec))
    row["unknown_original_field"] = "fixture"
    with pytest.raises(ValueError, match="fields/types"):
        record(data)


def test_manifest_keeps_full_metadata_and_no_duplicate_level_tasks(data):
    manifest = deepcopy(data[2])
    manifest["files"][0]["role"] = "reference_view"
    with pytest.raises(ValueError, match="full validation/test"):
        gaia.relation_spec(manifest, adapter_spec("GAIA"))
    source = source_spec("GAIA")
    assert len(source["files"]) == 119
    assert sum(x["bytes"] for x in source["files"]) == 110175514
    assert {x["path"] for x in source["files"] if x["role"] == "data"} == gaia.METADATA
    assert sum(x["role"] == "reference_view" for x in source["files"]) == 6


def test_digests_required_for_attachment_binding(data):
    manifest = deepcopy(data[2])
    manifest["files"][0].pop("sha256")
    with pytest.raises(ValueError, match="digests"):
        gaia.relation_spec(manifest, adapter_spec("GAIA"))


def test_prepared_asset_map_cannot_redirect_reader_to_modified_payload(data):
    root, _, _, spec = data
    modified = deepcopy(spec)
    modified["gaia_original_assets"]["2023/validation/metadata.parquet"]["sha256"] = (
        "0" * 64
    )
    with pytest.raises(ValueError, match="stored GAIA asset identities"):
        list(gaia.original_rows(root / "2023/validation/metadata.parquet", modified))


@pytest.mark.parametrize("wrong", ["revision", "mapping"])
def test_prepared_database_cannot_masquerade_as_original_gaia(tmp_path, wrong):
    source = {"source_id": "GAIA", "revision": source_spec("GAIA")["revision"]}
    spec = {"mapping": "gaia_original_tasks"}
    if wrong == "revision":
        source["revision"] = "invented-qbi-private-view"
    else:
        spec["mapping"] = "invented-qbi-evidence"
    db = tmp_path / "fixture.sqlite"
    with sqlite3.connect(db) as connection:
        connection.executescript(
            "CREATE TABLE sources(source_id TEXT,adapter_json TEXT); CREATE TABLE records(record_id TEXT,source_id TEXT,source_json TEXT,task_json TEXT);"
        )
        connection.execute(
            "INSERT INTO sources VALUES(?,?)", ("GAIA", json.dumps(spec))
        )
        connection.execute(
            "INSERT INTO records VALUES(?,?,?,?)",
            ("fixture", "GAIA", json.dumps(source), "{}"),
        )
    with pytest.raises(ValueError, match="pinned original GAIA"):
        resolve_input(db, "fixture", data_root=tmp_path)


@pytest.mark.parametrize("change", ["revision", "digest", "extra-asset"])
def test_changed_manifest_cannot_disguise_qbi_or_invented_inputs(data, change):
    manifest = deepcopy(data[2])
    if change == "revision":
        manifest["revision"] = "qbi-invented-substitute"
    elif change == "digest":
        manifest["files"][0]["sha256"] = "0" * 64
    else:
        manifest["files"].append(
            {"path": "invented.txt", "sha256": "0" * 64, "role": "attachment"}
        )
    with pytest.raises(ValueError, match="pinned original release"):
        gaia.relation_spec(manifest, adapter_spec("GAIA"))
