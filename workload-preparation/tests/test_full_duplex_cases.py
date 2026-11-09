"""Inert original-shape ZIP fixtures, not real audio or native execution."""

import hashlib
import json
import zipfile
from copy import deepcopy
from dataclasses import asdict

import pytest

from vllm_hust_workload_data.full_duplex import original_cases
from vllm_hust_workload_data import iter_workload
from vllm_hust_workload_data.bindings import bind_source
from vllm_hust_workload_data.consumer import resolve_input
from vllm_hust_workload_data.pipeline import run, prepared_rows
from vllm_hust_workload_data.registry import source_spec, adapter_spec


def fixture(tmp_path, *, missing=None, extra_unsafe=None):
    path = tmp_path / "v1.5/user_backchannel.zip"
    path.parent.mkdir(parents=True)
    with zipfile.ZipFile(path, "w") as archive:
        for identity in ["20", "3"]:
            prefix = f"user_backchannel/{identity}/"
            files = {
                "metadata.json": json.dumps(
                    {
                        "context_text": "ORIGINAL_CONTEXT_SENTINEL",
                        "backchannel_text": "REFERENCE_SENTINEL",
                        "timestamps": [1.5, 2.25],
                        "unknown_original": {"nested": [1, "preserved"]},
                    }
                ),
                "input.json": '{"chunks":[],"text":"TRANSCRIPT_SENTINEL"}',
                "clean_input.json": '{"chunks":[],"text":"CONTROL_SENTINEL"}',
                "input.wav": b"INERT_NOT_REAL_WAV",
                "clean_input.wav": b"INERT_CONTROL",
                "context.wav": b"INERT_CONTEXT",
                "backchannel.wav": b"INERT_COMPONENT",
                "extra.json": '{"unknown":"EXTRA_REFERENCE_SENTINEL"}',
            }
            for name, content in files.items():
                if name != missing:
                    archive.writestr(prefix + name, content)
        archive.writestr("user_backchannel/aggregate.json", '{"original":"global"}')
        archive.writestr("__MACOSX/._user_backchannel", b"packaging metadata")
        if extra_unsafe:
            archive.writestr(extra_unsafe, b"inert")
    raw = path.read_bytes()
    return path, {
        "path": "v1.5/user_backchannel.zip",
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "role": "data",
        "rows": 2,
    }


def test_original_order_annotations_and_control_separation(tmp_path):
    path, asset = fixture(tmp_path)
    rows = list(original_cases(path, asset))
    assert [x["original_case_id"] for x in rows] == ["20", "3"]
    assert rows[0]["task_id"] == "v1.5/user_backchannel/20"
    inputs = json.dumps(rows[0]["input"])
    assert set(rows[0]["input"]) == {"audio_stream"}
    for marker in ["SENTINEL", "clean_input.wav", "metadata.json", "timestamps"]:
        assert marker not in inputs
    outcome = rows[0]["outcome"]
    assert outcome["original_annotations"]["metadata.json"]["unknown_original"] == {
        "nested": [1, "preserved"]
    }
    assert outcome["original_annotations"]["metadata.json"]["backchannel_text"]
    assert "current_turn_text" not in outcome["original_annotations"]["metadata.json"]
    assert (
        outcome["related_media"]["clean_input.wav"]["native_role"]
        == "paired_control_input"
    )
    assert len(outcome["related_media"]) == 3
    assert outcome["archive_level_assets"][0]["archive_member"].endswith(
        "aggregate.json"
    )
    assert rows[0]["native_execution_complete"] is False
    with zipfile.ZipFile(path) as archive:
        for name, original in outcome["original_annotations"].items():
            assert original == json.loads(archive.read("user_backchannel/20/" + name))


def test_missing_case_annotation_rejected(tmp_path):
    path, asset = fixture(tmp_path, missing="metadata.json")
    with pytest.raises(ValueError, match="complete original case"):
        list(original_cases(path, asset))


def test_observed_count_not_silently_replaced_by_doc_count(tmp_path):
    path, asset = fixture(tmp_path)
    with pytest.raises(ValueError, match="case count"):
        list(original_cases(path, dict(asset, rows=99)))


def test_unsafe_member_rejected(tmp_path):
    path, asset = fixture(tmp_path, extra_unsafe="../escape.json")
    with pytest.raises(ValueError, match="unsafe"):
        list(original_cases(path, asset))


def test_mutation_after_yield_detected_on_close(tmp_path):
    path, asset = fixture(tmp_path)
    rows = original_cases(path, asset)
    next(rows)
    with path.open("ab") as stream:
        stream.write(b"mutation")
    with pytest.raises(ValueError):
        rows.close()


def test_common_provider_binding_index_and_readback(tmp_path):
    data_root = tmp_path / "original"
    path, asset = fixture(data_root)
    source = deepcopy(source_spec("FULL_DUPLEX_BENCH"))
    source["files"] = [asset]
    catalog = tmp_path / "catalog"
    directory = catalog / "data/catalogs"
    directory.mkdir(parents=True)
    (directory / "source_manifests_v1.json").write_text(
        json.dumps({"sources": {"FULL_DUPLEX_BENCH": source}}), encoding="utf-8"
    )
    (directory / "adapter_registry_v1.json").write_text(
        json.dumps(
            {"adapters": {"FULL_DUPLEX_BENCH": adapter_spec("FULL_DUPLEX_BENCH")}}
        ),
        encoding="utf-8",
    )
    expected = list(
        iter_workload("FULL_DUPLEX_BENCH", data_root=data_root, catalog_root=catalog)
    )
    assert len(expected) == 2
    assert expected[0].runtime["execution"] == "not_executed"
    with pytest.raises(ValueError):
        expected[0].request_body()
    bound = bind_source(
        "FULL_DUPLEX_BENCH",
        data_root=data_root,
        files=[asset["path"]],
        revision=source["revision"],
        catalog_root=catalog,
    )
    assert bound["files"][0]["rows"] == 2
    with pytest.raises(ValueError, match="registered audio archives"):
        bind_source(
            "FULL_DUPLEX_BENCH",
            data_root=data_root,
            files=[asset["path"]],
            revision="unreviewed-revision",
            catalog_root=catalog,
        )
    output = tmp_path / "prepared"
    run(
        {"sources": [{"source_id": "FULL_DUPLEX_BENCH", "data_root": str(data_root)}]},
        output=output,
        catalog_root=catalog,
    )
    database = output / "workload.sqlite"
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    assert b"SENTINEL" not in database.read_bytes()
    rows = list(prepared_rows(database))
    assert len(rows) == 2
    for row, original in zip(rows, expected, strict=True):
        assert asdict(
            resolve_input(database, row["record_id"], data_root=data_root)
        ) == asdict(original)
    assert before == hashlib.sha256(database.read_bytes()).hexdigest()
    assert path.read_bytes().startswith(b"PK")


def test_registered_original_observed_counts_and_identity():
    source = source_spec("FULL_DUPLEX_BENCH")
    assert len(source["files"]) == 9
    assert sum(x["rows"] for x in source["files"]) == 1225
    assert sum(x["bytes"] for x in source["files"]) == 705372348
    assert source["license"] is None
