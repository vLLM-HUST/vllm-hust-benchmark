"""Tiny inert paper packages; no original science task or model execution."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import sqlite3

import pytest

from vllm_hust_workload_data import iter_workload
from vllm_hust_workload_data.consumer import resolve_input
from vllm_hust_workload_data.paperbench import asset_role, original_paper_packages
from vllm_hust_workload_data.pipeline import run
from vllm_hust_workload_data.registry import adapter_spec, source_spec


def fixture(tmp_path):
    root = tmp_path / "original"
    prefix = "project/paperbench/data/papers/fixture-paper/"
    content = {
        "config.yaml": b"id: fixture-paper\ntitle: Inert test fixture\nunknown_field: preserve-me\n",
        "paper.md": b"Original paper fixture, not a generated model request.",
        "paper.pdf": b"%PDF-inert-fixture",
        "addendum.md": b"Original fixture instructions",
        "blacklist.txt": b"example.invalid/original-reference\n",
        "assets/input.bin": b"original fixture material",
        "rubric.json": b'{"hidden":"RUBRIC_SENTINEL"}',
        "judge.addendum.md": b"JUDGE_SENTINEL",
        "judge/data.bin": b"HIDDEN_DATA_SENTINEL",
    }
    files = []
    for name, raw in content.items():
        path = root / (prefix + name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        files.append(
            {
                "path": prefix + name,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "role": asset_role(prefix + name),
            }
        )
    source = {
        "schema_version": "source-file-manifest/v1",
        "source_id": "PAPERBENCH",
        "repo_id": "openai/frontier-evals",
        "revision": adapter_spec("PAPERBENCH")["code_revision"],
        "files": files,
        "unit": "scientific_paper_package",
        "provenance_kind": "inert_test_fixture",
    }
    catalog = tmp_path / "catalog"
    (catalog / "data/catalogs").mkdir(parents=True)
    (catalog / "data/catalogs/source_manifests_v1.json").write_text(
        json.dumps({"sources": {"PAPERBENCH": source}}), encoding="utf-8"
    )
    (catalog / "data/catalogs/adapter_registry_v1.json").write_text(
        json.dumps({"adapters": {"PAPERBENCH": adapter_spec("PAPERBENCH")}}),
        encoding="utf-8",
    )
    return root, source, catalog


def test_original_input_and_hidden_evaluation_separation(tmp_path):
    root, manifest, catalog = fixture(tmp_path)
    records = list(iter_workload("PAPERBENCH", data_root=root, catalog_root=catalog))
    assert len(records) == 1
    record = records[0]
    assert record.task["task_id"] == "fixture-paper"
    assert "unknown_field: preserve-me" in record.task["input"]["original_config_yaml"]
    inputs = json.dumps(record.input_record())
    for marker in [
        "RUBRIC_SENTINEL",
        "JUDGE_SENTINEL",
        "HIDDEN_DATA_SENTINEL",
        "/rubric.json",
        "/judge/",
    ]:
        assert marker not in inputs
    assert len(record.outcome["evaluation_assets"]) == 3
    assert record.runtime["execution"] == "not_executed"
    with pytest.raises(ValueError, match="not a ready model request"):
        record.request_body()
    assert len(manifest["files"]) == 9


def test_pipeline_full_graph_roundtrip_without_payload_index(tmp_path):
    root, manifest, catalog = fixture(tmp_path)
    prepared = tmp_path / "prepared"
    report = run(
        {"sources": [{"source_id": "PAPERBENCH", "data_root": str(root)}]},
        output=prepared,
        catalog_root=catalog,
    )
    assert report["status"] == "PREPARED_REFERENCES_ONLY"
    database = prepared / "workload.sqlite"
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    connection = sqlite3.connect(database)
    row = connection.execute("SELECT record_id,task_json FROM records").fetchone()
    spec = json.loads(
        connection.execute("SELECT adapter_json FROM sources").fetchone()[0]
    )
    connection.close()
    assert len(spec["original_package_source"]["files"]) == len(manifest["files"])
    assert "unknown_field" not in row[
        1
    ] and "RUBRIC_SENTINEL" not in database.read_bytes().decode("latin1")
    restored = resolve_input(database, row[0], data_root=root)
    expected = next(iter_workload("PAPERBENCH", data_root=root, catalog_root=catalog))
    assert asdict(restored) == asdict(expected)
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("hidden", [False, True])
def test_related_material_corruption_rejected(tmp_path, hidden):
    root, manifest, _ = fixture(tmp_path)
    name = "rubric.json" if hidden else "paper.md"
    item = next(x for x in manifest["files"] if x["path"].endswith("/" + name))
    (root / item["path"]).write_bytes(b"corruption")
    with pytest.raises(ValueError, match="asset identity mismatch"):
        list(original_paper_packages(root, manifest=manifest))


def test_mutation_detected_on_generator_close(tmp_path):
    root, manifest, _ = fixture(tmp_path)
    rows = original_paper_packages(root, manifest=manifest)
    next(rows)
    item = next(x for x in manifest["files"] if x["path"].endswith("/judge/data.bin"))
    (root / item["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="asset identity mismatch"):
        rows.close()


def test_hidden_asset_cannot_be_relabelled_as_input(tmp_path):
    root, manifest, _ = fixture(tmp_path)
    changed = deepcopy(manifest)
    next(x for x in changed["files"] if x["path"].endswith("/rubric.json"))["role"] = (
        "problem_asset"
    )
    with pytest.raises(ValueError, match="roles required"):
        list(original_paper_packages(root, manifest=changed))


def test_missing_required_material_rejected(tmp_path):
    root, manifest, _ = fixture(tmp_path)
    manifest["files"] = [
        x for x in manifest["files"] if not x["path"].endswith("/rubric.json")
    ]
    with pytest.raises(ValueError, match="complete original"):
        list(original_paper_packages(root, manifest=manifest))


@pytest.mark.parametrize(
    "path",
    [
        "../escape",
        "project/paperbench/data/papers/x/../paper.md",
        "project/paperbench/data/papers/x/unknown.txt",
    ],
)
def test_unreviewed_paths_rejected(path):
    with pytest.raises(ValueError):
        asset_role(path)


def test_registered_source_uses_same_single_asset_catalog():
    source = source_spec("PAPERBENCH")
    assert len(source["files"]) == 506
    assert sum(x["role"] == "data" for x in source["files"]) == 23
    assert sum(x["role"] == "evaluation_asset" for x in source["files"]) == 79
    assert sum(x["bytes"] for x in source["files"]) == 213434104
