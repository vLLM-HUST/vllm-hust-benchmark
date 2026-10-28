"""Offline package fixtures; not scientific execution or original task results."""

import hashlib
import json

import pytest

from vllm_hust_workload_data.naturebench import original_cpu_packages
from vllm_hust_workload_data.registry import read_catalog


def package(tmp_path):
    metadata = {
        "task_name": "fixture",
        "workflow_topology": "fixture",
        "methodology_paradigm": "fixture",
        "tooling_metadata": None,
        "domain_metadata": {"domain_tags": ["fixture"]},
        "compute_resource_requirements": {"cpu_compute": True},
        "performance_entries": [{"nested": {"reference": 42}}],
    }
    manifest = {
        "schema_version": "source-file-manifest/v1",
        "source_id": "NATUREBENCH",
        "repo_id": "fixture",
        "revision": "code-fixture",
        "dataset_revision": "data-fixture",
        "files": [],
    }
    for name, content, role in (
        ("metadata.json", json.dumps(metadata).encode(), "data"),
        ("problem/README.md", b"original\r\nproblem\r\n", "problem_asset"),
        ("problem/data_description.md", b"description", "problem_asset"),
        ("problem/data/metadata.json", b'{"not": "a task"}', "problem_asset"),
        ("evaluation/ground_truth.csv", b"hidden-answer", "evaluation_asset"),
        ("environment/Dockerfile.v3", b"FROM fixture", "environment_asset"),
        ("licenses/ATTRIBUTION.txt", b"fixture", "license_asset"),
    ):
        relative = "tasks/fixture/" + name
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        manifest["files"].append(
            {
                "path": relative,
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "role": role,
            }
        )
    return manifest, metadata


def test_original_relations_and_hidden_outcomes(tmp_path):
    manifest, metadata = package(tmp_path)
    rows = list(original_cpu_packages(tmp_path, manifest=manifest))
    assert len(rows) == 1
    row = rows[0]
    assert row["input"]["problem_texts"]["README.md"] == "original\r\nproblem\r\n"
    assert row["outcome"]["original_task_metadata"] == metadata
    assert row["source"]["code_revision"] != row["source"]["dataset_revision"]
    assert "hidden-answer" not in json.dumps(row["input"])
    assert "evaluation/" not in json.dumps(row["input"])
    assert "performance_entries" not in json.dumps(row["input"])
    assert not row["native_execution_complete"]


def test_hidden_asset_tamper_is_not_ignored(tmp_path):
    manifest, _ = package(tmp_path)
    (tmp_path / "tasks/fixture/evaluation/ground_truth.csv").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity mismatch"):
        list(original_cpu_packages(tmp_path, manifest=manifest))


def test_close_rechecks_original_assets(tmp_path):
    manifest, _ = package(tmp_path)
    rows = original_cpu_packages(tmp_path, manifest=manifest)
    next(rows)
    (tmp_path / "tasks/fixture/problem/README.md").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity mismatch"):
        rows.close()


def test_nested_metadata_cannot_become_task(tmp_path):
    manifest, _ = package(tmp_path)
    manifest["files"][3]["role"] = "data"
    with pytest.raises(ValueError, match="role"):
        list(original_cpu_packages(tmp_path, manifest=manifest))


def test_cannot_drop_hidden_evaluation_from_package_manifest(tmp_path):
    manifest, _ = package(tmp_path)
    manifest["files"] = [
        item for item in manifest["files"] if item["role"] != "evaluation_asset"
    ]
    with pytest.raises(ValueError, match="complete problem"):
        list(original_cpu_packages(tmp_path, manifest=manifest))


def test_packaged_inventory_has_three_tasks_not_seven_metadata_files():
    manifest = read_catalog("naturebench_original_cpu_assets_v1.json")
    assert len(manifest["files"]) == 576
    assert sum(item["bytes"] for item in manifest["files"]) == 2754532128
    assert sum(item["role"] == "data" for item in manifest["files"]) == 3
    assert (
        sum(item["path"].endswith("/metadata.json") for item in manifest["files"]) == 7
    )


def catalog(tmp_path, manifest):
    directory = tmp_path / "catalog" / "data" / "catalogs"
    directory.mkdir(parents=True)
    source = {**manifest, "unit": "source_rows", "provenance_kind": "fixture"}
    adapter = read_catalog("adapter_extensions_v1.json")["adapters"]["NATUREBENCH"]
    adapter = {
        **adapter,
        "code_revision": manifest["revision"],
        "dataset_revision": manifest["dataset_revision"],
    }
    for filename, content in (
        ("source_manifests_v1.json", {"sources": {"NATUREBENCH": source}}),
        ("adapter_registry_v1.json", {"adapters": {"NATUREBENCH": adapter}}),
        ("adapter_extensions_v1.json", {"adapters": {}}),
    ):
        (directory / filename).write_text(json.dumps(content), encoding="utf-8")
    return tmp_path / "catalog"


def test_unified_index_and_resolve_preserve_original_package(tmp_path):
    from vllm_hust_workload_data.adapters import iter_workload
    from vllm_hust_workload_data.bindings import bind_source
    from vllm_hust_workload_data.consumer import resolve_input
    from vllm_hust_workload_data.pipeline import prepared_rows, run

    data = tmp_path / "data"
    manifest, metadata = package(data)
    root = catalog(tmp_path, manifest)
    metadata_file = "tasks/fixture/metadata.json"
    bound = bind_source(
        "NATUREBENCH",
        data_root=data,
        files=[metadata_file],
        revision=manifest["revision"],
        catalog_root=root,
    )
    assert len(bound["files"]) == 1
    assert len(bound["native_task_assets"]) == 7
    binding = tmp_path / "binding.json"
    binding.write_text(json.dumps(bound), encoding="utf-8")
    expected = list(
        iter_workload("NATUREBENCH", data_root=data, manifest=bound, catalog_root=root)
    )[0]
    assert expected.outcome_record()["original_task_metadata"] == metadata
    assert "evaluation_assets" not in json.dumps(expected.input_record())
    with pytest.raises(ValueError, match="not a ready"):
        expected.request_body()
    output = tmp_path / "prepared"
    report = run(
        {
            "sources": [
                {
                    "source_id": "NATUREBENCH",
                    "data_root": str(data),
                    "manifest": str(binding),
                }
            ]
        },
        output=output,
        catalog_root=root,
    )
    assert report["native_tasks_run"] == 0
    assert report["sources"][0]["records"] == 1
    rows = list(prepared_rows(output / "workload.sqlite"))
    resolved = resolve_input(
        output / "workload.sqlite", rows[0]["record_id"], data_root=data
    )
    assert resolved.input_record() == expected.input_record()
    assert resolved.outcome_record() == expected.outcome_record()
    assert b"hidden-answer" not in (output / "workload.sqlite").read_bytes()
    assert b"original\\r\\nproblem" not in (output / "workload.sqlite").read_bytes()


def test_mismatched_data_revision_rejected(tmp_path):
    from vllm_hust_workload_data.adapters import iter_workload

    manifest, _ = package(tmp_path / "data")
    root = catalog(tmp_path, manifest)
    manifest["dataset_revision"] = "different"
    with pytest.raises(ValueError, match="revision mismatch"):
        list(
            iter_workload(
                "NATUREBENCH",
                data_root=tmp_path / "data",
                manifest=manifest,
                catalog_root=root,
            )
        )


def test_resource_header_cannot_disagree_with_dataset_revision(tmp_path):
    from vllm_hust_workload_data.registry import source_spec

    manifest, _ = package(tmp_path / "data")
    root = catalog(tmp_path, manifest)
    directory = root / "data/catalogs"
    path = directory / "source_manifests_v1.json"
    content = json.loads(path.read_text("utf-8"))
    content["sources"]["NATUREBENCH"].update(files=[], files_resource="assets.json")
    path.write_text(json.dumps(content), encoding="utf-8")
    (directory / "assets.json").write_text(
        json.dumps({**manifest, "dataset_revision": "wrong"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="identity"):
        source_spec("NATUREBENCH", root)


def test_installed_ci_evidence_does_not_mark_scientific_execution_complete():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    receipt = json.loads(
        (
            root
            / "docs/workload-data/validation/naturebench-original-readiness-20261028.json"
        ).read_text("utf-8")
    )
    package_reader = receipt["original_package_reader"]
    assert package_reader["installed_ci_wheel_research_readback_verified"]
    actual = package_reader["actual_ci_wheel"]
    assert actual["artifact_id"] == 12783786040
    assert actual["verified_installed_payloads"] == 59
    assert (
        actual["original_task_packages"]
        == actual["provider_readbacks"]
        == actual["research_readbacks"]
        == 3
    )
    assert actual["model_calls"] == 0
    assert not actual["scientific_execution"] and not actual["source_checkbox_eligible"]
