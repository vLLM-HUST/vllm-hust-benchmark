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
