"""Synthetic provenance checks; no real task execution or benchmark scores."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "swe_audit", Path(__file__).parents[1] / "scripts/audit_pyramidkv_swe.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture(tmp_path):
    import hashlib

    row = {
        "instance_id": "fixture",
        "repo": "owner/repo",
        "base_commit": "a" * 40,
        "docker_image": "ghcr.io/scaleapi/swe-bench_pro-v2:fixture",
        "version": "2.0.0",
        "hard": False,
        "problem_statement": "synthetic problem",
        "patch": "synthetic patch",
        "test_patch": "synthetic test",
        "fail_to_pass": '["test_a"]',
        "pass_to_pass": "[]",
        "selected_test_files_to_run": '["test.py"]',
    }

    def blob(data):
        return hashlib.sha1(
            b"blob " + str(len(data)).encode() + b"\0" + data
        ).hexdigest()

    receipt = {
        key: row[key] for key in ("instance_id", "repo", "base_commit", "docker_image")
    }
    receipt.update(harness_revision=audit.HARNESS_REVISION, harness_files={})
    contents = {
        "task.toml": (
            '[task]\nname="swebench-pro/fixture"\n[environment]\ndocker_image="'
            + row["docker_image"]
            + '"\n[agent]\ntimeout_sec=3000\n[verifier]\ntimeout_sec=3000\n'
        ).encode(),
        "environment/Dockerfile": ("FROM " + row["docker_image"] + "\n").encode(),
        "tests/config.json": json.dumps(
            {
                key: row[key]
                for key in (
                    "fail_to_pass",
                    "pass_to_pass",
                    "selected_test_files_to_run",
                )
            }
        ).encode(),
        **{
            key: b"synthetic"
            for key in (
                "tests/test.sh",
                "tests/run_script.sh",
                "tests/parser.py",
                "instruction.md",
            )
        },
    }
    contents["tests/test_patch.patch"] = row["test_patch"].encode()
    contents["solution/gold_patch.diff"] = row["patch"].encode()
    tree = {}
    folder = tmp_path / "harness-metadata/fixture"
    folder.mkdir(parents=True)
    for name, data in contents.items():
        (folder / name.replace("/", "-")).write_bytes(data)
        receipt["harness_files"][name] = {
            "git_blob_sha": blob(data),
            "sha256": audit.sha(data),
        }
        tree["v2/tasks/fixture/" + name] = {"sha": blob(data)}
    license_data = b"synthetic license fixture"
    digest = audit.sha(license_data)
    folder = tmp_path / "license-texts"
    folder.mkdir()
    (folder / (digest + ".txt")).write_bytes(license_data)
    receipt["root_license"] = {
        "sha256": digest,
        "sha": blob(license_data),
        "html_url": "https://github.com/owner/repo/blob/"
        + row["base_commit"]
        + "/LICENSE",
        "license": {"spdx_id": "MIT"},
    }
    config = b'{"os":"linux","architecture":"amd64"}'
    manifest = json.dumps(
        {
            "config": {"digest": "sha256:" + audit.sha(config)},
            "layers": [{"digest": "sha256:" + "a" * 64, "size": 123}],
        }
    ).encode()
    for directory, data in [("image-manifests", manifest), ("image-configs", config)]:
        folder = tmp_path / directory
        folder.mkdir()
        (folder / "fixture.json").write_bytes(data)
    receipt["image"] = {
        "media_type": "application/vnd.oci.image.manifest.v1+json",
        "digest": "sha256:" + audit.sha(manifest),
    }
    return row, receipt, tree


def test_task_manifest_verifies_three_independent_mappings(tmp_path):
    row, receipt, tree = fixture(tmp_path)
    result = audit.audit_task(row, receipt, tmp_path, tree)
    assert result["base_commit"] == row["base_commit"]
    assert result["image_tag"] == row["docker_image"]
    assert result["test_counts"] == {
        "fail_to_pass": 1,
        "pass_to_pass": 0,
        "selected_test_files_to_run": 1,
    }
    assert result["compressed_layer_bytes"] == 123
    assert result["tests_executed"] is False
    assert result["runtime_repo_head_verified"] is False


@pytest.mark.parametrize("key", ["instance_id", "repo", "base_commit", "docker_image"])
def test_receipt_cannot_be_rebound_to_different_task(tmp_path, key):
    row, receipt, tree = fixture(tmp_path)
    receipt[key] = "changed"
    with pytest.raises(ValueError, match="receipt differs"):
        audit.audit_task(row, receipt, tmp_path, tree)


def test_verifier_test_list_must_match_dataset(tmp_path):
    row, receipt, tree = fixture(tmp_path)
    row["fail_to_pass"] = '["different_test"]'
    with pytest.raises(ValueError, match="Verifier task data differs"):
        audit.audit_task(row, receipt, tmp_path, tree)


def test_missing_verifier_file_fails_closed(tmp_path):
    row, receipt, tree = fixture(tmp_path)
    del receipt["harness_files"]["tests/parser.py"]
    with pytest.raises(ValueError, match="Missing required harness"):
        audit.audit_task(row, receipt, tmp_path, tree)


def test_modified_image_config_rejected(tmp_path):
    row, receipt, tree = fixture(tmp_path)
    (tmp_path / "image-configs/fixture.json").write_text(
        '{"os":"linux","architecture":"arm64"}'
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        audit.audit_task(row, receipt, tmp_path, tree)


def test_unresolved_image_index_is_not_verified(tmp_path):
    row, receipt, tree = fixture(tmp_path)
    receipt["image"]["media_type"] = "application/vnd.oci.image.index.v1+json"
    with pytest.raises(FileNotFoundError):
        audit.audit_task(row, receipt, tmp_path, tree)


def test_license_on_a_mutable_branch_is_rejected(tmp_path):
    row, receipt, tree = fixture(tmp_path)
    receipt["root_license"]["html_url"] = (
        "https://github.com/owner/repo/blob/main/LICENSE"
    )
    with pytest.raises(ValueError, match="License not pinned"):
        audit.audit_task(row, receipt, tmp_path, tree)


def test_legacy_harness_list_literals_are_compared_without_execution():
    value = "['test_a', 'test_b']"
    assert audit.test_list(value, allow_literal=True) == ["test_a", "test_b"]
    with pytest.raises(ValueError):
        audit.test_list(value)
    for invalid in ("__import__('os').system('true')", "{'test': 'a'}", "[1]"):
        with pytest.raises(ValueError):
            audit.test_list(invalid, allow_literal=True)


def test_crlf_test_fixture_is_preserved_as_harness_authority(tmp_path):
    import hashlib

    source, receipt, tree = fixture(tmp_path)
    source["test_patch"] = "first\nsecond\n"
    harness = b"first\r\nsecond\r\n"
    digest = hashlib.sha1(
        b"blob " + str(len(harness)).encode() + b"\0" + harness
    ).hexdigest()
    receipt["harness_files"]["tests/test_patch.patch"]["git_blob_sha"] = digest
    tree["v2/tasks/fixture/tests/test_patch.patch"]["sha"] = digest
    path = tmp_path / "harness-metadata/fixture/tests-test_patch.patch"
    path.write_bytes(harness)
    result = audit.audit_task(source, receipt, tmp_path, tree)
    assert not result["test_patch_byte_identical"]
    assert result["harness_test_patch_sha256"] == audit.sha(harness)
    assert result["verifier_patch_authority"] == "pinned harness bytes"
    assert path.read_bytes() == harness


def test_unexplained_verifier_patch_difference_is_rejected(tmp_path):
    source, receipt, tree = fixture(tmp_path)
    source["test_patch"] = "changed test logic"
    with pytest.raises(
        ValueError, match="Unexplained dataset/harness test patch difference"
    ):
        audit.audit_task(source, receipt, tmp_path, tree)
