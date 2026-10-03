import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER = REPO_ROOT / "scripts/capture-official-runtime-provenance.py"


@pytest.fixture(autouse=True)
def _clear_fake_runtime_modules():
    for name in ("vllm", "vllm_ascend"):
        sys.modules.pop(name, None)
        sys.modules.pop(f"{name}._version", None)
    yield
    for name in ("vllm", "vllm_ascend"):
        sys.modules.pop(name, None)
        sys.modules.pop(f"{name}._version", None)


def _load_helper():
    spec = importlib.util.spec_from_file_location("official_runtime_provenance", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_repo(tmp_path: Path, package: str, version: str) -> tuple[Path, str]:
    repo = tmp_path / package
    module_dir = repo / package
    module_dir.mkdir(parents=True)
    (module_dir / "__init__.py").write_text(
        f"__version__ = {version!r}\n", encoding="utf-8"
    )
    _git(repo, "init", "--initial-branch=main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    _git(repo, "tag", f"v{version}")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_clean_prepared_worktree_runtime_pair_passes(
    tmp_path: Path, monkeypatch
) -> None:
    helper = _load_helper()
    engine_repo, engine_commit = _source_repo(tmp_path, "vllm", "1.2.3")
    plugin_repo, plugin_commit = _source_repo(tmp_path, "vllm_ascend", "4.5.6")
    monkeypatch.syspath_prepend(str(plugin_repo))
    monkeypatch.syspath_prepend(str(engine_repo))
    monkeypatch.setattr(
        helper,
        "_distribution_version",
        lambda names: (names[0], "1.2.3" if names[0] == "vllm" else "4.5.6"),
    )

    payload = helper.capture(engine_repo, engine_commit, plugin_repo, plugin_commit)

    assert payload["schema_version"] == "official-runtime-provenance/v1"
    assert payload["sources"]["engine"]["prepared_commit"] == engine_commit
    assert payload["sources"]["plugin"]["prepared_commit"] == plugin_commit
    assert payload["sources"]["engine"]["module_path"].startswith(str(engine_repo))
    assert payload["sources"]["plugin"]["module_path"].startswith(str(plugin_repo))


def test_runtime_evidence_binds_source_tree_and_patch_digests(
    tmp_path: Path, monkeypatch
) -> None:
    helper = _load_helper()
    engine_repo, engine_commit = _source_repo(tmp_path, "vllm", "1.2.3")
    plugin_repo, plugin_commit = _source_repo(tmp_path, "vllm_ascend", "4.5.6")
    monkeypatch.syspath_prepend(str(plugin_repo))
    monkeypatch.syspath_prepend(str(engine_repo))
    monkeypatch.setattr(
        helper,
        "_distribution_version",
        lambda names: (names[0], "1.2.3" if names[0] == "vllm" else "4.5.6"),
    )
    source_provenance = {
        "schema_version": "official-source-provenance/v1",
        "sources": {
            "engine": {
                "observed_commit": engine_commit,
                "tracked_patch_sha256": "a" * 64,
                "working_tree_sha256": "b" * 64,
                "status": "clean",
            },
            "plugin": {
                "observed_commit": plugin_commit,
                "tracked_patch_sha256": "c" * 64,
                "working_tree_sha256": "d" * 64,
                "status": "modified",
            },
        },
    }

    payload = helper.capture(
        engine_repo,
        engine_commit,
        plugin_repo,
        plugin_commit,
        source_provenance,
    )

    assert payload["sources"]["engine"]["source_tree_sha256"] == "b" * 64
    assert payload["sources"]["plugin"]["source_patch_sha256"] == "c" * 64
    assert payload["sources"]["plugin"]["source_status"] == "modified"


def test_stale_editable_module_path_is_rejected(tmp_path: Path, monkeypatch) -> None:
    helper = _load_helper()
    prepared_repo, prepared_commit = _source_repo(
        tmp_path / "prepared", "vllm", "1.2.3"
    )
    stale_repo, _ = _source_repo(tmp_path / "stale", "vllm", "1.2.3")
    monkeypatch.syspath_prepend(str(stale_repo))
    monkeypatch.setattr(
        helper, "_distribution_version", lambda names: (names[0], "1.2.3")
    )

    try:
        helper.capture_role("engine", prepared_repo, prepared_commit)
    except ValueError as error:
        assert "module path mismatch" in str(error)
        assert str(stale_repo) in str(error)
    else:
        raise AssertionError("stale editable module path was accepted")


def test_image_native_runtime_is_bound_by_immutable_image_identity(
    tmp_path: Path, monkeypatch
) -> None:
    helper = _load_helper()
    prepared_repo, prepared_commit = _source_repo(
        tmp_path / "prepared", "vllm", "1.2.3"
    )
    runtime_root = tmp_path / "image-root"
    runtime_package = runtime_root / "vllm"
    runtime_package.mkdir(parents=True)
    (runtime_package / "__init__.py").write_text(
        "__version__ = '1.2.3'\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(runtime_root))
    monkeypatch.setattr(
        helper, "_distribution_version", lambda names: (names[0], "1.2.3")
    )

    payload = helper.capture_role(
        "engine",
        prepared_repo,
        prepared_commit,
        runtime_root=runtime_root,
        image_commit=prepared_commit,
        image_id="sha256:" + "a" * 64,
    )

    assert payload["runtime_binding"] == "image-native-oci"
    assert payload["runtime_root"] == str(runtime_root)
    assert payload["runtime_image_commit"] == prepared_commit
    assert payload["runtime_image_id"] == "sha256:" + "a" * 64


@pytest.mark.parametrize("field", ["image_commit", "image_id"])
def test_image_native_runtime_rejects_missing_or_mismatched_image_identity(
    tmp_path: Path, monkeypatch, field: str
) -> None:
    helper = _load_helper()
    prepared_repo, prepared_commit = _source_repo(
        tmp_path / "prepared", "vllm", "1.2.3"
    )
    runtime_root = tmp_path / "image-root"
    runtime_package = runtime_root / "vllm"
    runtime_package.mkdir(parents=True)
    (runtime_package / "__init__.py").write_text(
        "__version__ = '1.2.3'\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(runtime_root))
    monkeypatch.setattr(
        helper, "_distribution_version", lambda names: (names[0], "1.2.3")
    )
    kwargs = {
        "runtime_root": runtime_root,
        "image_commit": prepared_commit,
        "image_id": "sha256:" + "a" * 64,
    }
    kwargs[field] = "" if field == "image_id" else "f" * 40

    with pytest.raises(ValueError, match="image (commit mismatch|id)"):
        helper.capture_role("engine", prepared_repo, prepared_commit, **kwargs)


def test_module_and_distribution_version_mismatch_is_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    helper = _load_helper()
    engine_repo, engine_commit = _source_repo(tmp_path, "vllm", "1.2.3")
    monkeypatch.syspath_prepend(str(engine_repo))
    monkeypatch.setattr(
        helper, "_distribution_version", lambda names: (names[0], "1.2.4")
    )

    try:
        helper.capture_role("engine", engine_repo, engine_commit)
    except ValueError as error:
        assert "runtime version mismatch" in str(error)
    else:
        raise AssertionError("mismatched runtime package versions were accepted")


def test_generated_version_module_proves_package_without_top_level_version(
    tmp_path: Path, monkeypatch
) -> None:
    helper = _load_helper()
    plugin_repo, plugin_commit = _source_repo(tmp_path, "vllm_ascend", "4.5.6")
    package = plugin_repo / "vllm_ascend"
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "_version.py").write_text(
        "__version__ = '4.5.6'\n"
        f"__commit_id__ = {plugin_commit[:8]!r}\n"
        "__upstream_commit__ = None\n",
        encoding="utf-8",
    )
    _git(plugin_repo, "add", ".")
    _git(plugin_repo, "commit", "-m", "move version metadata")
    plugin_commit = _git(plugin_repo, "rev-parse", "HEAD")
    (package / "_version.py").write_text(
        "__version__ = '4.5.6'\n"
        f"__commit_id__ = {plugin_commit[:8]!r}\n"
        "__upstream_commit__ = None\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(plugin_repo))
    monkeypatch.setattr(
        helper, "_distribution_version", lambda names: (names[0], "4.5.6")
    )

    payload = helper.capture_role("plugin", plugin_repo, plugin_commit)

    assert payload["module_version"] == "4.5.6"
    assert payload["module_commit"] == plugin_commit[:8]


def test_plugin_runtime_proves_official_wheel_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    helper = _load_helper()
    plugin_repo, plugin_commit = _source_repo(tmp_path, "vllm_ascend", "4.5.6")
    artifact = plugin_repo / "vllm_ascend" / "_cann_ops_custom" / "kernel.o"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"official-kernel")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "official-vllm-ascend-wheel-extraction/v1",
        "wheel": "/cache/vllm_ascend-4.5.6.whl",
        "wheel_version": "4.5.6",
        "wheel_sha256": "a" * 64,
        "artifact_count": 1,
        "artifacts": [
            {
                "path": "vllm_ascend/_cann_ops_custom/kernel.o",
                "size_bytes": artifact.stat().st_size,
                "sha256": digest,
            }
        ],
    }
    (plugin_repo / "vllm_ascend" / "_official_wheel_artifacts.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(plugin_repo))
    monkeypatch.setattr(
        helper, "_distribution_version", lambda names: (names[0], "4.5.6")
    )

    payload = helper.capture_role("plugin", plugin_repo, plugin_commit)

    evidence = payload["official_wheel_artifacts"]
    assert evidence["wheel_sha256"] == "a" * 64
    assert evidence["artifact_count"] == 1


def test_plugin_runtime_rejects_changed_official_wheel_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    helper = _load_helper()
    plugin_repo, plugin_commit = _source_repo(tmp_path, "vllm_ascend", "4.5.6")
    artifact = plugin_repo / "vllm_ascend" / "_cann_ops_custom" / "kernel.o"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"changed")
    manifest = {
        "schema_version": "official-vllm-ascend-wheel-extraction/v1",
        "artifacts": [
            {
                "path": "vllm_ascend/_cann_ops_custom/kernel.o",
                "sha256": "0" * 64,
            }
        ],
    }
    (plugin_repo / "vllm_ascend" / "_official_wheel_artifacts.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(plugin_repo))
    monkeypatch.setattr(
        helper, "_distribution_version", lambda names: (names[0], "4.5.6")
    )

    with pytest.raises(ValueError, match="artifact SHA256 mismatch"):
        helper.capture_role("plugin", plugin_repo, plugin_commit)


def test_generated_module_commit_mismatch_is_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    helper = _load_helper()
    engine_repo, engine_commit = _source_repo(tmp_path, "vllm", "1.2.3")
    monkeypatch.syspath_prepend(str(engine_repo))
    monkeypatch.setattr(
        helper, "_distribution_version", lambda names: (names[0], "1.2.3")
    )
    module = __import__("vllm")
    module.__commit_id__ = "f" * 8

    try:
        helper.capture_role("engine", engine_repo, engine_commit)
    except ValueError as error:
        assert "runtime build commit mismatch" in str(error)
    else:
        raise AssertionError("mismatched generated module commit was accepted")


def test_generated_module_commit_proves_untagged_source(
    tmp_path: Path, monkeypatch
) -> None:
    helper = _load_helper()
    engine_repo, engine_commit = _source_repo(tmp_path, "vllm", "1.2.3")
    _git(engine_repo, "tag", "--delete", "v1.2.3")
    monkeypatch.syspath_prepend(str(engine_repo))
    monkeypatch.setattr(
        helper, "_distribution_version", lambda names: (names[0], "1.2.3")
    )
    module = __import__("vllm")
    module.__commit_id__ = engine_commit[:8]

    payload = helper.capture_role("engine", engine_repo, engine_commit)

    assert payload["source_version"] == engine_commit[:7]
    assert payload["module_commit"] == engine_commit[:8]


def test_cli_does_not_write_output_after_validation_failure(tmp_path: Path) -> None:
    output = tmp_path / "runtime.json"
    result = subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "--engine-worktree",
            str(tmp_path / "missing-engine"),
            "--engine-commit",
            "a" * 40,
            "--plugin-worktree",
            str(tmp_path / "missing-plugin"),
            "--plugin-commit",
            "b" * 40,
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "official runtime provenance validation failed" in result.stderr
    assert not output.exists()
