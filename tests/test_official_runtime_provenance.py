import importlib.util
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
    yield
    for name in ("vllm", "vllm_ascend"):
        sys.modules.pop(name, None)


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
