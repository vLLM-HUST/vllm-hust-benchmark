import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER = REPO_ROOT / "scripts/capture-official-source-provenance.py"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_capture_records_commit_and_dirty_tree_digest(tmp_path: Path) -> None:
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "--initial-branch=main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    source_file = repo / "source.py"
    source_file.write_text("print('one')\n", encoding="utf-8")
    _git(repo, "add", "source.py")
    _git(repo, "commit", "-m", "initial")

    clean_output = tmp_path / "clean.json"
    subprocess.run(
        [str(HELPER), str(repo), "main", "example/source", str(clean_output)],
        check=True,
    )
    clean = json.loads(clean_output.read_text(encoding="utf-8"))
    assert clean["status"] == "clean"
    assert len(clean["observed_commit"]) == 40
    assert clean["tracked_patch_sha256"]
    assert clean["working_tree_sha256"]

    source_file.write_text("print('two')\n", encoding="utf-8")
    dirty_output = tmp_path / "dirty.json"
    subprocess.run(
        [str(HELPER), str(repo), "main", "example/source", str(dirty_output)],
        check=True,
    )
    dirty = json.loads(dirty_output.read_text(encoding="utf-8"))
    assert dirty["status"] == "modified"
    assert dirty["tracked_patch_sha256"] != clean["tracked_patch_sha256"]
    assert dirty["working_tree_sha256"] != clean["working_tree_sha256"]


def test_capture_includes_ignored_generated_runtime_inputs(tmp_path: Path) -> None:
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "--initial-branch=main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / ".gitignore").write_text("vllm_ascend/_build_info.py\n", encoding="utf-8")
    (repo / "tracked.txt").write_text("stable\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    generated = repo / "vllm_ascend" / "_build_info.py"
    generated.parent.mkdir()
    generated.write_text("__soc_version__ = 'ascend910b3'\n", encoding="utf-8")

    first_output = tmp_path / "first.json"
    subprocess.run(
        [str(HELPER), str(repo), "main", "example/source", str(first_output)],
        check=True,
    )
    first = json.loads(first_output.read_text(encoding="utf-8"))
    assert first["status"] == "modified"
    assert first["generated_file_count"] == 1

    generated.write_text("__soc_version__ = 'ascend910b2'\n", encoding="utf-8")
    second_output = tmp_path / "second.json"
    subprocess.run(
        [str(HELPER), str(repo), "main", "example/source", str(second_output)],
        check=True,
    )
    second = json.loads(second_output.read_text(encoding="utf-8"))
    assert second["working_tree_sha256"] != first["working_tree_sha256"]


def test_capture_includes_ignored_compiled_extensions(tmp_path: Path) -> None:
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "--initial-branch=main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / ".gitignore").write_text("*.so\n", encoding="utf-8")
    (repo / "tracked.txt").write_text("stable\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    extension = repo / "vllm_ascend" / "vllm_ascend_C.test.so"
    extension.parent.mkdir()
    extension.write_bytes(b"first-build")

    first_output = tmp_path / "first-extension.json"
    subprocess.run(
        [str(HELPER), str(repo), "main", "example/source", str(first_output)],
        check=True,
    )
    first = json.loads(first_output.read_text(encoding="utf-8"))
    assert first["status"] == "modified"
    assert first["generated_file_count"] == 1

    extension.write_bytes(b"stale-or-replaced-build")
    second_output = tmp_path / "second-extension.json"
    subprocess.run(
        [str(HELPER), str(repo), "main", "example/source", str(second_output)],
        check=True,
    )
    second = json.loads(second_output.read_text(encoding="utf-8"))
    assert second["working_tree_sha256"] != first["working_tree_sha256"]
