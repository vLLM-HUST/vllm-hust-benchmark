import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from vllm_hust_benchmark import swe_bench_pro_grading as grading
from vllm_hust_benchmark import swe_patch_state as state


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--allow-empty",
        "-qm",
        "initial",
    )
    return root


def new_patch(name="new.txt"):
    return (
        f"diff --git a/{name} b/{name}\nnew file mode 100644\n"
        "index 0000000..ce01362\n--- /dev/null\n"
        f"+++ b/{name}\n@@ -0,0 +1 @@\n+hello\n"
    )


def execute(root, tmp_path, patch, conflict=False):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "patch.diff").write_text(patch)
    shutil.copyfile(Path(state.__file__), workspace / grading.HELPER)
    if conflict:
        (root / "new.txt").write_text("pre-existing fixture\n")
    original = (
        grading.APPLY
        + "\n"
        + grading.TEST_MARKER
        + "\nprintf 'ran' > /workspace/tests-ran\n"
    )
    script = grading.instrument_entryscript(original)
    script = script.replace("/workspace/", str(workspace) + "/")
    # Use the exact Python running pytest, without relying on system aliases.
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    subprocess.run(["bash", "-c", script], cwd=root, env=env, check=True)
    return workspace


def test_success_preserves_test_execution_and_captures_content(repo, tmp_path):
    work = execute(repo, tmp_path, new_patch())
    value = grading.summarize_application(work)
    assert value["status"] == "applied"
    assert value["apply_exit_code"] == 0
    assert (work / "tests-ran").read_text() == "ran"
    before = json.loads((work / "hust-patch-before.json").read_text())
    after = json.loads((work / "hust-patch-after.json").read_text())
    assert before["files"]["new.txt"] == {"kind": "missing"}
    assert after["files"]["new.txt"]["sha256"] == hashlib.sha256(b"hello\n").hexdigest()


def test_existing_file_conflict_retained_not_cleaned(repo, tmp_path):
    work = execute(repo, tmp_path, new_patch(), conflict=True)
    value = grading.summarize_application(work)
    assert value["status"] == "apply_failed"
    assert value["apply_exit_code"] != 0
    assert (repo / "new.txt").read_text() == "pre-existing fixture\n"
    assert (work / "tests-ran").read_text() == "ran"
    assert "already exists" in (work / "hust-patch-apply.stderr").read_text()


def test_empty_patch_retains_known_apply_failure_even_if_snapshot_fails(repo, tmp_path):
    work = execute(repo, tmp_path, "")
    value = grading.summarize_application(work)
    assert value["status"] == "apply_failed"
    assert value["apply_exit_code"] != 0
    assert value["state_capture"] == "incomplete"
    assert (work / "tests-ran").read_text() == "ran"


def test_real_git_rename_records_both_names_before_and_after(repo, tmp_path):
    old_name, new_name = "old name.txt", "new name.txt"
    (repo / old_name).write_text("original content\\n")
    git(repo, "add", old_name)
    git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "source",
    )
    git(repo, "mv", old_name, new_name)
    patch = git(repo, "diff", "--cached", "--binary", "--find-renames").decode()
    git(repo, "reset", "--hard", "HEAD")
    work = execute(repo, tmp_path, patch)
    value = grading.summarize_application(work)
    assert value["status"] == "applied"
    before = json.loads((work / "hust-patch-before.json").read_text())["files"]
    after = json.loads((work / "hust-patch-after.json").read_text())["files"]
    assert set(before) == set(after) == {old_name, new_name}
    assert before[old_name]["kind"] == "file"
    assert before[new_name]["kind"] == "missing"
    assert after[old_name]["kind"] == "missing"
    assert after[new_name]["sha256"] == before[old_name]["sha256"]


@pytest.mark.parametrize(
    "script",
    [
        "",
        grading.APPLY,
        grading.APPLY + "\n" + grading.APPLY + "\n" + grading.TEST_MARKER,
        grading.TEST_MARKER + "\n" + grading.APPLY,
        grading.APPLY + "\n" + grading.TEST_MARKER + "\n# hust-patch-before",
    ],
)
def test_unsupported_entry_rejected(script):
    with pytest.raises(ValueError):
        grading.instrument_entryscript(script)


@pytest.mark.parametrize("name", ["../outside", "/outside", ".git/config"])
def test_unsafe_paths_rejected(repo, tmp_path, name, monkeypatch):
    monkeypatch.setattr(
        state.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout=f"1\t0\t{name}\0".encode()),
    )
    with pytest.raises(ValueError):
        state.affected_paths(tmp_path / "patch", repo)


def test_escaping_parent_symlink_not_read(repo, tmp_path):
    (repo / "external").symlink_to(tmp_path)
    with pytest.raises(ValueError):
        state.file_state(repo, "external/file")


def test_symlink_itself_is_not_followed(repo, tmp_path):
    (repo / "link").symlink_to(tmp_path / "private")
    assert state.file_state(repo, "link")["kind"] == "symlink"


def test_rename_and_tab_paths_use_nul_parser(repo, tmp_path, monkeypatch):
    monkeypatch.setattr(
        state.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            stdout=b"1\t0\t\0old name\0new name\0" + b"2\t0\ttab\tname\0"
        ),
    )
    assert state.affected_paths(tmp_path / "patch", repo) == [
        "new name",
        "old name",
        "tab\tname",
    ]


@pytest.mark.parametrize("corruption", ["patch", "capture", "files", "log"])
def test_incomplete_or_changed_evidence_not_attested(repo, tmp_path, corruption):
    work = execute(repo, tmp_path, new_patch())
    if corruption == "patch":
        (work / "patch.diff").write_text("different")
    elif corruption == "log":
        (work / "hust-patch-apply.stderr").unlink()
    else:
        path = work / "hust-patch-after.json"
        data = json.loads(path.read_text())
        if corruption == "capture":
            data["status"] = "capture_failed"
        else:
            data["files"] = {}
        path.write_text(json.dumps(data))
    assert grading.summarize_application(work)["status"] == "unattested"


def test_wrapper_preserves_original_patch_filter_and_scoring():
    module = SimpleNamespace()
    module.create_entryscript = lambda sample: (
        grading.APPLY + "\n" + grading.TEST_MARKER
    )

    def original_files(patch, sample):
        return {
            "patch.diff": patch.replace("binary", ""),
            "entryscript.sh": module.create_entryscript(sample),
        }, "unchanged-return"

    module.assemble_workspace_files = original_files
    scorer = object()
    module.main = scorer
    grading.install(module)
    files, returned = module.assemble_workspace_files("textbinary", {})
    assert files["patch.diff"] == "text"
    assert returned == "unchanged-return"
    assert module.main is scorer
    assert files["entryscript.sh"].count(grading.APPLY) == 1
    assert grading.HELPER in files


def test_absent_evidence_is_not_applied(tmp_path):
    assert grading.summarize_application(tmp_path)["status"] == "unattested"


def test_wrong_official_sha_rejected_before_import(tmp_path, monkeypatch):
    script = tmp_path / "official.py"
    script.write_text("raise RuntimeError('must not import')\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["adapter", "--official-script", str(script), "--official-sha256", "0" * 64],
    )
    with pytest.raises(ValueError, match="source changed"):
        grading.main()
