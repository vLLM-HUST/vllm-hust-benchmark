"""Real Git repositories exercise exact submodule fixture admission, no Docker."""

import copy
import hashlib
import os
import subprocess

import pytest

from vllm_hust_benchmark import swe_bench_pro as runner
from vllm_hust_benchmark import swe_submodule_fixture as fixture


def git(root, *args):
    return subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            *args,
        ],
        text=True,
        stderr=subprocess.PIPE,
    )


class RealShell:
    def __init__(self, root):
        self.root = root

    def execute(self, command):
        result = subprocess.run(
            ["bash", "-c", command],
            cwd=self.root,
            capture_output=True,
            text=True,
        )
        return {"returncode": result.returncode, "output": result.stdout}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def original(tmp_path):
    source, parent = tmp_path / "source", tmp_path / "parent"
    source.mkdir()
    parent.mkdir()
    git(source, "init", "-q")
    (source / "infogami").mkdir()
    (source / "infogami/code.py").write_text("original\n")
    git(source, "add", ".")
    git(source, "commit", "-qm", "original submodule")
    commit = git(source, "rev-parse", "HEAD").strip()
    git(parent, "init", "-q")
    (parent / "main.py").write_text("original parent\n")
    git(
        parent,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(source),
        "vendor/infogami",
    )
    git(parent, "add", ".")
    git(parent, "commit", "-qm", "original parent")
    base = git(parent, "rev-parse", "HEAD").strip()
    nested = parent / "vendor/infogami"
    link = nested / "infogami/infogami"
    link.symlink_to("vendor/infogami/infogami")
    policy = {
        "image_id": "sha256:" + "1" * 64,
        "base_commit": base,
        "path": "vendor/infogami",
        "gitlink_mode": "160000",
        "gitlink_commit": commit,
        "submodule_head": commit,
        "parent_status_sha256": sha(" M vendor/infogami\0"),
        "parent_diff_sha256": sha(""),
        "submodule_status_sha256": sha("?? infogami/infogami\0"),
        "submodule_diff_sha256": sha(""),
        "submodule_cached_diff_sha256": sha(""),
        "untracked_symlink": {
            "path": "infogami/infogami",
            "type": "symlink",
            "target": "vendor/infogami/infogami",
        },
    }
    return parent, nested, link, policy, RealShell(parent)


def admit(policy, env, **kwargs):
    receipt = fixture.observe(env, policy)
    before = env.execute("git rev-parse HEAD && git status --porcelain")
    runner.validate_initial_repository(
        before,
        kwargs.get("base", policy["base_commit"]),
        submodule_policy=policy,
        submodule_observations=receipt["observations"],
        image_id=kwargs.get("image", policy["image_id"]),
    )
    return before, receipt


def test_exact_original_symlink_is_preserved(original):
    parent, nested, link, policy, env = original
    before, receipt = admit(policy, env)
    assert before["output"] == policy["base_commit"] + "\n M vendor/infogami\n"
    assert receipt["observations"]["parent_diff"] == ""
    assert git(nested, "diff", "--binary", "HEAD") == ""
    assert os.readlink(link) == policy["untracked_symlink"]["target"]
    assert fixture.observe(env, policy) == receipt
    with pytest.raises(ValueError):
        runner.validate_initial_repository(before, policy["base_commit"])
    # Parent diff remains empty for this original untracked nested link.
    assert git(parent, "diff", "--binary", "HEAD") == ""


@pytest.mark.parametrize(
    "change",
    [
        "extra_parent_tracked",
        "extra_parent_untracked",
        "nested_tracked",
        "nested_staged",
        "extra_nested_untracked",
        "extra_nested_link",
        "link_missing",
        "link_regular_file",
        "link_wrong_target",
        "link_staged",
        "submodule_head",
        "parent_index_commit",
        "parent_index_mode",
        "wrong_image",
        "wrong_base",
        "submodule_path_symlink",
    ],
)
def test_actual_git_negative_states(original, change):
    parent, nested, link, policy, env = original
    args = {}
    if change == "extra_parent_tracked":
        (parent / "main.py").write_text("changed\n")
    elif change == "extra_parent_untracked":
        (parent / "other.txt").write_text("extra\n")
    elif change in ("nested_tracked", "nested_staged"):
        (nested / "infogami/code.py").write_text("changed\n")
        if change == "nested_staged":
            git(nested, "add", "infogami/code.py")
    elif change == "extra_nested_untracked":
        (nested / "extra.txt").write_text("extra\n")
    elif change == "extra_nested_link":
        (nested / "extra-link").symlink_to("other")
    elif change == "link_missing":
        link.unlink()
    elif change == "link_regular_file":
        link.unlink()
        link.write_text(policy["untracked_symlink"]["target"])
    elif change == "link_wrong_target":
        link.unlink()
        link.symlink_to("different")
    elif change == "link_staged":
        git(nested, "add", "infogami/infogami")
    elif change == "submodule_head":
        git(nested, "commit", "--allow-empty", "-qm", "changed commit")
    elif change == "parent_index_commit":
        git(
            parent,
            "update-index",
            "--cacheinfo",
            "160000," + "2" * 40 + ",vendor/infogami",
        )
    elif change == "parent_index_mode":
        blob = git(parent, "rev-parse", "HEAD:main.py").strip()
        git(
            parent, "update-index", "--cacheinfo", "100644," + blob + ",vendor/infogami"
        )
    elif change == "wrong_image":
        args["image"] = "sha256:" + "2" * 64
    elif change == "wrong_base":
        args["base"] = "3" * 40
    elif change == "submodule_path_symlink":
        moved = parent / "vendor/moved"
        nested.rename(moved)
        nested.symlink_to("moved", target_is_directory=True)
    with pytest.raises(ValueError):
        admit(policy, env, **args)


@pytest.mark.parametrize(
    "field",
    [
        "parent_status_sha256",
        "parent_diff_sha256",
        "submodule_status_sha256",
        "submodule_diff_sha256",
        "submodule_cached_diff_sha256",
        "gitlink_mode",
        "submodule_head",
    ],
)
def test_policy_cannot_freeze_additional_dirty_state(original, field):
    policy = copy.deepcopy(original[3])
    policy[field] = "2" * (40 if field == "submodule_head" else 64)
    with pytest.raises(ValueError):
        fixture.validate_policy(policy)


@pytest.mark.parametrize(
    "path",
    [
        "../escape",
        "/absolute",
        ".git/config",
        "vendor/*",
        "vendor//sub",
        "vendor/./sub",
    ],
)
def test_policy_path_rejected(original, path):
    policy = copy.deepcopy(original[3])
    policy["path"] = path
    with pytest.raises(ValueError):
        fixture.validate_policy(policy)


def test_command_failure_does_not_admit(original):
    _, _, _, policy, env = original
    execute = env.execute

    def failed(command):
        if "ls-tree" in command:
            return {"returncode": 1, "output": ""}
        return execute(command)

    env.execute = failed
    assert fixture.observe(env, policy)["observations"] == {}
    with pytest.raises(ValueError):
        admit(policy, env)


def test_no_combined_allowances(original):
    _, _, _, policy, env = original
    before, receipt = admit(policy, env)
    with pytest.raises(ValueError):
        runner.validate_initial_repository(
            before,
            policy["base_commit"],
            ["anything/"],
            submodule_policy=policy,
            submodule_observations=receipt["observations"],
            image_id=policy["image_id"],
        )


def test_original_default_gate_unchanged(original):
    parent, _, link, policy, env = original
    before = env.execute("git rev-parse HEAD && git status --porcelain")
    with pytest.raises(ValueError):
        runner.validate_initial_repository(before, policy["base_commit"])
    link.unlink()
    before = env.execute("git rev-parse HEAD && git status --porcelain")
    runner.validate_initial_repository(before, policy["base_commit"])
    (parent / "appendonly.aof").write_text("original\n")
    before = env.execute("git rev-parse HEAD && git status --porcelain")
    runner.validate_initial_repository(
        before, policy["base_commit"], ["appendonly.aof"]
    )
