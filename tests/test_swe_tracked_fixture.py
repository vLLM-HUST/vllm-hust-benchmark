"""Exact original tracked-fixture admission without altering task or patch."""

import copy
import hashlib
import subprocess

import pytest

from vllm_hust_benchmark import swe_bench_pro as runner


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True)


class RealShell:
    def __init__(self, root):
        self.root = root

    def execute(self, command):
        result = subprocess.run(
            ["bash", "-c", command], cwd=self.root, capture_output=True, text=True
        )
        return {"returncode": result.returncode, "output": result.stdout}


@pytest.fixture
def original(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "go.work.sum").write_text("original checksum\n")
    git(tmp_path, "add", "go.work.sum")
    git(
        tmp_path,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "base",
    )
    base = git(tmp_path, "rev-parse", "HEAD").strip()
    (tmp_path / "go.work.sum").write_text(
        "original checksum\noriginal image addition\n"
    )
    diff = git(
        tmp_path,
        "diff",
        "--binary",
        "--no-ext-diff",
        "--no-textconv",
        "HEAD",
        "--",
        "go.work.sum",
    )
    policy = {
        "image_id": "sha256:" + "1" * 64,
        "base_commit": base,
        "files": {
            "go.work.sum": {
                "file_sha256": hashlib.sha256(
                    (tmp_path / "go.work.sum").read_bytes()
                ).hexdigest(),
                "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
            }
        },
    }
    env = RealShell(tmp_path)
    before = env.execute("git rev-parse HEAD && git status --porcelain")
    return tmp_path, env, before, policy


def admit(before, policy, receipt, **kwargs):
    runner.validate_initial_repository(
        before,
        kwargs.get("base", policy["base_commit"]),
        tracked_policy=policy,
        tracked_observations=receipt["observations"],
        image_id=kwargs.get("image", policy["image_id"]),
    )


def test_exact_original_fixture_preserves_bytes_and_diff(original):
    root, env, before, policy = original
    receipt = runner.observe_initial_tracked(env, policy)
    admit(before, policy, receipt)
    with pytest.raises(ValueError):
        runner.validate_initial_repository(before, policy["base_commit"])
    assert runner.observe_initial_tracked(env, policy) == receipt
    patch = git(root, "diff", "--binary", "HEAD")
    git(root, "reset", "--hard", policy["base_commit"])
    subprocess.run(
        ["git", "-C", str(root), "apply", "-"], input=patch, text=True, check=True
    )
    assert (
        hashlib.sha256((root / "go.work.sum").read_bytes()).hexdigest()
        == policy["files"]["go.work.sum"]["file_sha256"]
    )


@pytest.mark.parametrize(
    "change",
    [
        "file",
        "diff",
        "extra_tracked",
        "extra_untracked",
        "missing_status",
        "staged",
        "image",
        "base",
    ],
)
def test_identity_and_status_negatives(original, change):
    root, env, before, policy = original
    receipt = runner.observe_initial_tracked(env, policy)
    kwargs = {}
    if change == "file":
        (root / "go.work.sum").write_text("changed after preparation\n")
        receipt = runner.observe_initial_tracked(env, policy)
    elif change == "diff":
        receipt["observations"]["go.work.sum"]["diff_sha256"] = "2" * 64
    elif change == "extra_tracked":
        before["output"] += " M other.go\n"
    elif change == "extra_untracked":
        before["output"] += "?? other.txt\n"
    elif change == "missing_status":
        before["output"] = policy["base_commit"] + "\n"
    elif change == "staged":
        before["output"] = before["output"].replace(" M ", "M  ")
    elif change == "image":
        kwargs["image"] = "sha256:" + "3" * 64
    elif change == "base":
        kwargs["base"] = "4" * 40
    with pytest.raises(ValueError):
        admit(before, policy, receipt, **kwargs)


@pytest.mark.parametrize(
    "path", ["../sum", "/sum", ".git/config", "a//b", "a/./b", "sum*", "sum\n", "a\\b"]
)
def test_unsafe_policy_paths_rejected(original, path):
    policy = copy.deepcopy(original[3])
    policy["files"] = {path: policy["files"]["go.work.sum"]}
    with pytest.raises(ValueError):
        runner.validate_initial_tracked_policy(policy)


def test_original_clean_and_untracked_semantics(original):
    root, env, _, policy = original
    git(root, "reset", "--hard", policy["base_commit"])
    before = env.execute("git rev-parse HEAD && git status --porcelain")
    runner.validate_initial_repository(before, policy["base_commit"])
    (root / "appendonly.aof").write_text("upstream fixture")
    before = env.execute("git rev-parse HEAD && git status --porcelain")
    runner.validate_initial_repository(
        before, policy["base_commit"], ["appendonly.aof"]
    )
    with pytest.raises(ValueError):
        runner.validate_initial_repository(before, policy["base_commit"])


def test_failed_observation_cannot_admit(original):
    root, env, before, policy = original
    (root / "go.work.sum").unlink()
    receipt = runner.observe_initial_tracked(env, policy)
    assert receipt["observations"] == {}
    with pytest.raises(ValueError):
        admit(before, policy, receipt)
