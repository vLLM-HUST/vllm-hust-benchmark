"""Exact composite fixtures use real Git repositories, without Docker."""

import copy

import pytest

from test_swe_submodule_fixture import fixture, git, original, runner, sha  # noqa: F401


@pytest.fixture
def composite(original):  # noqa: F811 - pytest injects the imported real-Git fixture
    parent, nested, link, policy, env = original
    (parent / "config.txt").write_text("base config\n")
    git(parent, "add", "config.txt")
    git(parent, "commit", "-qm", "second original tracked file")
    policy = copy.deepcopy(policy)
    policy["base_commit"] = git(parent, "rev-parse", "HEAD").strip()
    files = {}
    for name in ("main.py", "config.txt"):
        (parent / name).write_text("original image fixture\n")
        files[name] = {
            "file_sha256": sha((parent / name).read_text()),
            "diff_sha256": sha(
                git(
                    parent,
                    "diff",
                    "--binary",
                    "--no-ext-diff",
                    "--no-textconv",
                    "HEAD",
                    "--",
                    name,
                )
            ),
        }
    tracked = {
        "image_id": policy["image_id"],
        "base_commit": policy["base_commit"],
        "files": files,
    }
    return parent, nested, policy, tracked, env


def composite_observations(composite):
    _, _, policy, tracked, env = composite
    return (
        env.execute("git rev-parse HEAD && git status --porcelain"),
        runner.observe_initial_tracked(env, tracked)["observations"],
        fixture.observe(env, policy)["observations"],
    )


def admit_composite(composite, observations=None):
    _, _, policy, tracked, _ = composite
    before, tracked_actual, submodule_actual = (
        composite_observations(composite) if observations is None else observations
    )
    runner.validate_initial_repository(
        before,
        policy["base_commit"],
        tracked_policy=tracked,
        tracked_observations=tracked_actual,
        submodule_policy=policy,
        submodule_observations=submodule_actual,
        image_id=policy["image_id"],
    )


def test_composite_exact_union_preserves_raw_observations(composite):
    observed = composite_observations(composite)
    saved = copy.deepcopy(observed)
    admit_composite(composite, observed)
    assert observed == saved


@pytest.mark.parametrize(
    "change",
    [
        "tracked_bytes",
        "tracked_staged",
        "tracked_missing",
        "extra_parent",
        "nested_bytes",
        "nested_extra",
        "link_target",
    ],
)
def test_composite_actual_git_mutation_rejected(composite, change):
    parent, nested, _, _, _ = composite
    if change == "tracked_bytes":
        (parent / "main.py").write_text("changed again\n")
    elif change == "tracked_staged":
        git(parent, "add", "main.py")
    elif change == "tracked_missing":
        git(parent, "checkout", "--", "main.py")
    elif change == "extra_parent":
        (parent / "extra").write_text("extra\n")
    elif change == "nested_bytes":
        (nested / "infogami/code.py").write_text("changed\n")
    elif change == "nested_extra":
        (nested / "extra").write_text("extra\n")
    elif change == "link_target":
        link = nested / "infogami/infogami"
        link.unlink()
        link.symlink_to("different")
    with pytest.raises(ValueError):
        admit_composite(composite)


@pytest.mark.parametrize(
    "change",
    [
        "nul_extra",
        "nul_duplicate",
        "nul_missing",
        "nul_unterminated",
        "before_duplicate",
        "before_missing",
        "command_failure",
    ],
)
def test_composite_complete_parent_status_required(composite, change):
    before, tracked, sub = composite_observations(composite)
    if change == "nul_extra":
        sub["parent_status"] += "?? extra\0"
    elif change == "nul_duplicate":
        sub["parent_status"] += " M main.py\0"
    elif change == "nul_missing":
        sub["parent_status"] = " M vendor/infogami\0"
    elif change == "nul_unterminated":
        sub["parent_status"] = sub["parent_status"].rstrip("\0")
    elif change == "before_duplicate":
        before["output"] += " M main.py\n"
    elif change == "before_missing":
        before["output"] = before["output"].replace(" M main.py\n", "")
    elif change == "command_failure":
        before["returncode"] = 1
    with pytest.raises(ValueError):
        admit_composite(composite, (before, tracked, sub))


@pytest.mark.parametrize(
    "change", ["image", "base", "child", "parent", "same", "untracked"]
)
def test_composite_policy_identity_and_disjointness(composite, change):
    _, _, policy, tracked, _ = composite
    tracked = copy.deepcopy(tracked)
    allowed = ()
    if change == "image":
        tracked["image_id"] = "sha256:" + "2" * 64
    elif change == "base":
        tracked["base_commit"] = "2" * 40
    elif change in ("child", "parent", "same"):
        path = {
            "child": "vendor/infogami/file",
            "parent": "vendor",
            "same": "vendor/infogami",
        }[change]
        tracked["files"] = {path: next(iter(tracked["files"].values()))}
    elif change == "untracked":
        allowed = ("vendor/",)
    with pytest.raises(ValueError):
        runner._validate_initial_fixture_combination(tracked, policy, allowed)
