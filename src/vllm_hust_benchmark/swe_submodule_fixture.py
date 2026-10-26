"""Read-only admission for one exact original submodule with one extra symlink.

This does not update submodules, edit files, or change patch extraction/grading.
"""

import hashlib
import re
import shlex


EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _path(value):
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9_./-]+", value)
        or value.startswith("/")
        or any(part in ("", ".", "..", ".git") for part in value.split("/"))
    ):
        raise ValueError("Submodule fixture requires an exact relative path")


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def validate_policy(policy):
    if policy is None:
        return
    if not isinstance(policy, dict) or set(policy) != {
        "image_id",
        "base_commit",
        "path",
        "gitlink_mode",
        "gitlink_commit",
        "submodule_head",
        "parent_status_sha256",
        "parent_diff_sha256",
        "submodule_status_sha256",
        "submodule_diff_sha256",
        "submodule_cached_diff_sha256",
        "untracked_symlink",
    }:
        raise ValueError("Exact submodule fixture identity and state required")
    if not isinstance(policy["image_id"], str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", policy["image_id"]
    ):
        raise ValueError("Content-addressed submodule fixture image required")
    for field in ("base_commit", "gitlink_commit", "submodule_head"):
        if not isinstance(policy[field], str) or not re.fullmatch(
            r"[0-9a-f]{40}", policy[field]
        ):
            raise ValueError("Exact Git commit required")
    if policy["gitlink_mode"] != "160000" or (
        policy["submodule_head"] != policy["gitlink_commit"]
    ):
        raise ValueError("Original submodule HEAD must equal the gitlink commit")
    _path(policy["path"])
    link = policy["untracked_symlink"]
    if not isinstance(link, dict) or set(link) != {"path", "type", "target"}:
        raise ValueError("Bind exactly one original symlink")
    _path(link["path"])
    if (
        link["type"] != "symlink"
        or not isinstance(link["target"], str)
        or not link["target"]
        or any(ord(char) < 32 for char in link["target"])
    ):
        raise ValueError("Exact symlink type and literal target required")
    expected_hashes = {
        "parent_status_sha256": _sha(" M " + policy["path"] + "\0"),
        "submodule_status_sha256": _sha("?? " + link["path"] + "\0"),
        "parent_diff_sha256": EMPTY_SHA256,
        "submodule_diff_sha256": EMPTY_SHA256,
        "submodule_cached_diff_sha256": EMPTY_SHA256,
    }
    if any(policy[key] != value for key, value in expected_hashes.items()):
        raise ValueError("Only clean tracked/index state plus one symlink is allowed")


def _expected(policy):
    path, commit = policy["path"], policy["gitlink_commit"]
    link = policy["untracked_symlink"]
    return {
        "directory": "",
        "parent_head": policy["base_commit"] + "\n",
        "tree": "160000 commit " + commit + "\t" + path + "\0",
        "index": "160000 " + commit + " 0\t" + path + "\0",
        "parent_status": " M " + path + "\0",
        "parent_diff": "",
        "submodule_head": policy["submodule_head"] + "\n",
        "submodule_root_prefix": "\n",
        "submodule_status": "?? " + link["path"] + "\0",
        "submodule_diff": "",
        "submodule_cached_diff": "",
        "untracked_paths": link["path"] + "\0",
        "link_type": "symlink\n",
        "link_target": link["target"] + "\n",
    }


def observe(env, policy):
    validate_policy(policy)
    path = shlex.quote(policy["path"])
    link = shlex.quote(policy["path"] + "/" + policy["untracked_symlink"]["path"])
    git = "git --literal-pathspecs "
    nested = git + "-C " + path + " "
    diff = "diff --binary --no-ext-diff --no-textconv "
    status = "status --porcelain=v1 -z --untracked-files=all --ignore-submodules=none"
    commands = {
        "directory": "test -d " + path + " && test ! -L " + path,
        "parent_head": git + "rev-parse HEAD",
        "tree": git + "ls-tree -z HEAD -- " + path,
        "index": git + "ls-files --stage -z -- " + path,
        "parent_status": git + status,
        "parent_diff": git + diff + "HEAD -- " + path,
        "submodule_head": nested + "rev-parse HEAD",
        "submodule_root_prefix": nested + "rev-parse --show-prefix",
        "submodule_status": nested + status,
        "submodule_diff": nested + diff + "HEAD",
        "submodule_cached_diff": nested + diff + "--cached HEAD",
        "untracked_paths": nested + "ls-files --others --exclude-standard -z",
        "link_type": "test -L " + link + " && printf 'symlink\\n'",
        "link_target": "readlink -- " + link,
    }
    records = {
        key: env.execute("export GIT_OPTIONAL_LOCKS=0; " + command)
        for key, command in commands.items()
    }
    observations = {}
    if all(item["returncode"] == 0 for item in records.values()):
        observations = {key: item["output"] for key, item in records.items()}
    return {"records": records, "observations": observations}


def validate_observation(policy, observations, *, base_commit, image_id):
    validate_policy(policy)
    if (
        policy["base_commit"] != base_commit
        or policy["image_id"] != image_id
        or observations != _expected(policy)
    ):
        raise ValueError("Original submodule fixture identity or state mismatch")
    return " M " + policy["path"]
