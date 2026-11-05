"""Recover the frozen three original Git snapshots from publisher API/archives.

This is a bounded transport workaround, not a generated commit or benchmark run.
Every recovered commit and root tree must match its original Git object ID.
Ordinary users can instead clone the publisher's repository normally.
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
from urllib.request import Request, urlopen
import zlib

REPOSITORY = "swesmith/oauthlib__oauthlib.1fd52536"
COMMITS = {
    "oauthlib__oauthlib.1fd52536.combine_file__09vlzwgc": "3b1e6ec37ffeacc5bed0e55e287f86166b4db21c",  # pragma: allowlist secret - original public task Git object
    "oauthlib__oauthlib.1fd52536.combine_file__0fceycuu": "a480f4738267579b0f93cb0dbf2ea2389b5e44b0",  # pragma: allowlist secret - original public task Git object
    "oauthlib__oauthlib.1fd52536.combine_file__0fukhdzk": "ea82f1c5b3539878f309ec79e0d16092e743870b",  # pragma: allowlist secret - original public task Git object
}


def git_object(kind, raw):
    payload = kind.encode() + b" " + str(len(raw)).encode() + b"\0" + raw
    return hashlib.sha1(payload).hexdigest(), payload


def commit_bytes(metadata, expected):
    if metadata.get("verification", {}).get("signature"):
        raise ValueError("signed commit reconstruction is not supported")
    lines = ["tree " + metadata["tree"]["sha"]]
    lines += ["parent " + parent["sha"] for parent in metadata["parents"]]
    for who in ["author", "committer"]:
        person = metadata[who]
        stamp = int(
            datetime.fromisoformat(person["date"].replace("Z", "+00:00")).timestamp()
        )
        lines.append(f"{who} {person['name']} <{person['email']}> {stamp} +0000")
    for suffix in ["", "\n"]:
        raw = ("\n".join(lines) + "\n\n" + metadata["message"] + suffix).encode()
        if git_object("commit", raw)[0] == expected:
            return raw
    raise ValueError("publisher metadata did not reproduce the original commit ID")


def archive_objects(archive):
    root = {}
    objects = {}
    total = 0
    prefix = None
    with tarfile.open(archive) as tar:
        for member in tar:
            parts = PurePosixPath(member.name).parts
            if not parts or PurePosixPath(member.name).is_absolute() or ".." in parts:
                raise ValueError("unsafe archive path")
            if prefix is None:
                prefix = parts[0]
            if parts[0] != prefix:
                raise ValueError("single publisher archive root required")
            if member.isdir():
                continue
            if len(parts) < 2 or any(p in {".git", "."} for p in parts[1:]):
                raise ValueError("invalid Git tree path")
            if member.isfile():
                total += member.size
                if total > 32 * 1024 * 1024:
                    raise ValueError("fixed source archive exceeds unpacked budget")
                raw = tar.extractfile(member).read()
                mode = "100755" if member.mode & 0o111 else "100644"
            elif member.issym():
                raw, mode = member.linkname.encode(), "120000"
            else:
                raise ValueError("unsupported archive member")
            oid, payload = git_object("blob", raw)
            objects[oid] = payload
            node = root
            for part in parts[1:-1]:
                node = node.setdefault(part, {})
                if not isinstance(node, dict):
                    raise ValueError("colliding archive paths")
            if parts[-1] in node:
                raise ValueError("duplicate archive path")
            node[parts[-1]] = (mode, oid)

    def tree(node):
        entries = []
        for name, value in node.items():
            if isinstance(value, dict):
                mode, oid = "40000", tree(value)
                sort_name = name.encode() + b"/"
            else:
                mode, oid = value
                sort_name = name.encode()
            entries.append(
                (
                    sort_name,
                    mode.encode() + b" " + name.encode() + b"\0" + bytes.fromhex(oid),
                )
            )
        oid, payload = git_object(
            "tree", b"".join(value for _, value in sorted(entries))
        )
        objects[oid] = payload
        return oid

    return tree(root), objects


def recover(archive_dir, output, *, include_grading_parents=False):
    if output.exists():
        raise ValueError("fresh output required; inspect any previous attempt first")
    objects, commits, boundaries = {}, [], set()
    pending = list(COMMITS.items())
    for branch, expected in pending:
        url = f"https://api.github.com/repos/{REPOSITORY}/git/commits/{expected}"
        with urlopen(
            Request(url, headers={"User-Agent": "workload-native-delivery"}), timeout=30
        ) as response:
            metadata = json.loads(response.read(1024 * 1024))
        raw = commit_bytes(metadata, expected)
        tree, contents = archive_objects(archive_dir / (expected + ".tar.gz"))
        if tree != metadata["tree"]["sha"]:
            raise ValueError("archive tree differs from the original commit tree")
        objects.update(contents)
        objects[expected] = git_object("commit", raw)[1]
        commits.append({"branch": branch, "commit": expected, "tree": tree})
        if include_grading_parents and branch is not None:
            if len(metadata["parents"]) != 1:
                raise ValueError("fixed task requires exactly one grading parent")
            pending.append((None, metadata["parents"][0]["sha"]))
        else:
            boundaries.add(expected)
    subprocess.run(["git", "init", str(output)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(output), "config", "core.autocrlf", "false"], check=True
    )
    for oid, payload in objects.items():
        target = output / ".git/objects" / oid[:2] / oid[2:]
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(zlib.compress(payload))
    (output / ".git/shallow").write_text("\n".join(sorted(boundaries)) + "\n")
    for branch, oid in COMMITS.items():
        subprocess.run(
            ["git", "-C", str(output), "update-ref", "refs/heads/" + branch, oid],
            check=True,
        )
    subprocess.run(
        ["git", "-C", str(output), "checkout", next(iter(COMMITS))],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(output), "fsck", "--full", "--strict"],
        check=True,
        capture_output=True,
    )
    return {
        "status": "EXACT_ORIGINAL_SHALLOW_GIT_SNAPSHOTS_RECOVERED_NOT_EXECUTED",
        "commits": commits,
        "git_objects": len(objects),
        "grading_parents_recovered": include_grading_parents,
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-grading-parents", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            recover(
                args.archive_dir.resolve(strict=True),
                args.output.resolve(),
                include_grading_parents=args.include_grading_parents,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
