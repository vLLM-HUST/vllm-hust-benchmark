"""Run original SchemaStore validation, not a fabricated model benchmark."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

ROOT_TREE = "98c02d0c58f9cfee9eaa667af2bce08ee1111fc5"  # pragma: allowlist secret - original Git tree identity
REVISION = "b3c6a70f2a535d14e8ad2bf460b3aa8f5c0efb27"  # pragma: allowlist secret - original Git commit identity


def verify_tree(source, tree, expected_root=ROOT_TREE):
    """Recompute Git tree objects, not trust a supplied manifest's root label."""
    source = Path(source).resolve()
    if tree.get("truncated") or not isinstance(tree.get("tree"), list):
        raise ValueError("complete original recursive Git tree required")
    children = {}
    seen = set()
    blobs = []
    for item in tree["tree"]:
        path = PurePosixPath(item["path"])
        if (
            not path.parts
            or str(path) != item["path"]
            or path.is_absolute()
            or ".." in path.parts
            or str(path) in seen
        ):
            raise ValueError("distinct contained original tree paths required")
        seen.add(str(path))
        if (item["type"], item["mode"]) not in {
            ("tree", "040000"),
            ("blob", "100644"),
            ("blob", "100755"),
        }:
            raise ValueError("unexpected Git tree object or link")
        children.setdefault(str(path.parent), []).append(item)
        raw_path = source / str(path)
        if raw_path.is_symlink() or (item["type"] == "tree" and not raw_path.is_dir()):
            raise ValueError("original file/tree replaced by a filesystem link")
        if item["type"] == "blob":
            target = (source / str(path)).resolve()
            if not target.is_relative_to(source):
                raise ValueError("original file escapes source root")
            data = target.read_bytes()
            actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            if actual != item["sha"] or len(data) != item["size"]:
                raise ValueError("original file differs from pinned Git blob")
            blobs.append(str(path))

    directories = {item["path"] for item in tree["tree"] if item["type"] == "tree"}
    if any(parent != "." and parent not in directories for parent in children):
        raise ValueError("disconnected Git tree path")

    def digest(directory):
        entries = children.get(directory, [])
        body = b""
        for item in sorted(
            entries,
            key=lambda row: (
                PurePosixPath(row["path"]).name + ("/" if row["type"] == "tree" else "")
            ).encode(),
        ):
            name = PurePosixPath(item["path"]).name
            if item["type"] == "tree" and digest(item["path"]) != item["sha"]:
                raise ValueError("original nested Git tree differs")
            mode = "40000" if item["type"] == "tree" else item["mode"]
            body += (
                mode.encode()
                + b" "
                + name.encode()
                + b"\0"
                + bytes.fromhex(item["sha"])
            )
        return hashlib.sha1(f"tree {len(body)}\0".encode() + body).hexdigest()

    if digest(".") != expected_root:
        raise ValueError("original root Git tree differs")
    return len(blobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--original-tree", type=Path, required=True)
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "source_id": "SCHEMASTORE",
                    "status": "PLAN_ONLY",
                    "command": ["node", "cli.js", "check"],
                    "model_calls": 0,
                    "source_checkbox_eligible": False,
                }
            )
        )
        return
    source = args.source.resolve()
    output = args.output.resolve()
    if (
        output.exists()
        or output.is_relative_to(source)
        or source.is_relative_to(output)
    ):
        raise ValueError("fresh output disjoint from original source required")
    tree = json.loads(args.original_tree.read_text("utf-8"))
    count = verify_tree(source, tree)
    version = subprocess.check_output([str(args.node), "--version"], text=True).strip()
    if version != "v22.14.0":
        raise ValueError("qualified private Node22.14.0 required")
    output.mkdir(mode=0o700, parents=True)
    receipt = {
        "source_id": "SCHEMASTORE",
        "status": "STARTED",
        "source_revision": REVISION,
        "node": version,
        "verified_original_files_before": count,
        "original_command": ["node", "cli.js", "check"],
        "model_calls": 0,
        "source_checkbox_eligible": False,
        "performance_or_model_quality_claim": False,
    }
    target = output / "receipt-public.json"
    target.write_text(json.dumps(receipt, indent=2), "utf-8")
    try:
        with (output / "official-check.private.log").open("xb") as log:
            result = subprocess.run(
                [str(args.node), "cli.js", "check"],
                cwd=source,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1800,
            )
        receipt["exit_code"] = result.returncode
        receipt["verified_original_files_after"] = verify_tree(source, tree)
        receipt["status"] = (
            "ORIGINAL_SCHEMASTORE_CHECK_PASSED_HANDOFF_PENDING"
            if result.returncode == 0
            else "ORIGINAL_SCHEMASTORE_CHECK_FAILED_NOT_REPAIRED_OR_FILTERED"
        )
    except BaseException as error:
        receipt.update(
            status="EXECUTION_ERROR_NOT_COMPLETE", error_type=type(error).__name__
        )
        raise
    finally:
        target.write_text(json.dumps(receipt, indent=2), "utf-8")
    print(json.dumps(receipt))
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
