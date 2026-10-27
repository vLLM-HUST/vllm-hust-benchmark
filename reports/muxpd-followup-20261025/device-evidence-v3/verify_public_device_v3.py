"""Offline custody of v3 anonymous readback, including failed attempts; no network."""

import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root):
    root = Path(root)
    public = root / "public-readback-v3"
    mapping = read(public / "copy-map.json")["copies"]
    by_source = {}
    for row in mapping:
        path = public / row["canonical_relative_path"]
        data = path.read_bytes()
        require(
            len(data) == row["bytes"]
            and hashlib.sha256(data).hexdigest() == row["sha256"],
            "Copied readback bytes differ",
        )
        by_source[row["path"]] = path
    selection = read(public / "small-copy-selection-v3.json")
    require(
        len(mapping) == len(selection["files"]) == 67,
        "Small metadata selection differs",
    )
    require(
        sum(row["bytes"] for row in mapping) == 762914,
        "Small metadata byte accounting differs",
    )

    def bound(item):
        path = by_source[item["path"]]
        data = path.read_bytes()
        require(
            len(data) == item["bytes"]
            and hashlib.sha256(data).hexdigest() == item["sha256"],
            "Receipt-bound source differs",
        )
        return path

    result = read(bound(selection["completion_receipt"]))
    require(
        result["passed"] and result["authentication"] == "none",
        "Anonymous completion did not pass",
    )
    require(
        (
            result["asset_count"],
            result["asset_bytes"],
            result["capsule_count"],
            result["original_members_verified"],
        )
        == (16, 65585167, 10, 786),
        "Readback scope changed",
    )
    require(
        result["all_asset_sizes_sha256_verified"]
        and result["all_capsule_member_sets_and_hashes_verified"],
        "Missing complete checks",
    )
    for item in result["failed_attempts_preserved"]:
        require(read(bound(item))["passed"] is False, "Earlier failure was relabeled")
    require(
        len(result["failed_attempts_preserved"]) == 2, "Failed attempt history missing"
    )
    bound(result["producer"])
    before, after, old = [
        read(bound(result[key]))
        for key in ("before_api", "after_api", "old23_public_api_snapshot")
    ]
    fields = result["public_inventory_stable_fields"]

    def inventory(value):
        return {
            row["name"]: {key: row[key] for key in fields} for row in value["assets"]
        }

    previous, first, last = map(inventory, (old, before, after))
    require(
        len(previous) == 23 and len(first) == len(last) == 39 and first == last,
        "Public inventory differs",
    )
    require(
        all(first.get(name) == row for name, row in previous.items()),
        "Previous23 stable API identities differ",
    )
    upload = read(root / "upload-selection-v3.json")
    expected = {Path(row["path"]).name: row for row in upload["files"]}
    require(set(first) - set(previous) == set(expected), "New asset inventory differs")
    require(
        {row["name"] for row in result["assets"]} == set(expected),
        "Downloaded asset coverage differs",
    )
    for row in result["assets"]:
        entry = expected[row["name"]]
        require(
            row["verified"]
            and row["bytes"] == entry["bytes"]
            and row["sha256"] == entry["sha256"],
            "Downloaded identity differs from frozen selection",
        )
        require(
            first[row["name"]]["digest"] == "sha256:" + entry["sha256"]
            and first[row["name"]]["state"] == "uploaded",
            "Public API digest/state differs",
        )
    index = read(root / "archive-index-v3.json")
    capsule_by_id = {row["capsule_id"]: row for row in index["capsules"]}
    total = 0
    for item in result["capsule_verification_receipts"]:
        receipt = read(bound(item))
        capsule = capsule_by_id[receipt["capsule_id"]]
        manifest_path = root / "custody" / (receipt["capsule_id"] + ".manifest.json")
        manifest = read(manifest_path)
        require(
            receipt["verified"] and receipt["complete_member_set"],
            "Incomplete capsule verification",
        )
        require(
            receipt["archive"]["sha256"] == capsule["archive"]["sha256"],
            "Capsule archive identity differs",
        )
        require(
            receipt["embedded_manifest_sha256"]
            == hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "Embedded manifest differs",
        )
        actual = {row["member"]: row for row in receipt["member_results"]}
        require(
            set(actual) == {row["member"] for row in manifest["members"]},
            "Verified member set differs",
        )
        for row in manifest["members"]:
            checked = actual[row["member"]]
            keys = (
                ("type", "link_target")
                if row["type"] == "symlink"
                else ("type", "bytes", "sha256")
            )
            require(
                all(row[key] == checked[key] for key in keys),
                "Verified member identity differs",
            )
        total += receipt["members_verified"]
    require(
        total == 786 and len(result["capsule_verification_receipts"]) == 10,
        "Missing capsule coverage",
    )
    nested = read(bound(result["nested_verification"]))
    require(
        nested["verified"]
        and nested["complete_member_set"]
        and nested["members_verified"] == 99
        and nested["directories_verified"] == 9,
        "Incomplete nested source proof",
    )
    require(
        nested["archive_sha256"] == result["nested_source98"]["archive_sha256"],
        "Nested source archive differs",
    )
    return {
        "passed": True,
        "copies": 67,
        "public_assets": 16,
        "public_bytes": 65585167,
        "capsules": 10,
        "outer_members": 786,
        "nested_source_files": 99,
        "public_inventory": 39,
        "failed_attempts_preserved": 2,
        "old23_bytes_redownloaded": False,
        "scope": "Offline custody of timestamped anonymous completion, not new network or experimental acceptance.",
    }


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
