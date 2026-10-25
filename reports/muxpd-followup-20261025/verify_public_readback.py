"""Offline verification of preserved public readback receipts; no network/devices."""

import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(directory):
    directory = Path(directory)
    root = directory / "public-readback-v2"
    copies = read(root / "copy-map.json")["copies"]
    paths = [row["canonical_path"] for row in copies]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate canonical receipt")
    for row in copies:
        path = root / row["canonical_path"]
        if sha(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError(f"Readback receipt bytes changed: {path}")
    summary = read(root / "public-readback-receipt-v2.json")
    selection = read(root / "upload-selection-v2.json")
    inventory = read(root / "inventory-stability.json")
    first, last = [
        read(root / name) for name in ("release-api.json", "release-api-after.json")
    ]
    api_receipt = read(root / "release-api-receipt.json")
    if (
        summary["selection_sha256"] != sha(root / "upload-selection-v2.json")
        or summary["producer_sha256"] != sha(root / "verify_public_readback.py.source")
        or summary["builder_sha256"] != sha(directory / "tools/build_capsule.py")
        or inventory["receipt_sha256"] != sha(root / "public-readback-receipt-v2.json")
        or inventory["before_api_sha256"] != sha(root / "release-api.json")
        or inventory["after_api_sha256"] != sha(root / "release-api-after.json")
        or api_receipt["sha256"] != inventory["before_api_sha256"]
        or api_receipt["http_status"] != 200
    ):
        raise ValueError("Readback producer/API identity mismatch")
    if any(
        value.get("authentication") != "none"
        for value in (summary, inventory, api_receipt)
    ):
        raise ValueError("Anonymous public access was not established")
    fields = (
        "id",
        "name",
        "size",
        "state",
        "digest",
        "browser_download_url",
        "updated_at",
    )
    snapshots = [
        {row["name"]: {key: row.get(key) for key in fields} for row in value["assets"]}
        for value in (first, last)
    ]
    if (
        first["id"] != last["id"]
        or snapshots[0] != snapshots[1]
        or first["draft"]
        or last["draft"]
    ):
        raise ValueError("First/last public asset inventories differ")
    expected = {Path(row["path"]).name: row for row in selection["files"]}
    observed = {row["name"]: row for row in summary["assets"]}
    if (
        len(expected) != 21
        or set(expected) != set(observed)
        or set(expected) != set(snapshots[0])
    ):
        raise ValueError("Public asset inventory is incomplete")
    for name, row in expected.items():
        asset = observed[name]
        receipt = read(root / "receipts" / (name + ".json"))
        api = snapshots[0][name]
        if (
            not asset["verified"]
            or not receipt["verified"]
            or receipt["authentication"] != "none"
            or receipt["http_status"] != 200
            or any(
                value["sha256"] != row["sha256"] or value["bytes"] != row["bytes"]
                for value in (asset, receipt)
            )
            or any(
                value["requested_url"] != row["planned_url"]
                for value in (asset, receipt)
            )
            or api["size"] != row["bytes"]
            or api["state"] != "uploaded"
            or api["browser_download_url"] != row["planned_url"]
            or api["digest"] not in (None, "sha256:" + row["sha256"])
        ):
            raise ValueError(f"Public download identity mismatch: {name}")
    index = read(directory / "archive-index-v2.json")
    if summary["public_index_sha256"] != sha(directory / "archive-index-v2.json"):
        raise ValueError("Public index differs from preserved canonical index")
    by_id = {row["capsule_id"]: row for row in summary["capsules"]}
    members = 0
    for capsule in index["capsules"]:
        cid = capsule["capsule_id"]
        receipt = read(root / "archive-verification" / (cid + ".json"))
        proof = json.loads(receipt["stdout"])
        manifest = directory / "custody" / (cid + ".manifest.json")
        count = len(read(manifest)["members"])
        if (
            receipt["returncode"] != 0
            or not receipt["verified"]
            or not proof["verified"]
            or receipt["manifest_sha256"] != sha(manifest)
            or receipt["builder_sha256"] != summary["builder_sha256"]
            or not by_id[cid]["verified"]
            or by_id[cid]["members_verified"] != count
            or proof["members_verified"] != count
            or proof["archive"]["sha256"] != capsule["archive"]["sha256"]
            or proof["archive"]["bytes"] != capsule["archive"]["bytes"]
        ):
            raise ValueError(f"Capsule member proof differs: {cid}")
        members += count
    if len(by_id) != 13 or members != 1882 or summary["asset_bytes"] != 725765919:
        raise ValueError("Bounded readback scope changed")
    return {
        "passed": True,
        "public_assets": 21,
        "public_bytes": 725765919,
        "capsules": 13,
        "members_verified": members,
        "anonymous_access": True,
        "first_last_inventory_identical": True,
        "quality_or_performance_qualified": False,
        "scope": "Offline custody of successful timestamped anonymous readback; not a new live request or model acceptance.",
    }


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
