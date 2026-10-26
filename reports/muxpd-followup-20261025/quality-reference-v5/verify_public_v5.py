"""Validate saved v5 anonymous readback custody only; no network or archive execution."""

import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def file_identity(path, row):
    data = path.read_bytes()
    require(
        len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"],
        "Frozen readback bytes differ",
    )


def verify(root):
    root = Path(root)
    public = root / "public-readback-v5"
    mapping = read(public / "copy-map.json")
    selection_path = public / "small-copy-selection-v5.json"
    require(
        hashlib.sha256(selection_path.read_bytes()).hexdigest()
        == mapping["source_selection"]["sha256"],
        "Selection identity differs",
    )
    selection = read(selection_path)
    copies = mapping["copies"]
    require(len(copies) == len(selection["files"]) == 50, "Readback copy count differs")
    require(
        sum(r["bytes"] for r in copies) == 3424951, "Readback metadata size differs"
    )
    require(
        len({r["canonical_relative_path"] for r in copies}) == len(copies),
        "Duplicate destination",
    )
    by_source = {r["path"]: public / r["canonical_relative_path"] for r in copies}
    require(
        len(by_source) == 50
        and set(by_source) == {r["path"] for r in selection["files"]},
        "Copy source set differs",
    )
    for row in selection["files"]:
        file_identity(by_source[row["path"]], row)
    for row in copies:
        file_identity(by_source[row["path"]], row)

    def bound(row):
        path = by_source[row["path"]]
        file_identity(path, row)
        return path

    result = read(bound(selection["completion_receipt"]))
    require(
        result["passed"] and result["authentication"] == "none",
        "Anonymous readback did not pass",
    )
    require(
        (
            result["asset_count"],
            result["asset_bytes"],
            result["capsule_count"],
            result["original_members_verified"],
            result["exact_embedded_manifests_verified"],
            result["support_members_verified"],
        )
        == (13, 209021252, 8, 2463, 8, 1166),
        "Public scope changed",
    )
    require(
        result["all_asset_sizes_sha256_verified"]
        and result["stable_public_inventory"]
        and result["after_api_request_verified"]
        and result["all_checks_before_after_api"],
        "Missing full verification",
    )
    bound(result["producer"])
    api = [
        read(bound(result[key]))
        for key in [
            "api_before",
            "api_after",
            "old_v4_public_api_snapshot",
            "old_preupload_api_snapshot",
        ]
    ]
    fields = result["immutable_api_fields"]

    def inventory(value):
        rows = value["assets"]
        require(len({r["name"] for r in rows}) == len(rows), "Duplicate public asset")
        return {r["name"]: {key: r.get(key) for key in fields} for r in rows}

    before, after, old, prior = map(inventory, api)
    require(len({v["id"] for v in api}) == 1, "Release identity differs")
    require(
        len(before) == len(after) == 70 and len(old) == len(prior) == 57,
        "API scope differs",
    )
    require(
        before == after
        and old == prior
        and all(before.get(k) == v for k, v in old.items()),
        "Stable API identities differ",
    )
    upload = read(root / "upload-selection-v5.json")
    expected = {Path(r["path"]).name: r for r in upload["files"]}
    require(
        len(expected) == 13 and set(before) - set(old) == set(expected),
        "Wrong new public assets",
    )
    require(
        len(result["assets"]) == 13
        and {r["name"] for r in result["assets"]} == set(expected),
        "Missing downloaded asset",
    )
    original_attempt = Path(selection["completion_receipt"]["path"]).parent
    for row in result["assets"]:
        entry = expected[row["name"]]
        require(
            row["verified"]
            and row["bytes"] == entry["bytes"]
            and row["sha256"] == entry["sha256"]
            and row["requested_url"] == entry["planned_url"],
            "Downloaded identity differs",
        )
        metadata = before[row["name"]]
        require(
            metadata["digest"] == "sha256:" + entry["sha256"]
            and metadata["state"] == "uploaded"
            and metadata["size"] == entry["bytes"],
            "Public asset digest/state differs",
        )
        http = read(
            by_source[str(original_attempt / "receipts" / (row["name"] + ".json"))]
        )
        require(
            http["verified"]
            and http["http_status"] == 200
            and http["authentication"] == "none"
            and http["method"] == "GET",
            "Anonymous HTTP receipt missing",
        )
        require(
            http["sha256"] == entry["sha256"] and http["bytes"] == entry["bytes"],
            "HTTP bytes differ",
        )
    index_path = root / "archive-index-v5.json"
    file_identity(index_path, result["public_index"])
    index = read(index_path)
    capsule_rows = {r["capsule_id"]: r for r in index["capsules"]}
    require(
        len(result["capsules"]) == 8
        and {r["capsule_id"] for r in result["capsules"]} == set(capsule_rows),
        "Incomplete capsule set",
    )
    total = 0
    for row in result["capsules"]:
        name = row["capsule_id"]
        proof = read(by_source[str(original_attempt / "archives" / (name + ".json"))])
        require(
            {k: v for k, v in proof.items() if k != "member_results"} == row,
            "Capsule receipt and final summary differ",
        )
        require(
            proof["verified"]
            and proof["complete_member_set"]
            and proof["exact_embedded_manifest_verified"],
            "Incomplete member proof",
        )
        manifest_path = root / "custody" / (name + ".manifest.json")
        manifest = read(manifest_path)
        require(
            hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            == proof["embedded_manifest_sha256"],
            "Embedded manifest differs",
        )
        downloaded_manifest = by_source[
            str(original_attempt / "manifests" / (name + ".json"))
        ]
        require(
            downloaded_manifest.read_bytes() == manifest_path.read_bytes(),
            "Readback manifest bytes differ",
        )
        require(
            all(
                proof["archive"][k] == capsule_rows[name]["archive"][k]
                for k in ("bytes", "sha256")
            ),
            "Archive versus canonical index differs",
        )
        members = {r["member"]: r for r in proof["member_results"]}
        require(
            len(members)
            == len(proof["member_results"])
            == len(manifest["members"])
            == proof["members_verified"],
            "Member count differs",
        )
        require(
            set(members) == {r["member"] for r in manifest["members"]},
            "Member set differs",
        )
        for member in manifest["members"]:
            require(
                member["type"] == "file"
                and all(
                    members[member["member"]][key] == member[key]
                    for key in ("type", "bytes", "sha256")
                ),
                "Regular member bytes differ",
            )
        total += proof["members_verified"]
    require(total == 2463, "Missing full member coverage")
    return {
        "passed": True,
        "copied_files": 50,
        "copied_bytes": 3424951,
        "assets": 13,
        "asset_bytes": 209021252,
        "capsules": 8,
        "original_members": 2463,
        "embedded_manifests": 8,
        "support_members": 1166,
        "public_inventory": 70,
        "old57_bytes_redownloaded": False,
        "scope": "Offline custody of the timestamped anonymous readback; no new network or experimental acceptance.",
    }


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
