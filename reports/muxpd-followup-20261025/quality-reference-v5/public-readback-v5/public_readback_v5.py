"""Anonymous v5 GET readback; tar contents are hashed, never extracted/executed."""

import argparse
import concurrent.futures
import re
import hashlib
import io
import json
import tarfile
import traceback
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent / "public-readback-v5"
PUBLICATION = ROOT.parent
SELECTION_SHA = "89a59924b1950fa1e37e1c4b8e47f03090c7456b48be0ad74700ab6acc7c8b08"
INDEX_SHA = "3c3d18ef355518c4ea7a679e30cd7f5c72c31d1ba1a1b528e8336a6de35aab9b"
API = "https://api.github.com/repos/vLLM-HUST/vllm-hust-benchmark/releases/tags/evidence-muxpd-followup-20261025"
OLD_API = PUBLICATION / "public-readback-v4/attempt-01/release-api-after.json"
OLD_API_SHA = "7ff2bb6b719eee3014fd7aaf331a4e33c1d183aee771ab282e490ea2461d6c21"
PREUPLOAD_API = PUBLICATION / "prior57-release-api-v5.json"
PREUPLOAD_API_SHA = "a4c6d7a77048354ac9bff7894f2e635698f5e4a2ede8a09507cd16a6548c0a83"
STABLE_FIELDS = ("id", "node_id", "name", "label", "content_type", "state", "size", "digest", "created_at", "updated_at", "browser_download_url")


def now():
    return datetime.now(timezone.utc).isoformat()


def require(value, message):
    if not value:
        raise ValueError(message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def identity(path):
    data = Path(path).read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": sha_bytes(data)}


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def fetch(url, target, receipt_path, expected=None):
    receipt = {"requested_url": url, "authentication": "none", "method": "GET", "started_at": now(), "verified": False}
    total, digest = 0, hashlib.sha256()
    try:
        # No GitHub authentication, netrc, gh CLI, or auth handler. Standard unauthenticated network proxy only.
        is_api = urllib.parse.urlsplit(url).hostname == "api.github.com"
        receipt["transport"] = "direct-no-proxy" if is_api else "standard-environment-proxy"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}) if is_api else urllib.request.ProxyHandler())
        request = urllib.request.Request(url, headers={"User-Agent": "MuxPD-independent-anonymous-readback/5", "Accept": "application/vnd.github+json" if url.startswith("https://api.github.com/") else "application/octet-stream"})
        with opener.open(request, timeout=30) as response, target.open("xb") as output:
            final = urllib.parse.urlsplit(response.url)
            receipt.update(http_status=response.status, final_host=final.hostname, final_path=final.path, headers={k: response.headers.get(k) for k in ("Date", "ETag", "Last-Modified", "Content-Length", "Content-Type", "X-GitHub-Request-Id")})
            require(response.status == 200 and final.scheme == "https", "Expected HTTPS HTTP200")
            while block := response.read(1024 * 1024):
                output.write(block)
                total += len(block)
                digest.update(block)
        receipt.update(bytes=total, sha256=digest.hexdigest())
        if expected is not None:
            receipt["expected"] = {k: expected[k] for k in ("bytes", "sha256")}
            require(total == expected["bytes"] and digest.hexdigest() == expected["sha256"], "Downloaded bytes differ from pinned selection")
        receipt["verified"] = True
    except Exception:
        receipt.update(bytes=total, sha256=digest.hexdigest(), error=traceback.format_exc())
    receipt["finished_at"] = now()
    write_new(receipt_path, receipt)
    return receipt


def stable_inventory(api):
    rows = api["assets"]
    require(len({r["name"] for r in rows}) == len(rows), "Duplicate public asset name")
    return {r["name"]: {k: r.get(k) for k in STABLE_FIELDS} for r in rows}


def safe_name(value):
    path = PurePosixPath(value)
    require(value and not path.is_absolute() and ".." not in path.parts and "\\" not in value, "Unsafe archive member")
    require(path.as_posix() == value.rstrip("/"), "Noncanonical archive member")
    return value.rstrip("/")


def verify_tar(archive, members, manifest_bytes=None, capture=(), allow_dirs=False):
    expected = {r["member"]: r for r in members}
    require(len(expected) == len(members), "Duplicate expected member")
    seen, directory_seen, retained, details = set(), set(), {}, []
    allowed_dirs = set()
    if allow_dirs:
        for name in expected:
            allowed_dirs.update(str(p) for p in PurePosixPath(name).parents if str(p) != ".")
    source = io.BytesIO(archive) if isinstance(archive, bytes) else None
    with tarfile.open(fileobj=source, name=None if source else str(archive), mode="r:*") as tar:
        for member in tar:
            name = safe_name(member.name)
            require(name not in seen and name not in directory_seen, "Duplicate tar member")
            if member.isdir():
                require(allow_dirs and name in allowed_dirs, "Unexpected directory")
                directory_seen.add(name)
                continue
            seen.add(name)
            if name == "capsule-manifest.json" and manifest_bytes is not None:
                require(member.isfile(), "Embedded manifest is not regular")
                data = tar.extractfile(member).read()
                require(data == manifest_bytes, "Embedded manifest differs from public-index pin")
                continue
            require(name in expected, "Unexpected archive member: " + name)
            row = expected[name]
            if row["type"] == "symlink":
                require(member.issym() and member.linkname == row["link_target"], "Symlink mismatch")
                link_bytes = member.linkname.encode()
                require(len(link_bytes) == row["bytes"] and sha_bytes(link_bytes) == row["sha256"], "Symlink bytes/hash mismatch")
                details.append({"member": name, "type": "symlink", "link_target": member.linkname, "bytes": len(link_bytes), "sha256": sha_bytes(link_bytes)})
                continue
            require(row["type"] == "file" and member.isfile() and member.size == row["bytes"], "Member type/size mismatch")
            count, digest, chunks = 0, hashlib.sha256(), []
            stream = tar.extractfile(member)
            while block := stream.read(1024 * 1024):
                count += len(block)
                digest.update(block)
                if name in capture:
                    chunks.append(block)
            require(count == row["bytes"] and digest.hexdigest() == row["sha256"], "Member bytes mismatch: " + name)
            details.append({"member": name, "type": "file", "bytes": count, "sha256": digest.hexdigest()})
            if name in capture:
                retained[name] = b"".join(chunks)
    require(seen == set(expected) | ({"capsule-manifest.json"} if manifest_bytes is not None else set()), "Incomplete archive member set")
    require(not allow_dirs or directory_seen == allowed_dirs, "Incomplete nested directory set")
    return {"verified": True, "members_verified": len(expected), "directories_verified": len(directory_seen), "complete_member_set": True, "member_results": details}, retained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", default="attempt-01")
    args = parser.parse_args()
    require(re.fullmatch(r"attempt-[0-9]{2,}", args.attempt), "Explicit fresh attempt name required")
    ROOT.mkdir(exist_ok=True)
    attempt = ROOT / args.attempt
    attempt.mkdir(exist_ok=False)
    for name in ("assets", "receipts", "archives", "manifests"):
        (attempt / name).mkdir()
    result = {
        "schema": "muxpd-anonymous-public-readback-v5", "started_at": now(),
        "authentication": "none", "passed": False, "producer": identity(Path(__file__)),
        "scope": "Anonymous asset bytes and complete archive-member custody only. No extraction, execution, GitHub mutation, device action, or experimental acceptance.",
    }
    try:
        selection_path = PUBLICATION / "upload-selection-v5.json"
        require(identity(selection_path)["sha256"] == SELECTION_SHA, "Selection pin differs")
        selection = json.loads(selection_path.read_bytes())
        require(len(selection["files"]) == 13 and sum(r["bytes"] for r in selection["files"]) == 209021252, "Expected 13 assets / 209021252 bytes")
        for source, name, pin in (
            (selection_path, "upload-selection-v5.json", SELECTION_SHA),
            (OLD_API, "previous57-v4-public-api.json", OLD_API_SHA),
            (PREUPLOAD_API, "previous57-preupload-api.json", PREUPLOAD_API_SHA),
        ):
            data = source.read_bytes()
            require(sha_bytes(data) == pin, "Pinned metadata identity differs")
            with (attempt / name).open("xb") as stream:
                stream.write(data)
        result["selection"] = identity(attempt / "upload-selection-v5.json")
        before_receipt = fetch(API, attempt / "release-api-before.json", attempt / "release-api-before-receipt.json")
        require(before_receipt["verified"], "Anonymous before-API failed; receipt retained")
        before = json.loads((attempt / "release-api-before.json").read_bytes())
        old = json.loads((attempt / "previous57-v4-public-api.json").read_bytes())
        preupload = json.loads((attempt / "previous57-preupload-api.json").read_bytes())
        inventory, old_inventory, preupload_inventory = map(stable_inventory, (before, old, preupload))
        require(before["id"] == old["id"] == preupload["id"], "Release identity differs")
        require(len(inventory) == 70 and len(old_inventory) == len(preupload_inventory) == 57, "Expected 70 current and 57 prior assets")
        require(old_inventory == preupload_inventory, "Prior v4 versus preupload57 stable identities differ")
        require(all(inventory.get(name) == row for name, row in old_inventory.items()), "Old57 point-in-time identity metadata changed")
        selected_names = {Path(row["path"]).name for row in selection["files"]}
        require(len(selected_names) == 13 and set(inventory) - set(old_inventory) == selected_names, "Expected exactly the selected13 additions")
        for row in selection["files"]:
            asset = inventory[Path(row["path"]).name]
            require(asset["size"] == row["bytes"] and asset["digest"] == "sha256:" + row["sha256"]
                    and asset["browser_download_url"] == row["planned_url"] and asset["state"] == "uploaded",
                    "Public API selection mismatch")

        def download(row):
            name = Path(row["path"]).name
            receipt = fetch(row["planned_url"], attempt / "assets" / name,
                            attempt / "receipts" / (name + ".json"), row)
            print(json.dumps({"asset": name, "verified": receipt["verified"], "bytes": receipt.get("bytes")}), flush=True)
            return {"name": name, **receipt}

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            downloads = list(pool.map(download, selection["files"]))
        write_new(attempt / "download-summary.json", {"all_verified": all(r["verified"] for r in downloads), "assets": downloads})
        require(all(r["verified"] for r in downloads), "Download failure retained; no retry/overwrite")
        index_path = attempt / "assets/publication-index-v5.json"
        require(identity(index_path)["sha256"] == INDEX_SHA, "Downloaded public index pin differs")
        index = json.loads(index_path.read_bytes())
        require(len(index["capsules"]) == 8 and index["counts"]["original_members"] == 2463
                and index["counts"]["terminal_runs"] == 7, "Index counts differ")
        require(sum(c["kind"] == "terminal-run" for c in index["capsules"]) == 7
                and sum(c["kind"] == "explicit-support" for c in index["capsules"]) == 1, "Capsule types differ")
        capsule_results = []
        support_count = 0
        for capsule in index["capsules"]:
            archive = attempt / "assets" / Path(capsule["archive"]["path"]).name
            archive_id = identity(archive)
            require(all(archive_id[k] == capsule["archive"][k] for k in ("bytes", "sha256")), "Archive versus public-index identity mismatch")
            manifest_bytes = Path(capsule["manifest"]["path"]).read_bytes()
            require(len(manifest_bytes) == capsule["manifest"]["bytes"]
                    and sha_bytes(manifest_bytes) == capsule["manifest"]["sha256"], "Local manifest differs from downloaded public-index pin")
            manifest = json.loads(manifest_bytes)
            require(manifest["capsule_id"] == capsule["capsule_id"]
                    and len(manifest["members"]) == capsule["original_members"], "Manifest capsule/count mismatch")
            with (attempt / "manifests" / (capsule["capsule_id"] + ".json")).open("xb") as stream:
                stream.write(manifest_bytes)
            verified, _ = verify_tar(archive, manifest["members"], manifest_bytes)
            require(sum(r.get("bytes", 0) for r in verified["member_results"]) == capsule["original_bytes"], "Original member byte total differs")
            verified.update(capsule_id=capsule["capsule_id"], kind=capsule["kind"], archive=archive_id,
                            embedded_manifest_sha256=sha_bytes(manifest_bytes), exact_embedded_manifest_verified=True)
            if capsule["kind"] == "explicit-support":
                require(capsule["capsule_id"] == "muxpd-20261026-support-v5" and verified["members_verified"] == 1166, "Support identity/count differs")
                support_count = 1166
            write_new(attempt / "archives" / (capsule["capsule_id"] + ".json"), verified)
            capsule_results.append({k: v for k, v in verified.items() if k != "member_results"})
            print(json.dumps({"capsule": capsule["capsule_id"], "members": verified["members_verified"], "verified": True}), flush=True)
        require(sum(r["members_verified"] for r in capsule_results) == 2463 and support_count == 1166, "Incomplete member verification")
        result.update(
            asset_count=13, asset_bytes=sum(r["bytes"] for r in downloads), all_asset_sizes_sha256_verified=True,
            assets=[{k: r[k] for k in ("name", "requested_url", "bytes", "sha256", "verified")} for r in downloads],
            public_index=identity(index_path), capsule_count=8, original_members_verified=2463,
            exact_embedded_manifests_verified=8, support_members_verified=1166, capsules=capsule_results,
            old_v4_public_api_snapshot=identity(attempt / "previous57-v4-public-api.json"),
            old_preupload_api_snapshot=identity(attempt / "previous57-preupload-api.json"),
            old57_identity_comparison="Stable public API identity fields match both prior57 snapshots. Old asset bytes were not downloaded again; no continuous-immutability claim between snapshots.",
            immutable_api_fields=list(STABLE_FIELDS), all_checks_before_after_api=True,
        )
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        after_receipt = fetch(API, attempt / "release-api-after.json", attempt / "release-api-after-receipt.json")
        result["after_api_request_verified"] = after_receipt["verified"]
        try:
            require(after_receipt["verified"], "Anonymous after-API failed; receipt retained")
            before = json.loads((attempt / "release-api-before.json").read_bytes())
            after = json.loads((attempt / "release-api-after.json").read_bytes())
            require(before["id"] == after["id"] and len(after["assets"]) == 70
                    and stable_inventory(before) == stable_inventory(after), "Public stable inventory drift")
            result.update(api_before=identity(attempt / "release-api-before.json"), api_after=identity(attempt / "release-api-after.json"),
                          public_inventory_count_before=70, public_inventory_count_after=70,
                          stable_public_inventory=True, dynamic_fields_excluded=["download_count", "uploader"])
            result["passed"] = "error" not in result and result.get("all_checks_before_after_api", False)
        except Exception:
            result["after_api_error"] = traceback.format_exc()
        result["finished_at"] = now()
        write_new(attempt / "public-readback-receipt-v5.json", result)
    print(json.dumps({"passed": result["passed"], "receipt": str(attempt / "public-readback-receipt-v5.json")}), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

