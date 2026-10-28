"""Bounded anonymous verification of the small 24-case quality Release only."""

import argparse
from datetime import UTC, datetime
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import traceback
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import getproxies

HERE = Path(__file__).resolve().parent
MAX_ASSET_BYTES = 2_000_000
CHUNK = 1024 * 1024


def now():
    return datetime.now(UTC).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    h, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while block := stream.read(CHUNK):
            h.update(block)
            size += len(block)
    return {"bytes": size, "sha256": h.hexdigest()}


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def final_headers(text):
    blocks = [block for block in re.split(r"\r?\n\r?\n", text) if block.startswith("HTTP/")]
    require(blocks, "No final HTTP headers")
    return {key.strip().title(): value.strip()
            for line in blocks[-1].splitlines()[1:] if ":" in line
            for key, value in [line.split(":", 1)]}


def fetch(base, expected, directory):
    name = expected["name"]
    require(Path(name).name == name and 0 < expected["bytes"] <= MAX_ASSET_BYTES,
            "This verifier accepts only named small assets")
    partial, final = directory / (name + ".partial"), directory / name
    headers_path = directory / (name + ".headers")
    url = base + "/" + quote(name, safe="")
    row = {"name": name, "public_url": url, "expected": expected,
           "started_at": now(), "passed": False, "http_status": None, "error": None,
           "anonymous": True, "authorization_header": False, "cookie_header": False,
           "curlrc_enabled": False, "netrc_enabled": False, "range_requested": False,
           "certificate_verification_enabled": True,
           "transport": "curl --disable --http1.1 --tlsv1.2 --tls-max 1.2"}
    try:
        command = ["curl", "--disable", "--http1.1", "--tlsv1.2", "--tls-max", "1.2",
                   "--proto", "=https", "--proto-redir", "=https", "--location",
                   "--connect-timeout", "15", "--max-time", "45", "--retry", "0",
                   "--max-filesize", str(expected["bytes"]), "--silent", "--show-error",
                   "--header", "Accept-Encoding: identity", "--header", "Cache-Control: no-cache",
                   "--dump-header", str(headers_path), "--output", str(partial),
                   "--write-out", "%{http_code}\n%{url_effective}\n%{ssl_verify_result}\n%{http_version}\n", url]
        result = subprocess.run(command, capture_output=True, text=True, timeout=50)
        row.update(curl_exit_code=result.returncode, curl_stderr=result.stderr)
        lines = result.stdout.splitlines()
        require(len(lines) >= 4, "Missing curl transfer metadata")
        row["http_status"] = int(lines[0]) if lines[0] != "000" else None
        response = urlsplit(lines[1])
        row["response_url_without_query"] = urlunsplit((response.scheme, response.netloc, response.path, "", ""))
        row["ssl_verify_result"], row["http_version"] = int(lines[2]), lines[3]
        require(result.returncode == 0, "curl transfer failed")
        require(row["http_status"] == 200 and response.scheme == "https" and int(lines[2]) == 0,
                "Expected anonymous HTTPS200 with verified certificate")
        headers = final_headers(headers_path.read_text())
        row["response_headers"] = {key: headers.get(key) for key in ("Content-Length", "Content-Type", "ETag")}
        if headers.get("Content-Length") is not None:
            require(int(headers["Content-Length"]) == expected["bytes"], "HTTP size differs")
        actual = identity(partial)
        row["downloaded"] = actual
        require(actual == {key: expected[key] for key in ("bytes", "sha256")}, "Downloaded size/SHA256 differs")
        partial.rename(final)
        row["passed"] = True
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
        if partial.exists():
            row["downloaded"] = identity(partial)
    row["finished_at"] = now()
    save(directory.parent / (name + ".http.json"), row)
    print(f"{now()} {name}: HTTP {row['http_status']} passed={row['passed']}", flush=True)
    return row


def safe_member(name):
    path = PurePosixPath(name)
    require(name and not path.is_absolute() and ".." not in path.parts and str(path) == name,
            "Unsafe or noncanonical tar member")


def verify_archive(archive_path, manifest_path, wanted_archive_sha, expected_original_count):
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    require(identity(archive_path)["sha256"] == wanted_archive_sha, "Public archive digest changed")
    require(manifest["original_member_count"] == expected_original_count, "Original member count changed")
    wanted = {}
    for row in manifest["members"]:
        name = row["member"]
        safe_member(name)
        require(name not in wanted, "Duplicate manifest member")
        wanted[name] = {"bytes": row["bytes"], "sha256": row["sha256"]}
    require(len(wanted) == expected_original_count, "Manifest payload inventory differs")
    embedded = manifest["embedded_manifest_member"]
    safe_member(embedded)
    require(embedded not in wanted, "Embedded manifest collides with payload")
    wanted[embedded] = {"bytes": len(manifest_bytes), "sha256": hashlib.sha256(manifest_bytes).hexdigest()}
    seen, total, member_receipts = set(), 0, []
    with gzip.open(archive_path, "rb") as decoded:
        with tarfile.open(fileobj=decoded, mode="r|", bufsize=CHUNK) as archive:
            for member in archive:
                safe_member(member.name)
                require(member.isfile() and member.name in wanted and member.name not in seen,
                        "Unexpected/duplicate/nonregular public tar member")
                expected = wanted[member.name]
                require(member.size == expected["bytes"], "Tar header size differs")
                h, count = hashlib.sha256(), 0
                with archive.extractfile(member) as stream:
                    while block := stream.read(CHUNK):
                        h.update(block)
                        count += len(block)
                require(count == expected["bytes"] and h.hexdigest() == expected["sha256"],
                        "Public tar member SHA/size differs: " + member.name)
                seen.add(member.name)
                total += count
                member_receipts.append({"member": member.name, "bytes": count, "sha256": h.hexdigest()})
        while block := decoded.read(CHUNK):
            require(not any(block), "Unexpected nonzero data after tar end")
    require(seen == set(wanted), "Public archive missing declared members")
    return {"passed": True, "public_archive": identity(archive_path),
            "public_manifest": identity(manifest_path), "payload_members": expected_original_count,
            "embedded_manifest_members": 1, "total_members": len(seen),
            "uncompressed_member_bytes": total, "gzip_eof_crc_verified": True,
            "all_members": member_receipts, "downloaded_code_executed": False,
            "tar_extraction_performed": False, "local_source_payloads_used": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", choices=("attempt-01", "attempt-02"), required=True)
    args = parser.parse_args()
    expected_path = HERE / "expected-publication-v1.json"
    expected = json.loads(expected_path.read_text())
    require(len(expected["assets"]) == 12, "Expected exactly twelve small quality assets")
    directory = HERE / args.attempt
    directory.mkdir(exist_ok=False)
    downloads = directory / "downloads"
    downloads.mkdir()
    save(directory / "expected-publication-v1.json", expected)
    report = {"schema": "muxpd-small-quality-public-readback-v1", "attempt": args.attempt,
              "started_at": now(), "status": "INCOMPLETE", "passed": False,
              "script": identity(Path(__file__)), "trust_anchor": identity(expected_path),
              "public_release": expected["public_release"], "downloads": [], "errors": [],
              "scope": "Only this 636KB 24-case quality Release; no layer/layout bulk readback",
              "github_write_performed": False, "downloaded_code_executed": False}
    save(directory / "started-v1.json", report)
    try:
        for key, value in getproxies().items():
            if key in ("http", "https"):
                url = urlsplit(value)
                require(url.username is None and url.password is None and url.scheme in ("http", "https"),
                        "Routing proxy must have no embedded credentials")
        report["network_routing"] = "Existing credential-free routing proxy; no GitHub authentication"
        for asset in expected["assets"]:
            row = fetch(expected["public_asset_base"], asset, downloads)
            report["downloads"].append(row)
            require(row["passed"], "Public download failed: " + asset["name"])
        sums = {}
        for line in (downloads / "SHA256SUMS").read_text().splitlines():
            sha, name = line.split("  ", 1)
            require(name not in sums, "Duplicate public checksum member")
            sums[name] = sha
        require(sums == {a["name"]: a["sha256"] for a in expected["assets"] if a["name"] != "SHA256SUMS"},
                "Public checksum inventory differs from trust anchor")
        record = json.loads((downloads / "archive-record-v1.json").read_text())
        archive_name, manifest_name = expected["archive_name"], expected["manifest_name"]
        require(record["archive"]["name"] == archive_name and record["manifest"]["name"] == manifest_name,
                "Public archive record identifies different assets")
        require(record["archive"] == {"name": archive_name, **identity(downloads / archive_name)} and
                record["manifest"]["sha256"] == identity(downloads / manifest_name)["sha256"],
                "Public archive/manifest is not bound by the public record")
        report["archive_verification"] = verify_archive(downloads / archive_name, downloads / manifest_name,
                                                       record["archive"]["sha256"], 45)
        report["archive_record_original_creation_status"] = record["status"]
        report["creation_status_note"] = "Original local-staging metadata is retained unchanged; this new receipt establishes public readback separately"
        report["status"], report["passed"] = "PASS", True
    except Exception:
        report["errors"].append(traceback.format_exc())
    report["finished_at"] = now()
    save(directory / "readback-v1.json", report)
    print(json.dumps({"status": report["status"], "receipt": str(directory / "readback-v1.json"),
                      "errors": report["errors"]}, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
