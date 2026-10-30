"""One anonymous HTTPS attempt per immutable release asset; no retries."""

import concurrent.futures
import datetime
import gzip
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import tarfile
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parent
BASE = "https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/"
ARCHIVE = "muxpd-first-mixed-evidence-v1.tar.gz"


def now():
    return datetime.datetime.now(datetime.UTC).isoformat()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def without_query(value):
    url = urlsplit(value)
    return urlunsplit((url.scheme, url.netloc, url.path, "", ""))


def get_asset(item, tag):
    name, expected = item
    target = ROOT / name
    url = BASE + quote(tag, safe="") + "/" + quote(name, safe="")
    row = {
        "name": name,
        "started_at": now(),
        "requested_public_url": url,
        "expected": expected,
        "attempts": 1,
        "passed": False,
        "errors": [],
    }
    try:
        if target.exists():
            raise FileExistsError(target)
        command = [
            "curl",
            "-q",
            "--silent",
            "--show-error",
            "--location",
            "--http1.1",
            "--tlsv1.2",
            "--proto",
            "=https",
            "--proto-redir",
            "=https",
            "--connect-timeout",
            "20",
            "--max-time",
            "900",
            "--output",
            str(target),
            "--write-out",
            "%{json}",
            url,
        ]
        environment = {
            key: value
            for key, value in os.environ.items()
            if key not in {"GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN"}
        }
        for key, value in environment.items():
            if key.lower() in {"http_proxy", "https_proxy", "all_proxy"}:
                proxy = urlsplit(value)
                if proxy.username is not None or proxy.password is not None:
                    raise ValueError("Credential-bearing proxy is not anonymous")
        process = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=930,
            env=environment,
        )
        row["curl_exit_code"] = process.returncode
        # Do not persist signed redirect URLs, credentials, or complete curl JSON.
        row["stderr"] = re.sub(
            r"https?://[^\s]+", lambda match: without_query(match[0]), process.stderr
        )
        metadata = json.loads(process.stdout)
        row["transport"] = {
            key: metadata.get(key)
            for key in (
                "http_code",
                "ssl_verify_result",
                "http_version",
                "num_redirects",
                "size_download",
                "time_total",
            )
        }
        row["response_url_without_query"] = without_query(metadata["url_effective"])
        row["https_response"] = urlsplit(metadata["url_effective"]).scheme == "https"
    except Exception as error:  # noqa: BLE001 -- retain every failed GET attempt
        row["errors"].append(f"{type(error).__name__}: {error}")
    row["bytes"] = target.stat().st_size if target.exists() else 0
    row["sha256"] = digest(target) if target.exists() else None
    transport = row.get("transport", {})
    row["passed"] = (
        not row["errors"]
        and row.get("curl_exit_code") == 0
        and transport.get("http_code") == 200
        and transport.get("ssl_verify_result") == 0
        and row.get("https_response") is True
        and row["bytes"] == expected["size"]
        and row["sha256"] == expected["sha256"]
    )
    row["finished_at"] = now()
    save(ROOT / (name + ".readback.json"), row)
    print(
        json.dumps({key: row[key] for key in ("name", "bytes", "passed")}), flush=True
    )
    return row


def main():
    expected_path = ROOT / "expected-assets-v1.json"
    expected = json.loads(expected_path.read_text())
    if len(expected["assets"]) != 9:
        raise ValueError("Expected exactly nine declared assets")
    save(
        ROOT / "started-v1.json",
        {
            "started_at": now(),
            "script_sha256": digest(Path(__file__)),
            "expected_assets_sha256": digest(expected_path),
            "anonymous": True,
            "max_parallel_gets": 2,
            "max_seconds_per_get": 900,
            "retries": 0,
            "ranges": False,
            "scope": "Each asset uses one public HTTPS GET, redirect following, verified TLS; curlrc disabled, no netrc, no authentication or cookies supplied. Partial failures are retained. No GitHub writes.",
        },
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(
            pool.map(
                lambda item: get_asset(item, expected["release_tag"]),
                expected["assets"].items(),
            )
        )
    record = {
        "schema": "muxpd-first-mixed-anonymous-readback-v1",
        "assets": rows,
        "archive_verified": False,
        "errors": [],
        "api_inventory_claimed": False,
        "scope": "Downloaded bytes, not API digests or local original archives, establish readback. One retained attempt per asset; no retries or selective replacement.",
    }
    passed = {row["name"] for row in rows if row["passed"]}
    try:
        needed = {ARCHIVE, "source-manifest-v1.json", "build_evidence_v1.py"}
        if not needed <= passed:
            raise ValueError("Archive verification inputs did not fully download")
        archive_path = ROOT / ARCHIVE
        spec = importlib.util.spec_from_file_location(
            "verified_packager", ROOT / "build_evidence_v1.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        verification = module.verify(archive_path)
        if verification["member_count"] != 2172 or not verification["passed"]:
            raise ValueError("Unexpected verified archive inventory")
        with tarfile.open(archive_path, "r|gz") as archive:
            first = next(iter(archive))
            if first.name != "MANIFEST.json":
                raise ValueError("First member is not MANIFEST.json")
            embedded = archive.extractfile(first).read()
        if embedded != (ROOT / "source-manifest-v1.json").read_bytes():
            raise ValueError("Embedded manifest differs from public manifest bytes")
        # Reading through gzip EOF additionally verifies the complete CRC/trailer.
        with gzip.open(archive_path, "rb") as stream:
            while stream.read(1024 * 1024):
                pass
        record.update(
            archive_verified=True,
            archive_verification=verification,
            embedded_manifest_byte_exact=True,
            gzip_eof_crc_verified=True,
        )
    except Exception as error:  # noqa: BLE001 -- archive errors remain INCOMPLETE
        record["errors"].append(f"{type(error).__name__}: {error}")
    record["finished_at"] = now()
    record["passed"] = all(row["passed"] for row in rows) and record["archive_verified"]
    record["status"] = "PASS" if record["passed"] else "INCOMPLETE"
    save(ROOT / "final-status-v1.json", record)
    print(
        json.dumps({"status": record["status"], "errors": record["errors"]}), flush=True
    )
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
