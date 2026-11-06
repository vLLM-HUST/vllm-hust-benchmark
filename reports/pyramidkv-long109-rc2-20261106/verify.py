"""CPU-only offline verifier. Run with Python 3.11+; no third-party packages."""

import argparse
import base64
import gzip
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

SOURCE_COMMIT = "77b5c7fd8817b14c9800f6369bd4db622f53f43e"  # pragma: allowlist secret
BANK_SHA256 = "cff89e9ce8fb72911b4f35ad3ecd3c821e2f5dbf4395dd367bc7af2a5ad38d4e"  # pragma: allowlist secret
WHEEL_SHA256 = "5392b2a8afa971cf0af1b9928d693ab80a71976256de87e7f85b5f9c3a4819be"  # pragma: allowlist secret


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def safe_name(name):
    p = PurePosixPath(name)
    require(
        bool(name)
        and not p.is_absolute()
        and all(x not in ("", ".", "..") for x in name.split("/"))
        and "\\" not in name,
        "Unsafe member path: " + name,
    )
    return p


def sums(path):
    result = {}
    for line in path.read_text().splitlines():
        digest, name = line.split("  ", 1)
        safe_name(name)
        require(
            re.fullmatch("[a-f0-9]{64}", digest) and name not in result,
            "Invalid checksum manifest",
        )
        result[name] = digest
    return result


def extract(report, output):
    public = sums(report / "SHA256SUMS")
    actual = {
        p.relative_to(report).as_posix()
        for p in report.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.relative_to(report).as_posix() != "SHA256SUMS"
    }
    require(set(public) == actual, "Public file inventory mismatch")
    for name, digest in public.items():
        require(
            sha((report / name).read_bytes()) == digest,
            "Public checksum mismatch: " + name,
        )
    members = json.loads(gzip.decompress((report / "members.json.gz").read_bytes()))
    archives = {row["archive"] for row in members.values()}
    require(
        archives == {"raw-common.tar.gz", "raw-qwen35-on-558.tar.gz"},
        "Archive inventory mismatch",
    )
    seen = set()
    for archive in sorted(archives):
        require((report / archive).stat().st_size < 50 * 2**20, "Oversized archive")
        with tarfile.open(report / archive, "r:gz") as tar:
            for member in tar:
                safe_name(member.name)
                require(
                    member.isfile() and member.name not in seen,
                    "Nonregular or duplicate tar member",
                )
                require(member.name in members, "Undeclared tar member")
                spec = members[member.name]
                require(
                    spec["archive"] == archive and spec["bytes"] == member.size,
                    "Member location/size mismatch",
                )
                data = tar.extractfile(member).read()
                require(
                    len(data) == spec["bytes"] and sha(data) == spec["sha256"],
                    "Member checksum mismatch: " + member.name,
                )
                destination = output / member.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                seen.add(member.name)
    require(seen == set(members), "Missing archived member")
    originals = sums(output / "SHA256SUMS")
    require(len(originals) == 617, "Original file count changed")
    require(
        set(originals)
        == {
            p
            for p in members
            if not p.startswith("supplemental/") and p != "SHA256SUMS"
        },
        "Original inventory mismatch",
    )
    for name, digest in originals.items():
        require(
            sha((output / name).read_bytes()) == digest,
            "Original checksum mismatch: " + name,
        )
    return len(members)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def source_identity(root):
    identity = read(root / "supplemental/source-identity.json")
    require(identity["commit"] == SOURCE_COMMIT, "Source commit mismatch")
    objects = {}
    for oid, record in identity["objects"].items():
        data = base64.b64decode(record["base64"], validate=True)
        kind = record["type"]
        require(kind in ("commit", "tree"), "Unexpected Git object kind")
        require(
            hashlib.sha1(
                (kind + " " + str(len(data)) + "\0").encode() + data
            ).hexdigest()
            == oid,
            "Git object hash mismatch",
        )
        objects[oid] = data
    commit = objects[SOURCE_COMMIT]
    tree = commit.splitlines()[0].decode().removeprefix("tree ")

    def entry(oid, name):
        remaining = objects[oid]
        while remaining:
            header, remaining = remaining.split(b"\0", 1)
            child, remaining = remaining[:20].hex(), remaining[20:]
            _, filename = header.decode().split(" ", 1)
            if filename == name:
                return child
        raise ValueError("Source path absent from committed tree: " + name)

    for name in identity["selected_paths"]:
        oid = tree
        for part in PurePosixPath(name).parts:
            oid = entry(oid, part)
        data = (root / "supplemental/source" / name).read_bytes()
        require(
            hashlib.sha1(("blob " + str(len(data)) + "\0").encode() + data).hexdigest()
            == oid,
            "Committed source blob mismatch: " + name,
        )
    wheel = (
        root / "dist-final/vllm_ascend_pyramidkv_hust-0.4.0rc2.dev0-py3-none-any.whl"
    )
    require(sha(wheel.read_bytes()) == WHEEL_SHA256, "Wheel identity mismatch")
    with zipfile.ZipFile(wheel) as package:
        names = [
            n for n in package.namelist() if n.startswith("vllm_ascend_pyramidkv/")
        ]
        require(
            set(names)
            == {
                n.removeprefix("src/")
                for n in identity["selected_paths"]
                if n.startswith("src/")
            },
            "Wheel source inventory mismatch",
        )
        for name in names:
            require(
                package.read(name)
                == (root / "supplemental/source/src" / name).read_bytes(),
                "Wheel/source mismatch: " + name,
            )
            require(
                package.read(name) == (root / "candidate-overlay" / name).read_bytes(),
                "Executed overlay/wheel mismatch: " + name,
            )
    return tree


def validate(root):
    contract = read(root / "contract.json")
    for name, digest in contract["files_sha256"].items():
        require(
            sha((root / name).read_bytes()) == digest,
            "Frozen contract file mismatch: " + name,
        )
    require(
        contract["scope"].startswith("Fresh functional release acceptance"),
        "Contract scope mismatch",
    )
    require(contract["physical_devices"] == [4, 6], "Device identity mismatch")
    require(
        contract["lifecycles"]
        == [
            {
                "name": "qwen35-on-558",
                "model": "qwen35",
                "enabled": True,
                "concurrency": 128,
            }
        ],
        "Lifecycle contract mismatch",
    )
    require(
        contract["shared_settings"]["enforce_eager"] is True
        and contract["shared_settings"]["tp"] == 2
        and contract["shared_settings"]["speculative_config"] is None,
        "TP2 Eager/no-MTP mismatch",
    )
    require(
        contract["client_connector_limit"] == 0
        and contract["concurrency_by_model"] == {"qwen35": 128},
        "Concurrency contract mismatch",
    )
    require(
        sha((root / "qwen35-requests.json").read_bytes()) == BANK_SHA256,
        "Request bank changed",
    )
    historical_contract = root / "supplemental/historical/pressure-contract-v2.json"
    require(
        read(root / "supplemental/historical/contract.json")["prior_contract_sha256"]
        == contract["prior_contract_sha256"],
        "Inherited lineage pointer changed",
    )
    require(
        sha(historical_contract.read_bytes()) == contract["prior_contract_sha256"],
        "Historical contract lineage mismatch",
    )
    tree = source_identity(root)
    run = root / "runs/qwen35-on-558"
    audit_path = root / "supplemental/source/scripts/model-comparison/audit.py"
    audited = subprocess.check_output(
        [
            sys.executable,
            str(audit_path),
            "--requests",
            str(root / "qwen35-requests.json"),
            "--run-dir",
            str(run),
        ],
        text=True,
    )
    require(
        json.loads(audited) == read(root / "release-audit.json"),
        "Strict transaction audit differs",
    )
    raw_before = (root / "raw-verification.json").read_bytes()
    subprocess.run(
        [sys.executable, "-I", str(root / "verify_streams.py")],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    require(
        (root / "raw-verification.json").read_bytes() == raw_before,
        "Raw stream audit differs",
    )
    audit = module("strict_audit", audit_path)
    measure = module("frozen_measure", root / "measure.py")
    bank = read(root / "qwen35-requests.json")
    identity = read(run / "evaluation/identity.json")
    require(
        identity
        == {
            "requests_sha256": BANK_SHA256,
            "client_sha256": sha((root / "measure.py").read_bytes()),
        },
        "Client identity mismatch",
    )
    expected_pool = {
        "physical_blocks": 558,
        "logical_block_size": 2048,
        "effective_capacity_tokens": 457113,
        "max_concurrency_at_max_model_len": 55.8,
    }
    for name in ("startup-metrics.prom", "final-metrics.prom"):
        require(
            audit.pool((run / name).read_text()) == expected_pool,
            "Native pool mismatch: " + name,
        )
    require(
        read(run / "pool-confirmation.json") == expected_pool,
        "Pool confirmation mismatch",
    )
    summaries = []
    all_ids = set()
    for kind, rows in bank.items():
        folder = run / "evaluation" / kind
        group = read(folder / "group.json")
        results = []
        for row in [dict(rows[0], id=kind + "-readiness"), *rows]:
            result = read(folder / (row["id"] + ".result.json"))
            events = read(folder / (row["id"] + ".events.json"))["events"]
            calculated = measure.request_metrics(
                events,
                result["start"],
                row["input_tokens"],
                row["payload"]["max_tokens"],
            )
            require(
                all(result[k] == v for k, v in calculated.items()),
                "Recomputed stream metrics mismatch: " + row["id"],
            )
            require(
                result["start"] <= events[0]["t"] <= events[-1]["t"] <= result["end"],
                "Request timing boundary mismatch",
            )
            ids = {e["data"]["id"] for e in events if isinstance(e["data"], dict)}
            require(ids == {"cmpl-" + row["id"]}, "Response identity mismatch")
            require(not ids & all_ids, "Repeated response identity")
            all_ids.update(ids)
            if row["id"] != kind + "-readiness":
                results.append(result)
        require(group["results"] == results, "Group result identity mismatch")
        require(
            group["start"] <= min(r["start"] for r in results)
            and group["end"] >= max(r["end"] for r in results),
            "Group timing mismatch",
        )
        summaries.append(
            {
                "kind": kind,
                **measure.group_metrics(results, group["end"] - group["start"]),
            }
        )
    require(
        summaries == read(run / "evaluation/summary.json"),
        "Recomputed aggregate mismatch",
    )
    telemetry = [
        json.loads(line)
        for line in (run / "evaluation/telemetry.jsonl").read_text().splitlines()
    ]
    require(
        all(
            "error" not in row
            and measure.metric_values(row["metrics_text"]) == row["metrics"]
            for row in telemetry
        ),
        "Telemetry/raw metric mismatch",
    )
    require(
        all(b["t"] >= a["t"] for a, b in zip(telemetry, telemetry[1:])),
        "Telemetry timing mismatch",
    )
    log = (run / "server.log").read_text()
    require(not re.search(r"\b(?:ERROR|Traceback)\b", log), "Server error present")
    require(
        re.search(
            r"replay boundary request_id=cmpl-long-109-0-\S+ computed_tokens=6144 scheduled_tokens=1025 capped_tokens=1024 prompt_tokens=7168(?=\s|$)",
            log,
        ),
        "Missing actual long-109 replay boundary",
    )
    for name in ("npu-after.txt",):
        text = (run / name).read_text()
        for device in (4, 6):
            require(
                re.search(r"\|\s*" + str(device) + r"\s+910B2\s+\|\s+OK", text),
                "Hardware identity/health mismatch",
            )
            require(
                "No running processes found in NPU " + str(device) in text,
                "Device not released",
            )
    final_devices = (root / "devices-final.txt").read_text()
    require(
        all(
            "No running processes found in NPU " + str(device) in final_devices
            for device in (4, 6)
        ),
        "Final device receipt mismatch",
    )
    require(read(run / "shutdown.json")["returncode"] == 0, "Unclean shutdown")
    completion = read(run / "completion.json")
    require(
        completion["measured_requests"] == 176
        and completion["readiness_requests"] == 2,
        "Completion count mismatch",
    )
    historical = (
        root / "supplemental/historical/runs/qwen35-on-558/server.log"
    ).read_text()
    expected_ids = ["long-readiness", *[r["id"] for r in bank["long"]]]
    try:
        audit.transactions(historical, "qwen35", expected_ids)
    except ValueError as error:
        require(
            str(error) == "Missing or duplicate request transaction: long-109",
            "Historical failure changed",
        )
    else:
        raise ValueError("Historical incomplete run unexpectedly accepted")
    acceptance = read(root / "acceptance.json")
    require(
        acceptance["source_commit"] == SOURCE_COMMIT
        and acceptance["transaction_gate"] == json.loads(audited),
        "Acceptance identity mismatch",
    )
    for name, digest in acceptance["final_artifacts"].items():
        require(
            sha((root / "dist-final" / name).read_bytes()) == digest,
            "Final artifact mismatch",
        )
    return {
        "status": "verified",
        "original_files": 617,
        "raw_streams": 178,
        "target_commits": 128,
        "scheduler_commits": 129,
        "worker_acknowledgements": 258,
        "native_pool": expected_pool,
        "source_commit": SOURCE_COMMIT,
        "source_tree": tree,
        "historical_long109": "rejected",
        "devices_released": [4, 6],
    }


def delivery(report, root, source_tree):
    value = read(report / "delivery-receipt.json")
    require(value["schema_version"] == 1, "Unknown delivery receipt schema")
    commit = base64.b64decode(value["merged_git_commit_base64"], validate=True)
    merged = value["merged_source_commit"]
    require(
        hashlib.sha1(
            ("commit " + str(len(commit)) + "\0").encode() + commit
        ).hexdigest()
        == merged,
        "Merged commit object mismatch",
    )
    require(
        commit.splitlines()[0].decode()
        == "tree " + source_tree
        == "tree " + value["source_tree"],
        "Merged source tree differs",
    )
    documents = {}
    raw = {}
    for name, item in value["receipts"].items():
        safe_name(name)
        data = item["utf8"].encode()
        require(
            len(data) == item["bytes"] and sha(data) == item["sha256"],
            "Delivery receipt byte mismatch: " + name,
        )
        raw[name] = data
        if name.endswith(".json"):
            documents[name] = json.loads(data)
    require(
        documents["source-merge.json"]["merged"] is True
        and documents["source-merge.json"]["sha"] == merged,
        "Source merge receipt mismatch",
    )
    for name, expected_sha, count in [
        ("source-pr-checks.json", SOURCE_COMMIT, 7),
        ("source-main-checks.json", merged, 4),
    ]:
        records = documents[name]
        rows = records if isinstance(records, list) else records["check_runs"]
        require(
            len(rows) == count
            and all(
                row["head_sha"] == expected_sha
                and row["status"] == "completed"
                and row["conclusion"] == "success"
                for row in rows
            ),
            "Source CI receipt incomplete: " + name,
        )
    release = documents["release-created.json"]
    require(
        release["draft"] is False
        and release["prerelease"] is True
        and release["tag_name"] == "v0.4.0rc2.dev0",
        "Candidate release status mismatch",
    )
    require(release["html_url"] == value["release_url"], "Release URL mismatch")
    provenance = documents["release-assets/release-provenance.json"]
    require(
        provenance["source_commit"] == merged
        and provenance["source_tree"] == source_tree
        and provenance["reviewed_source_commit"] == SOURCE_COMMIT,
        "Public source provenance mismatch",
    )
    readback = documents["public-readback.json"]
    require(
        readback["verified"] is True
        and readback["downloaded_wheel_clean_install"] is True
        and readback["downloaded_sdist_validation_script"] is True,
        "Public candidate acceptance missing",
    )
    require(
        readback["source_commit"] == merged
        and readback["source_tree"] == source_tree
        and readback["release_id"] == release["id"]
        and readback["release_url"] == value["release_url"]
        and readback["tag"] == release["tag_name"],
        "Public readback identity mismatch",
    )
    require(
        len(documents["public-install-log.json"]) == 2
        and all(row["returncode"] == 0 for row in documents["public-install-log.json"]),
        "Public clean install failed",
    )
    for name, expected in provenance["artifacts"].items():
        data = (root / "dist-final" / name).read_bytes()
        require(
            sha(data) == expected["sha256"] == readback["assets"][name]["sha256"]
            and len(data) == expected["bytes"] == readback["assets"][name]["bytes"],
            "Public artifact differs from accepted artifact: " + name,
        )
    for name in ("release-provenance.json", "SHA256SUMS"):
        data = raw["release-assets/" + name]
        require(
            sha(data) == readback["assets"][name]["sha256"]
            and len(data) == readback["assets"][name]["bytes"],
            "Public release metadata differs: " + name,
        )
    assets = {item["name"]: item for item in release["assets"]}
    require(
        set(assets) == set(readback["assets"]), "Public release asset inventory differs"
    )
    for name, item in readback["assets"].items():
        require(
            item["bytes"] == assets[name]["size"]
            and item["url"] == assets[name]["browser_download_url"],
            "Release API asset mismatch: " + name,
        )
    for line in raw["release-assets/SHA256SUMS"].decode().splitlines():
        digest, name = line.split("  ", 1)
        require(
            readback["assets"][name]["sha256"] == digest,
            "Public release checksum mismatch: " + name,
        )
    return {
        "merged_source_commit": merged,
        "release_url": value["release_url"],
        "captured_public_readback": "verified",
        "captured_source_pr_checks": 7,
        "captured_source_main_checks": 4,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="pyramidkv-rc2-verify-") as temp:
        root = Path(temp)
        count = extract(args.report, root)
        result = validate(root)
        for name in ("acceptance.json", "raw-verification.json", "release-audit.json"):
            require(
                (args.report / name).read_bytes() == (root / name).read_bytes(),
                "Public summary differs: " + name,
            )
        result["archived_members"] = count
        result["delivery"] = delivery(args.report, root, result["source_tree"])
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
