"""Reuse pinned original research acquisition; never observation or native scoring.

Default is an offline plan. Use external server storage. Gated sources must be
obtained with the publisher's authorized client, then verified here; no auto-login.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

CATALOG = (
    Path(__file__).resolve().parents[1]
    / "workload-preparation/data/catalogs/source_manifests_v1.json"
)
spec = importlib.util.spec_from_file_location(
    "pinned_asset_io", Path(__file__).with_name("prepare_xsum_original_assets.py")
)
IO = importlib.util.module_from_spec(spec)
spec.loader.exec_module(IO)


def selection(source_id, files=None):
    source = json.loads(CATALOG.read_text("utf-8"))["sources"].get(source_id)
    if not source or not source.get("historical_acquisition_reuse"):
        raise ValueError("source requires an audited historical acquisition profile")
    registered = {
        item["path"]: item for item in source["files"] if item["role"] == "raw_source"
    }
    names = list(registered) if files is None else files
    if (
        not names
        or len(set(names)) != len(names)
        or any(name not in registered for name in names)
    ):
        raise ValueError("select distinct original registered files")
    return source, [registered[name] for name in names]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_id")
    parser.add_argument(
        "--mode", choices=["plan", "verify", "download"], default="plan"
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--files", nargs="+")
    parser.add_argument("--max-bytes", type=int, default=550_000_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--hf-endpoint",
        choices=["https://huggingface.co", "https://hf-mirror.com"],
        default="https://huggingface.co",
    )
    args = parser.parse_args()
    source, selected = selection(args.source_id, args.files)
    receipt = {
        "source_id": args.source_id,
        "revision": source["revision"],
        "status": "OFFLINE_ORIGINAL_ASSET_PLAN",
        "files": selected,
        "bytes": sum(item["bytes"] for item in selected),
        "historical_acquisition_reuse": source["historical_acquisition_reuse"],
        "full_registered_selection": len(selected) == len(source["files"]),
        "model_calls": 0,
        "source_checkbox_eligible": False,
        "native_execution_complete": False,
        "research_analysis_migrated": False,
        "raw_payloads_published": False,
    }
    if args.mode == "plan":
        print(json.dumps(receipt, indent=2))
        return
    if args.data_root is None:
        parser.error("verify/download require an explicit external --data-root")
    if args.mode == "verify":
        for item in selected:
            IO.verify(IO.bound_path(args.data_root, "", item), item)
        receipt["status"] = "ORIGINAL_RAW_ASSETS_VERIFIED_READ_ONLY_NOT_NATIVE_DELIVERY"
        print(json.dumps(receipt, indent=2))
        return
    if source.get("access", {}).get("gated") not in {None, False}:
        raise ValueError(
            "publisher access gate requires user authorization/client; obtain originals then use --mode verify"
        )
    if args.max_bytes <= 0 or receipt["bytes"] > args.max_bytes:
        raise ValueError(
            "selection exceeds explicit byte budget; no download or directory created"
        )
    profile_sha = hashlib.sha256(
        json.dumps(source, sort_keys=True).encode()
    ).hexdigest()
    target = args.data_root
    progress = target / "source-preparation.private.json"
    if target.exists():
        if not args.resume or not progress.is_file():
            raise ValueError(
                "fresh root or explicit matching resume required; use verify for existing research assets"
            )
        old = json.loads(progress.read_text("utf-8"))
        if (
            old.get("source_id") != args.source_id
            or old.get("profile_sha256") != profile_sha
        ):
            raise ValueError("resume original source/profile differs")
    else:
        target.mkdir(mode=0o700, parents=True)
    receipt.update(
        status="ACQUIRING_PINNED_ORIGINAL_RAW_ASSETS",
        profile_sha256=profile_sha,
        verified_files=[],
    )
    try:
        progress.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        for item in selected:
            path = IO.bound_path(target, "", item)
            reused = path.exists()
            if reused:
                IO.verify(path, item)
            else:
                url = item["url"].replace(
                    "https://huggingface.co/", args.hf_endpoint + "/"
                )
                IO.fetch(path, item, url)
            receipt["verified_files"].append({"path": item["path"], "reused": reused})
            progress.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        receipt["status"] = "PINNED_ORIGINAL_RAW_ASSETS_ACQUIRED_NOT_NATIVE_DELIVERY"
    except Exception as exc:
        receipt.update(
            status="FAILED_NOT_COMPLETE", error_type=type(exc).__name__, error=str(exc)
        )
        raise
    finally:
        progress.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
