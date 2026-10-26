"""Offline source catalog and explicit read-only validation CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .assets import verify_file
from .registry import adapter_spec, all_adapters, read_catalog, source_spec


def list_workloads(root=None):
    definitions = {
        f["family_id"]: f
        for f in read_catalog("workload_taxonomy_v1.json", root)["canonical_families"]
    }
    adapters = all_adapters(root)
    alias = {
        a["id"]: a["source_id"]
        for a in read_catalog("testing_outline_index_v1.json", root)["qbi_reuse"]
    }
    result = []
    for binding in read_catalog("workload_data_bindings_v1.json", root)["families"]:
        ids = list(
            dict.fromkeys(
                binding["primary_sources"]
                + binding["auxiliary_sources"]
                + binding["local_asset_ids"]
            )
        )
        result.append(
            {
                **binding,
                "name": definitions[binding["family_id"]]["canonical_name"],
                "definition": definitions[binding["family_id"]]["strict_definition"],
                "source_adapters": {
                    sid: adapters.get(alias.get(sid, sid), {}).get(
                        "kind", "NOT_IMPLEMENTED"
                    )
                    for sid in ids
                },
                "asset_availability": "NOT_CHECKED_ON_THIS_MACHINE",
                "runtime_reproduction": "NOT_EXECUTED_BY_THIS_PACKAGE",
                "synthetic_fallback": False,
            }
        )
    return result


def get_workload(family_id, root=None):
    for family in list_workloads(root):
        if family["family_id"] == family_id:
            return family
    raise KeyError(
        f"unknown research workload: {family_id}; legacy generator IDs are retired"
    )


def download_plan(source_id, root=None):
    source = source_spec(source_id, root)
    files = [
        {
            k: f[k]
            for k in (
                "path",
                "url",
                "sha256",
                "bytes",
                "rows",
                "role",
                "locale",
                "split",
                "archive_member",
                "generation",
            )
            if k in f
        }
        for f in source["files"]
    ]
    return {
        "source_id": source_id,
        "publisher": source.get("source_url"),
        "revision": source.get("revision"),
        "access": source["access"],
        "files": files,
        **(
            {"media_archives": source["media_archives"]}
            if source.get("media_archives")
            else {}
        ),
        "preparation_command": source.get("preparation_command"),
        "method": "EXTERNAL_AUTHORIZED_ASSET"
        if source_id.startswith("PAIO-")
        else (
            "ORIGINAL_PUBLISHER_GENERATOR"
            if source_id == "RULER"
            else (
                "PINNED_PUBLISHER_ARCHIVE"
                if source_id == "GPQA"
                else (
                    "PINNED_FILE_URLS" if files else "PUBLISHER_DATA_DOWNLOAD_SECTION"
                )
            )
        ),
        "instruction": "Obtain selected files under publisher/owner terms and supply their root explicitly; this command does not download.",
        "asset_availability": "NOT_CHECKED_ON_THIS_MACHINE",
    }


def verify_source(source_id, data_root, root=None):
    source = source_spec(source_id, root)
    data_files = [f for f in source["files"] if f["role"] == "data"]
    if not data_files:
        raise ValueError(
            "No pinned data files; source listing is not an implemented adapter"
        )
    return {
        "source_id": source_id,
        "revision": source["revision"],
        "files": [verify_file(Path(data_root), f) for f in data_files],
        "status": "FILE_IDENTITIES_VERIFIED",
        "adapter_semantics": "NOT_CHECKED_BY_HASH_VERIFICATION",
        "runtime": "NOT_EXECUTED",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        help="optional checkout catalog root; installed catalogs work without it",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    sub.add_parser(
        "research-list",
        help="active v2 research classes; list retains v1 compatibility groups",
    )
    classify = sub.add_parser("classify")
    classify.add_argument("id")
    classify.add_argument("--native-subtype", required=True)
    classify.add_argument("--evidence-kind", required=True)
    for name in ("show", "source", "download-plan", "verify"):
        command = sub.add_parser(name)
        command.add_argument("id")
        if name == "verify":
            command.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "research-list":
            from .taxonomy import research_classes

            value = research_classes(args.root)
        elif args.command == "classify":
            from .taxonomy import classify_record

            value = classify_record(
                args.id,
                args.native_subtype,
                evidence_kind=args.evidence_kind,
                root=args.root,
            )
        elif args.command == "list":
            value = [
                {
                    k: f[k]
                    for k in (
                        "family_id",
                        "name",
                        "source_adapters",
                        "asset_availability",
                    )
                }
                for f in list_workloads(args.root)
            ]
        elif args.command == "show":
            value = get_workload(args.id, args.root)
        elif args.command == "source":
            value = source_spec(args.id, args.root)
            try:
                value["adapter"] = adapter_spec(args.id, args.root)
            except NotImplementedError:
                value["adapter"] = {"status": "NOT_IMPLEMENTED"}
        elif args.command == "verify":
            value = verify_source(args.id, args.data_root, args.root)
        else:
            value = download_plan(args.id, args.root)
    except (KeyError, ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(value, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
