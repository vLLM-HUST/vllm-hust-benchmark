"""Build and verify the pinned GenNova MOD evidence import."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/gennova-mod-evidence-20261105"
SNAPSHOTS = REPORT / "snapshots"
SOURCE_REPOSITORY = "GenNova-HUST/inference-engine-project"
SOURCE_COMMIT = "c8416aad2c9e161abe2dde5e01690edc48e756e3"  # pragma: allowlist secret
CAPTURED_DATE = "2026-11-05"

SOURCES = {
    "results-catalog.json": {
        "source_path": "config/results-catalog.json",
        "blob_sha": "c6c7d81ab60d873dc5e0656296d500f40af8b6e1",  # pragma: allowlist secret
        "source_sha256": "082ef2da20212dbdac5881ba587e81ae0c7d7590fa721365a444d95dcba07e13",  # pragma: allowlist secret
        "snapshot_sha256": "082ef2da20212dbdac5881ba587e81ae0c7d7590fa721365a444d95dcba07e13",  # pragma: allowlist secret
    },
    "gennova-positive-integrations.json": {
        "source_path": "config/gennova-positive-integrations.json",
        "blob_sha": "add6afa4fde491aa9daccca791a34b9416a66ee5",  # pragma: allowlist secret
        "source_sha256": "70b6c40ef198f4214d1993a977896c0c7e410c89655446c1e0aeeb3db19de7f3",  # pragma: allowlist secret
        "snapshot_sha256": "11e3cb6f5e30b44ef51aabd5614b129a2f4f319704917729b546abc2892f6d24",  # pragma: allowlist secret
    },
    "combination-plan.json": {
        "source_path": "mods/combination-plan.json",
        "blob_sha": "ec39e1952cfc3c3e932b2b4bc487f29392baad29",  # pragma: allowlist secret
        "source_sha256": "e67d75759f1f3520314a830093c7612a08b18ccf700e0a5d49ea41671c7a7f7a",  # pragma: allowlist secret
        "snapshot_sha256": "e67d75759f1f3520314a830093c7612a08b18ccf700e0a5d49ea41671c7a7f7a",  # pragma: allowlist secret
    },
    "negative-results.json": {
        "source_path": "mods/negative-results.json",
        "blob_sha": "e8f7c6444efde7b49eadf704b07f7c290ea48991",  # pragma: allowlist secret
        "source_sha256": "cd4f05491194efb916613c5c9065cca286d3aa2f459a99d3491b248be3ba4502",  # pragma: allowlist secret
        "snapshot_sha256": "cd4f05491194efb916613c5c9065cca286d3aa2f459a99d3491b248be3ba4502",  # pragma: allowlist secret
    },
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_snapshot(name):
    path = SNAPSHOTS / name
    raw = path.read_bytes()
    actual = digest(raw)
    expected = SOURCES[name]["snapshot_sha256"]
    if actual != expected:
        raise ValueError(f"{path.relative_to(ROOT)}: expected {expected}, got {actual}")
    return raw, json.loads(raw)


def source_url(path, kind="blob"):
    return (
        f"https://github.com/{SOURCE_REPOSITORY}/{kind}/{SOURCE_COMMIT}/"
        f"{quote(path, safe='/')}"
    )


def build_manifest():
    loaded = {name: load_snapshot(name) for name in SOURCES}
    catalog = loaded["results-catalog.json"][1]

    snapshots = []
    for name, metadata in SOURCES.items():
        raw, value = loaded[name]
        stats = {"schema": value.get("schema")}
        if name == "results-catalog.json":
            stats.update(
                entry_count=len(value["entries"]),
                category_counts=dict(
                    sorted(
                        Counter(item["category"] for item in value["entries"]).items()
                    )
                ),
            )
        elif name == "gennova-positive-integrations.json":
            stats.update(
                integration_count=len(value["entries"]),
                combination_effect_count=len(value["combination_effects"]),
                unique_integrated_candidates=value["unique_integrated_candidates"],
            )
        elif name == "combination-plan.json":
            stats.update(
                measured_combination_count=len(value["measured_combinations"]),
                profile_count=len(value["profiles"]),
                integration_queue_count=len(value["integration_queue"]),
                pending_register_count=len(value["pending_register"]),
            )
        elif name == "negative-results.json":
            stats["negative_result_count"] = len(value["entries"])
        snapshots.append(
            {
                "snapshot_path": f"snapshots/{name}",
                "source_path": metadata["source_path"],
                "source_url": source_url(metadata["source_path"]),
                "source_blob_sha": metadata["blob_sha"],
                "source_sha256": metadata["source_sha256"],
                "sha256": metadata["snapshot_sha256"],
                "size_bytes": len(raw),
                "statistics": stats,
            }
        )

    catalog_entries = []
    for entry in catalog["entries"]:
        item = {
            "id": entry["id"],
            "category": entry["category"],
            "tasks": entry["tasks"],
            "source_path": entry["path"],
            "source_url": source_url(entry["path"], kind="tree"),
        }
        if "title" in entry:
            item["title"] = entry["title"]
        catalog_entries.append(item)

    return {
        "schema": "gennova-mod-evidence-import/v1",
        "source": {
            "repository": SOURCE_REPOSITORY,
            "commit": SOURCE_COMMIT,
            "commit_url": f"https://github.com/{SOURCE_REPOSITORY}/commit/{SOURCE_COMMIT}",
            "captured_date": CAPTURED_DATE,
        },
        "scope": {
            "description": "Pinned catalog and bounded MOD result summaries imported for audit and discovery.",
            "is_official_leaderboard_submission": False,
            "contains_new_measurements": False,
            "raw_result_files_duplicated": False,
            "website_publication_enabled": False,
            "exclusions": [
                "raw result payloads and large logs",
                "private endpoints, credentials, host addresses and temporary paths",
                "coordination notes and unverified conversational claims",
            ],
        },
        "snapshots": snapshots,
        "catalog_entries": catalog_entries,
    }


def render_json(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def generated_files():
    manifest = render_json(build_manifest()).encode()
    checksummed = [("manifest.json", manifest)]
    checksummed.extend(
        (f"snapshots/{name}", (SNAPSHOTS / name).read_bytes()) for name in SOURCES
    )
    sums = "".join(f"{digest(raw)}  {path}\n" for path, raw in checksummed).encode()
    return {REPORT / "manifest.json": manifest, REPORT / "SHA256SUMS": sums}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    expected = generated_files()
    if args.write:
        for path, content in expected.items():
            path.write_bytes(content)
        print(f"Updated {len(expected)} generated files in {REPORT.relative_to(ROOT)}")
        return 0

    stale = [
        str(path.relative_to(ROOT))
        for path, data in expected.items()
        if not path.is_file() or path.read_bytes() != data
    ]
    if stale:
        print("Stale GenNova MOD evidence import: " + ", ".join(stale))
        return 1
    print("GenNova MOD evidence import is current and source snapshot hashes match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
