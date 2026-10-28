"""Generate a public, metadata-only entry directory. Never acquire or execute data."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workload-preparation/src"))
from vllm_hust_workload_data.catalog import download_plan  # noqa: E402
from vllm_hust_workload_data.provider import delivery_assets, delivery_classes  # noqa: E402
from vllm_hust_workload_data.registry import source_spec  # noqa: E402

REVISION = "07628b9b5b32d0de37dd3b6fa3bc48d350d8ed71"  # pragma: allowlist secret - public source pin
BASE = f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{REVISION}/"


def build():
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text()
    )
    overlay = json.loads(
        (ROOT / "docs/dataset-entrypoints/primary-clients.json").read_text()
    )
    rows = []
    for asset in delivery_assets():
        sid = asset["source_id"]
        registered = asset["data_provider_source_registered"]
        source = source_spec(sid) if registered else {}
        plan = download_plan(sid) if registered else {}
        status = ledger["delivery_inventory"].get(sid, {})
        evidence = [
            {"label": key, "url": BASE + value}
            for key, value in status.items()
            if key.endswith("receipt")
            and isinstance(value, str)
            and value.startswith(("docs/", "reports/", "results/"))
        ]
        rows.append(
            {
                "source_id": sid,
                "asset_group": asset["asset_group"],
                "classes": asset["related_workload_ids"],
                "provider_registered": registered,
                "publisher": source.get("source_url"),
                "revision": source.get("revision"),
                "access": source.get("access"),
                "metadata_snapshot": source.get("metadata_snapshot"),
                "files": plan.get("files", []),
                "entrypoints": [
                    {**entry, "url": BASE + entry["path"]}
                    for entry in asset["entrypoints"]
                ],
                "source_plan_command": f"python -m vllm_hust_benchmark.workload_data source-plan {sid}",
                "download_plan_command": f"python -m vllm_hust_workload_data.catalog download-plan {sid}"
                if registered
                else None,
                "ledger_status": status.get("status", "NOT_RECORDED"),
                "ledger_scope": status.get("scope"),
                "scope_limits": status.get("scope_limits"),
                "native_validation": status.get("native_entrypoint_validation"),
                "blocker": status.get("blocker"),
                "evidence": evidence,
                "guide_id": sid
                if sid in {e["source_id"] for e in overlay["entries"]}
                else None,
            }
        )
    assert len(rows) == 118 and len({r["source_id"] for r in rows}) == 118
    assert sum(r["provider_registered"] for r in rows) == 104
    return {
        "schema": "dataset-entrypoints/v1",
        "source_revision": REVISION,
        "scope": "Metadata and existing entrypoints; not all sources are executable or fully evaluated.",
        "counts": {
            "sources": 118,
            "classes": len(delivery_classes()),
            "provider_registered": 104,
            "provider_unregistered": 14,
            "separate_assets": sum(
                r["asset_group"] != "task-class-candidates" for r in rows
            ),
        },
        "platform_api": "BLOCKED_NOT_INTEGRATED",
        "new_measurements": 0,
        "primary_entries": overlay["entries"],
        "sources": rows,
        "provenance": overlay["provenance"],
        "source_sha256": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in (
                "docs/workload-data/dataset-validation-status.json",
                "workload-preparation/data/catalogs/workload_delivery_classification_v1.json",
                "workload-preparation/data/catalogs/source_manifests_v1.json",
                "docs/dataset-entrypoints/primary-clients.json",
            )
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = ROOT / "docs/dataset-entrypoints/dataset-entrypoints-v1.json"
    raw = (json.dumps(build(), ensure_ascii=False, indent=2) + "\n").encode()
    if args.check:
        assert path.read_bytes() == raw, "Directory is stale"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    print(f"118 sources; {len(raw)} bytes; sha256={hashlib.sha256(raw).hexdigest()}")


if __name__ == "__main__":
    main()
