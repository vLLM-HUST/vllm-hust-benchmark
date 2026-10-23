"""Inspect a publisher's data files first; fetch only with --download, without model calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from urllib.parse import quote

from materialize_public_datasets import acquire
from verify_public_sources import fetch_json
from vllm_hust_workload_data.registry import source_spec

ROOT = Path(__file__).resolve().parents[1]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_id")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--files", nargs="+")
    parser.add_argument("--revision")
    parser.add_argument("--max-bytes", type=int, default=550_000_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.download and args.output is None:
        parser.error("--download requires an explicit --output receipt path")
    catalog = json.loads(
        (ROOT / "data/catalogs/authoritative_datasets_v1.json").read_text(
            encoding="utf-8"
        )
    )
    source = next(
        (s for s in catalog["sources"] if s["source_id"] == args.source_id), None
    )
    if not source or source["platform"] != "hf":
        parser.error(
            "not an HF source; use catalog download-plan for the publisher-specific route"
        )
    repo = source["repo_id"]
    data = fetch_json(
        "https://huggingface.co/api/datasets/" + quote(repo, safe="/") + "?blobs=true"
    )
    files = {f["rfilename"]: f for f in data["siblings"]}
    selected = args.files
    # Reuse the exact previously acquired selection by default, not all duplicate variants.
    if selected is None:
        old = source_spec(args.source_id, ROOT)
        if old["files"]:
            selected = [f["path"] for f in old["files"] if f["role"] == "data"]
            if data["sha"] != old["revision"] and args.revision is None:
                parser.error(
                    "publisher HEAD changed; explicit revision/selection review required; old files remain unchanged"
                )
    if args.revision and args.revision != data["sha"]:
        parser.error(
            "this downloader requires the reviewed current publisher revision; use existing pinned URLs for older assets"
        )
    if not args.download:
        print(
            json.dumps(
                {
                    "source_id": args.source_id,
                    "revision": data["sha"],
                    "gated": data.get("gated"),
                    "selected_files": selected,
                    "files": [
                        {
                            "path": n,
                            "bytes": f.get("size"),
                            "sha256": (f.get("lfs") or {}).get("sha256"),
                        }
                        for n, f in files.items()
                        if n.endswith(
                            (".jsonl", ".json", ".parquet", ".zip", ".tar.gz")
                        )
                    ],
                    "next": "Use --download --files exact/path with a reviewed byte limit; archives/media are not parsed by this command.",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if not selected or any(n not in files for n in selected):
        parser.error("explicit existing publisher data files required")
    if data.get("gated") is not False:
        parser.error(
            "access gate requires user action at publisher; no automatic acceptance"
        )
    metadata = {
        "revision": data["sha"],
        "gated": False,
        "reported_data_license": (data.get("cardData") or {}).get("license"),
    }
    output = (
        args.output
        or ROOT / f"results/acquisition-{args.source_id}-{data['sha'][:12]}.json"
    )
    prior = None
    if output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        prior = next(
            (
                s
                for s in previous.get("sources", [])
                if s.get("source_id") == args.source_id
            ),
            None,
        )
        if (
            not prior
            or prior.get("revision") != data["sha"]
            or set(selected)
            != {f["file"] for f in prior.get("files", []) if "row_count" in f}
        ):
            parser.error(
                "receipt exists for a different selection; use a fresh --output"
            )
    result = acquire(source, metadata, {args.source_id: selected}, args.max_bytes)
    output.parent.mkdir(parents=True, exist_ok=True)
    if prior:
        expected = {f["file"]: f["sha256"] for f in prior["files"]}
        observed = {f["file"]: f["sha256"] for f in result.get("files", [])}
        if expected != observed or prior["row_count"] != result.get("row_count"):
            parser.error(
                "revalidation differs from frozen receipt; original receipt preserved"
            )
    else:
        output.write_text(
            json.dumps(
                {"version": "selected-data-acquisition/v1", "sources": [result]},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    # Receipt ownership is explicit; never mutate a research receipt registry.
    print(
        json.dumps(
            {
                "source_id": args.source_id,
                "status": result["status"],
                "rows": result.get("row_count"),
                "receipt": str(output),
            }
        )
    )
    if result["status"] != "SELECTED_ASSETS_DOWNLOADED_AND_PARSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
