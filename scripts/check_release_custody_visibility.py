"""Verify source/target release visibility and every asset digest without mutation."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from migrate_devhub_releases import SOURCE, TARGET, check_visibility, releases


def verify(output):
    if output.exists():
        raise ValueError("Refusing to overwrite a verification receipt")
    sources = releases(SOURCE)
    targets = {r["tag_name"]: r for r in releases(TARGET)}
    rows = []
    for source in sources:
        if not source["tag_name"].startswith("szyn-swebench-qwen35-"):
            continue
        target = targets[source["tag_name"]]
        check_visibility(source, target)
        assets = {a["name"]: a for a in target["assets"]}
        for original in source["assets"]:
            copied = assets[original["name"]]
            if (copied["digest"], copied["size"]) != (
                original["digest"],
                original["size"],
            ):
                raise ValueError("Asset mismatch")
        rows.append(
            {
                "tag": source["tag_name"],
                "source_release_id": source["id"],
                "destination_release_id": target["id"],
                "source_draft": source["draft"],
                "destination_draft": target["draft"],
                "source_prerelease": source["prerelease"],
                "destination_prerelease": target["prerelease"],
                "assets_verified": len(source["assets"]),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "verified_at": datetime.now(timezone.utc).isoformat(),
                "releases": rows,
                "asset_bytes_modified": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Verified visibility and asset digests for {len(rows)} releases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    verify(parser.parse_args().output)
