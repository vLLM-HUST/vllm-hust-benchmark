"""Verify source/target release visibility and every asset digest without mutation."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from migrate_devhub_releases import SOURCE, TARGET, check_visibility, releases


def verify(output):
    if output.exists():
        raise ValueError("Refusing to overwrite a verification receipt")
    root = Path(__file__).resolve().parents[1]
    baseline = json.loads(
        (
            root / "archive/dev-hub-releases-20261020/visibility-verification.json"
        ).read_text()
    )
    sources = {r["id"]: r for r in releases(SOURCE)}
    targets = {r["id"]: r for r in releases(TARGET)}
    rows = []
    for expected in baseline["releases"]:
        source = sources[expected["source_release_id"]]
        target = targets[expected["destination_release_id"]]
        if (
            source["tag_name"] != expected["tag"]
            or target["tag_name"] != expected["tag"]
        ):
            raise ValueError("Release tag identity drift")
        check_visibility(source, target)
        assets = {a["name"]: a for a in target["assets"]}
        if len(source["assets"]) != expected["assets_verified"] or set(assets) != {
            a["name"] for a in source["assets"]
        }:
            raise ValueError("Release asset membership drift")
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
                "destination_api_url": target["url"],
                "destination_html_url_snapshot": target["html_url"],
                "source_draft": source["draft"],
                "destination_draft": target["draft"],
                "source_prerelease": source["prerelease"],
                "destination_prerelease": target["prerelease"],
                "assets_verified": len(source["assets"]),
                "assets": [
                    {
                        "name": a["name"],
                        "api_url": a["url"],
                        "browser_download_url_snapshot": a["browser_download_url"],
                        "digest": a["digest"],
                        "bytes": a["size"],
                    }
                    for a in target["assets"]
                ],
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
