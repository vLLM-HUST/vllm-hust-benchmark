"""Prepare five fixed original MARBLE scenario files; no Agent or scoring."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_crosscodeeval_original_assets import acquire, verify_git_blob
from prepare_xsum_original_assets import bound_path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs/workload-data/dataset-validation-status.json"


def prepare(data_root, *, action="plan", max_bytes=None):
    row = json.loads(LEDGER.read_text("utf-8"))["delivery_inventory"]["MULTIAGENTBENCH"]
    pins = row["protocol_review"]
    digests = row["original_asset_acquisition"]["sha256_by_scenario"]
    data_root = Path(data_root).resolve()
    if data_root.is_relative_to(ROOT):
        raise ValueError("original data must remain outside the repository")
    if action not in {"plan", "prepare", "verify"}:
        raise ValueError("unknown preparation action")
    total = sum(item["bytes"] for item in pins["original_assets"])
    if action == "prepare" and (max_bytes is None or max_bytes < total):
        raise ValueError("explicit max-bytes must cover all five original files")
    report = {
        "source_id": "MULTIAGENTBENCH",
        "revision": pins["revision"],
        "status": "PLAN_ONLY_NO_DOWNLOAD"
        if action == "plan"
        else "ORIGINAL_FILES_VERIFIED_NOT_NATIVE_EXECUTION",
        "files": [],
        "total_bytes": total,
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }
    for original in pins["original_assets"]:
        scenario = Path(original["path"]).parent.name
        item = {
            **original,
            "sha256": digests[scenario],
            "url": "https://api.github.com/repos/ulab-uiuc/MARBLE/git/blobs/"
            + original["git_blob"],
        }
        target = bound_path(data_root, "", item)
        result = {
            "path": item["path"],
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        }
        if action != "plan":
            if action == "prepare" and not target.exists():
                acquire(target, item)
            verify_git_blob(target, item)
            records = [
                json.loads(line)
                for line in target.read_text("utf-8").splitlines()
                if line.strip()
            ]
            if len(records) != 100 or not all(
                isinstance(record, dict) for record in records
            ):
                raise ValueError("100 original scenario objects required")
            result["rows"] = len(records)
        report["files"].append(result)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--action", choices=["plan", "prepare", "verify"], default="plan"
    )
    parser.add_argument("--max-bytes", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.data_root, action=args.action, max_bytes=args.max_bytes),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
