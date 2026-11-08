"""Acquire pinned publisher transcript/lexicon assets, not audio or TTS outputs.

Plan-only by default. No conversation flattening, invented prompts, model calls
or claim that metadata acquisition qualifies the original synthesis workflow.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CATALOG = (
    ROOT
    / "workload-preparation/data/catalogs/dailytalk_original_metadata_files_v1.json"
)


def prepare(args):
    pins = json.loads(CATALOG.read_text("utf-8"))
    total = sum(item["bytes"] for item in pins["files"])
    report = {
        "source_id": "DAILYTALK",
        "status": "PLAN_ONLY_NO_DOWNLOAD",
        "original_metadata_files": len(pins["files"]),
        "original_metadata_bytes": total,
        "audio_files_acquired": 0,
        "model_calls": 0,
        "native_synthesis_or_scoring_executed": False,
        "source_checkbox_eligible": False,
    }
    if args.action == "plan":
        return report
    destination = args.data_root.resolve()
    if destination.is_relative_to(ROOT):
        raise ValueError("external private data root required")
    if args.action == "prepare" and (args.max_bytes is None or args.max_bytes < total):
        raise ValueError(f"explicit metadata budget at least {total} bytes required")
    sys.path.insert(0, str(ROOT / "scripts"))
    from prepare_crosscodeeval_original_assets import acquire, verify_git_blob
    from prepare_xsum_original_assets import bound_path

    for original in pins["files"]:
        item = {
            **original,
            "url": f"https://api.github.com/repos/{pins['repository']}/git/blobs/{original['git_blob']}",
        }
        target = bound_path(destination, "", item)
        if args.action == "prepare" and not target.exists():
            acquire(target, item)
        verify_git_blob(target, item)
    report["status"] = "ORIGINAL_METADATA_VERIFIED_AUDIO_AND_NATIVE_TTS_PENDING"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--action", choices=["plan", "prepare", "verify"], default="plan"
    )
    parser.add_argument("--max-bytes", type=int)
    print(json.dumps(prepare(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
