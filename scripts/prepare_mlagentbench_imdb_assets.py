"""Acquire or verify pinned original IMDb/DistilBERT files outside the repo."""

import argparse
import json
from pathlib import Path

from prepare_mlagentbench_cifar_assets import acquire, verify
from prepare_mlagentbench_imdb_offline import PROFILE

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    files = json.loads(PROFILE.read_text("utf-8"))["files"]
    total = sum(spec["bytes"] for spec in files.values())
    report = {
        "source_id": "MLAGENTBENCH",
        "task_id": "imdb",
        "status": "PLAN_ONLY_NO_DOWNLOAD",
        "files": len(files),
        "selected_bytes": total,
        "source_checkbox_eligible": False,
        "model_calls": 0,
    }
    if args.action == "plan":
        return report
    root = args.output.resolve()
    if root.is_relative_to(ROOT):
        raise ValueError("original assets must remain outside the repository")
    if args.action == "acquire" and (args.max_bytes is None or args.max_bytes < total):
        raise ValueError("explicit budget must cover all selected original bytes")
    verified = []
    for name, identity in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("contained relative original path required")
        parent = root / relative.parent
        if not parent.resolve().is_relative_to(root):
            raise ValueError("asset parent must remain inside the selected root")
        spec = dict(identity, path=relative.name)
        if args.action == "acquire":
            # Reuse exact-length/hash/resume/disk-budget protection, no new client.
            acquire(parent, spec, max_bytes=args.max_bytes, resume=args.resume)
        verified.append({"asset": name, **verify(parent / relative.name, spec)})
    return dict(
        report,
        status="ORIGINAL_ASSETS_VERIFIED_NOT_AGENT_EXECUTION",
        verified=verified,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--action", choices=["plan", "acquire", "verify"], default="plan"
    )
    parser.add_argument("--max-bytes", type=int)
    parser.add_argument("--resume", action="store_true")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
