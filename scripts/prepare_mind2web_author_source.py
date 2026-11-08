"""Prepare pinned original Mind2Web software externally, without executing it."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_swebench_author_source import prepare  # noqa: E402

PINS = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/mind2web-author-source.json"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    pins = json.loads(PINS.read_text("utf-8"))
    report = {
        "revision": pins["revision"],
        "files": len(pins["files"]),
        "native_execution": "NOT_RUN",
        "software_prepared": False,
    }
    if args.execute:
        if args.archive is None:
            parser.error("--execute requires the original publisher --archive")
        report["files"] = prepare(args.archive, args.author_root, pins)
        report["software_prepared"] = True
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
