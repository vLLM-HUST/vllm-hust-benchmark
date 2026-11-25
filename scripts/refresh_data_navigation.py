"""Regenerate or check the existing data-navigation projections in dependency order."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    flag = "--write" if args.write else "--check"
    for name in (
        "data_index.py",
        "build_website_evidence_links.py",
        "data_entrypoints.py",
    ):
        command = [sys.executable, "-X", "utf8", str(ROOT / "scripts" / name)]
        if args.check or name != "build_website_evidence_links.py":
            command.append(flag)
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
