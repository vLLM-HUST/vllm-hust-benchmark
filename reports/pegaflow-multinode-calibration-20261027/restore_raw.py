"""Reassemble frozen captures into a new external directory."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
root = Path(__file__).resolve().parent
for entry in json.loads((root / "raw-archives.json").read_text())["archives"]:
    digest = hashlib.sha256()
    with (a.output / entry["archive"]).open("xb") as stream:
        for name in entry["parts"]:
            payload = (root / name).read_bytes()
            digest.update(payload)
            stream.write(payload)
    if digest.hexdigest() != entry["sha256"]:
        raise ValueError("Archive digest differs: " + entry["archive"])
    print(entry["archive"])
