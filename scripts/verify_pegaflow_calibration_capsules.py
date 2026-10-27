"""Verify byte-preserved PegaFlow calibration capsules against their manifests."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPSULES = (
    "pegaflow-multinode-calibration-20261027",
    "pegaflow-pipeline-generalization-20261027",
    "pegaflow-evening-checkpoint-20261027",
)


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for member in manifest["members"]:
        path = (root / member["path"]).resolve()
        path.relative_to(root.resolve())
        if path.stat().st_size != member["bytes"]:
            raise ValueError(f"Member size differs: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != member["sha256"]:
            raise ValueError(f"Member identity differs: {path}")
    print(f"Verified {len(manifest['members'])} members: {root.name}")


if __name__ == "__main__":
    for name in CAPSULES:
        verify(ROOT / "reports" / name)
