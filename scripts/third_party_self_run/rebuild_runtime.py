"""Recover the matching school028 base from its verified local SDK archive.

This creates a new, explicitly labelled build. It does not claim to recover the
historical full-answer worker or optimization combination.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
SDK = Path("/root/test7-gdn-cache-20261018/main-adapter-cpu-r1/sdk213-r1-direct.tgz")
SDK_SHA = "c03e96929d7b061e5cd66aba22d02ba1cb2df270ddb7654f562f8c6202f2e805"
PREFIX = "venv-school-r1/lib/python3.12/site-packages/"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, required=True, help="New isolated build directory"
    )
    args = parser.parse_args()
    if digest(SDK) != SDK_SHA:
        raise ValueError("SDK archive differs from the frozen school028 donor receipt")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    runtime = out / "venv-school-r1"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(runtime)], check=True
    )
    print("Verified SDK; recovering into", runtime, flush=True)
    protected = {}
    with tarfile.open(SDK, "r:gz") as archive:
        for member in archive:
            if not member.name.startswith(PREFIX):
                continue
            # Only the verified package subtree; retain safe internal library links.
            archive.extract(member, out, filter="data")
            relative = member.name[len(PREFIX) :]
            if member.isfile() and relative.split("/")[0] in [
                "vllm",
                "vllm_ascend",
                "betterscale",
                "ie_betterscale_bundle",
            ]:
                path = out / member.name
                protected[relative] = digest(path)
    manifest = {
        "schema": "reconstructed-school028-source/v1",
        "sdk": str(SDK),
        "sdk_sha256": SDK_SHA,
        "files": protected,
        "site_packages": str(runtime / "lib/python3.12/site-packages"),
        "qualification": "New verified-SDK reconstruction of the test family, not the historical full-answer optimized B1 implementation.",
    }
    manifest_path = out / "source-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    profile_path = ROOT / "runtime-profile.json"
    profiles = json.loads(profile_path.read_text()) if profile_path.exists() else {}
    profiles["B1"] = {
        "python": str(runtime / "bin/python"),
        "pythonpath": "/usr/local/Ascend/cann-9.1.0/python/site-packages:/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe",
        "identity_mode": "reconstructed",
        "reconstruction_manifest": str(manifest_path),
        "reconstruction_manifest_sha256": digest(manifest_path),
        "qualification": manifest["qualification"],
    }
    profile_path.write_text(json.dumps(profiles, indent=2) + "\n")
    print(
        "Recovered",
        len(protected),
        "protected source/library files; wrote explicit reconstructed profile",
        flush=True,
    )


if __name__ == "__main__":
    main()
