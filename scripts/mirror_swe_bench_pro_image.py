#!/usr/bin/env python3
"""Publish the pinned introductory ARM task image to an operator-owned registry.

Run once on a machine with upstream access. Uses ordinary crane/Docker registry
credentials; never copies any old workspace, daemon data or local task changes.
"""

import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--crane", required=True)
    p.add_argument(
        "--destination-prefix",
        required=True,
        help="operator registry[/prefix], without scheme",
    )
    p.add_argument("--source-prefix", default="")
    p.add_argument(
        "--insecure",
        action="store_true",
        help="explicitly allow an HTTP private registry",
    )
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--cache-directory",
        type=Path,
        help="retain an OCI layout so slow upstream reads do not stall registry uploads",
    )
    args = p.parse_args()
    if "://" in args.destination_prefix or "@" in args.destination_prefix:
        p.error(
            "destination-prefix must be a registry reference without credentials or scheme"
        )
    if args.output.exists():
        p.error("output already exists; preserve the previous registry proof")
    pin = json.loads(
        (
            ROOT / "src/vllm_hust_benchmark/data/swe_bench_pro_bootstrap_lock.json"
        ).read_text()
    )["single_task_image"]
    source = args.source_prefix + pin["source"] + "@" + pin["manifest_digest"]
    destination = (
        args.destination_prefix.rstrip("/") + "/" + pin["source"] + ":" + pin["tag"]
    )
    command = [args.crane] + (["--insecure"] if args.insecure else [])
    cache = args.cache_directory or args.output.parent / "swe-pro-arm-image.oci"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                args.crane,
                "pull",
                "--platform=linux/arm64",
                "--format=oci",
                source,
                str(cache),
            ],
            check=True,
            timeout=1800,
        )
    index = json.loads((cache / "index.json").read_text())
    if (
        len(index["manifests"]) != 1
        or index["manifests"][0]["digest"] != pin["manifest_digest"]
    ):
        raise ValueError("OCI cache does not contain the pinned upstream ARM manifest")
    subprocess.run(
        command + ["push", "--platform=linux/arm64", str(cache), destination],
        check=True,
        timeout=1800,
    )
    digest = subprocess.check_output(
        command + ["digest", "--platform=linux/arm64", destination],
        text=True,
        timeout=60,
    ).strip()
    if digest != pin["manifest_digest"]:
        raise ValueError(
            "private registry image differs from frozen upstream ARM manifest"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "source": source,
                "destination": destination,
                "manifest_digest": digest,
                "status": "VERIFIED",
            },
            indent=2,
        )
        + "\n"
    )
    print("VERIFIED private registry contains the pinned upstream ARM image")


if __name__ == "__main__":
    main()
