#!/usr/bin/env python3
"""Publish pinned ARM task images to an operator-owned registry.

Run once on a machine with upstream access. Uses ordinary crane/Docker registry
credentials; never copies any old workspace, daemon data or local task changes.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

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
    p.add_argument("--cohort", choices=["v1-200"])
    p.add_argument(
        "--max-new-images",
        type=int,
        help="bound a cohort publication; 0 only checks the destination",
    )
    args = p.parse_args()
    if "://" in args.destination_prefix or "@" in args.destination_prefix:
        p.error(
            "destination-prefix must be a registry reference without credentials or scheme"
        )
    if args.output.exists():
        p.error("output already exists; preserve the previous registry proof")
    if args.max_new_images is not None and (not args.cohort or args.max_new_images < 0):
        p.error("--max-new-images requires a cohort and must be nonnegative")
    if args.cohort:
        return mirror_cohort(args)
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
    digest = publish(
        args.crane, command, source, destination, cache, pin["manifest_digest"]
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


def publish(crane, command, source, destination, cache, expected):
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                crane,
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
    if len(index["manifests"]) != 1 or index["manifests"][0]["digest"] != expected:
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
    if digest != expected:
        raise ValueError(
            "private registry image differs from frozen upstream ARM manifest"
        )
    return digest


def mirror_cohort(args):
    data = ROOT / "src/vllm_hust_benchmark/data"
    cohort = json.loads((data / "swe_bench_pro_v1_200.json").read_text())
    lock = json.loads((data / "swe_bench_pro_v1_lock.json").read_text())
    images = cohort["arm64_manifest_digests"]
    if (
        set(images) != set(cohort["instance_ids"])
        or len(images) != 200
        or cohort["dataset_sha256"] != lock["dataset"]["parquet_sha256"]
    ):
        raise ValueError("The pinned cohort has an invalid image inventory")
    repository = lock["community_images"]["dockerhub_username"] + "/sweap-images"
    destination = args.destination_prefix.rstrip("/") + "/" + repository
    command = [args.crane] + (["--insecure"] if args.insecure else [])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Each line is a complete receipt. A subsequent invocation with a new
    # receipt file will skip every digest that is already readable.
    copied = 0
    with args.output.open("x") as proof:
        for digest in dict.fromkeys(images.values()):
            reference = destination + "@" + digest
            result = subprocess.run(
                command + ["manifest", reference],
                check=False,
                capture_output=True,
                timeout=60,
            )
            if result.returncode == 0:
                hashes = {
                    "sha256:" + hashlib.sha256(raw).hexdigest()
                    for raw in (result.stdout, result.stdout.removesuffix(b"\n"))
                }
                if digest not in hashes:
                    raise ValueError(
                        "Registry returned a different digest for " + reference
                    )
                status = "ALREADY_PRESENT"
            elif not any(
                marker in result.stderr
                for marker in (b"MANIFEST_UNKNOWN", b"NAME_UNKNOWN", b"404")
            ):
                raise RuntimeError(
                    "Registry availability check failed for " + reference
                )
            elif args.max_new_images is not None and copied >= args.max_new_images:
                status = "MISSING"
            else:
                tag = destination + ":swe-pro-v1-200-" + digest.removeprefix("sha256:")
                source = args.source_prefix + repository + "@" + digest
                if args.cache_directory is not None:
                    cache = args.cache_directory / digest.removeprefix("sha256:")
                    publish(args.crane, command, source, tag, cache, digest)
                else:
                    with tempfile.TemporaryDirectory(
                        dir=args.output.parent
                    ) as directory:
                        publish(
                            args.crane,
                            command,
                            source,
                            tag,
                            Path(directory) / "image",
                            digest,
                        )
                copied += 1
                status = "VERIFIED"
            proof.write(json.dumps({"destination": reference, "status": status}) + "\n")
            proof.flush()
    print(f"Published {copied} new ARM image digests; see {args.output}")


if __name__ == "__main__":
    main()
