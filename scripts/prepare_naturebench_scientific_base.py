"""Derive a disclosed ARM64 CPU scientific recipe from verified original code.

Software preparation only: no downloads, installs, Docker or model execution.
"""

import argparse
import hashlib
import json
from pathlib import Path

from prepare_naturebench_native_runtime import original_files


BASE = "nvidia/cuda@sha256:8f9dd0d09d3ad3900357a1cf7f887888b5b74056636cd6ef03c160c3cd4b1d95"


def derive(raw, cached=False):
    replacements = [
        (
            b"FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04",
            ("FROM " + BASE).encode(),
        ),
        (
            b"--index-url https://download.pytorch.org/whl/cu118",
            b"--index-url https://pypi.org/simple",
        ),
    ]
    for before, after in replacements:
        if raw.count(before) != 1:
            raise ValueError("original scientific recipe differs; no heuristic patch")
        raw = raw.replace(before, after, 1)
    if cached:
        marker = b"RUN python -m pip install"
        if marker not in raw:
            raise ValueError("original scientific install marker missing")
        raw = raw.replace(
            marker,
            b"COPY wheelhouse /opt/naturebench-wheelhouse\nENV PIP_FIND_LINKS=/opt/naturebench-wheelhouse\n"
            + marker,
            1,
        )
    return raw


def prepare(author_root, output, *, cached=False, write=False):
    root, pins = original_files(author_root)
    output = Path(output).resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError(
            "scientific recipe output must be separate from original source"
        )
    if output.exists():
        raise FileExistsError("fresh scientific recipe output required")
    original = (root / "docker/Dockerfile.base").read_bytes()
    derived = derive(original, cached)
    result = {
        "source_id": "NATUREBENCH",
        "status": "SCIENTIFIC_RECIPE_PREPARED_NOT_BUILT"
        if write
        else "OFFLINE_SCIENTIFIC_RECIPE_PLAN",
        "code_revision": pins["revision"],
        "original_recipe_sha256": hashlib.sha256(original).hexdigest(),
        "derived_recipe_sha256": hashlib.sha256(derived).hexdigest(),
        "base_image": BASE,
        "wheelhouse_required_in_build_context": cached,
        "compatibility": [
            "Same official base image pinned by immutable index",
            "Original torch/vision/audio versions unchanged; official PyPI ARM64 CPU wheels instead of unavailable cu118 wheels",
        ],
        "fixed_original_package_versions_changed": False,
        "model_calls": 0,
        "scientific_task_executed": False,
        "source_checkbox_eligible": False,
        "output_created": write,
    }
    if write:
        output.mkdir(mode=0o700, parents=True, exist_ok=False)
        for name, raw in [
            ("Dockerfile.base.original", original),
            ("Dockerfile.cpu-arm64", derived),
            ("LICENSE", (root / "LICENSE").read_bytes()),
            ("NOTICE", (root / "NOTICE").read_bytes()),
        ]:
            with (output / name).open("xb") as stream:
                stream.write(raw)
        (output / "recipe-public.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cached-wheelhouse",
        action="store_true",
        help="Explicit local software cache; blobs are not copied or verified here",
    )
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.author_root,
                args.output,
                cached=args.cached_wheelhouse,
                write=args.write,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
