"""Prepare an original-dependency Docker context, without building or inference.

The explicit base must already provide Python3.10 and venv. It is a compatibility
base, not an official DiscoveryBench image. Transitive packages are not locked.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re


REQUIREMENTS_SHA256 = "dbcaad5519f58509cd1403e59472b1b3a5c2b7ba52e348e6380aaa4fbd36aaa4"  # pragma: allowlist secret - pinned original publisher requirements
AUTHOR_REVISION = "c31fcf011e070f021a5f5b906896d0821f6880e8"  # pragma: allowlist secret - public publisher software revision


def dockerfile(base_image, base_python):
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", base_image):
        raise ValueError("immutable local base image ID required")
    if (
        not re.fullmatch(r"/[a-zA-Z0-9_./-]+", base_python)
        or ".." in Path(base_python).parts
    ):
        raise ValueError("explicit absolute Python path without shell syntax required")
    return (
        f"FROM {base_image}\n"
        "COPY requirements.txt /opt/discovery-original-requirements.txt\n"
        f"RUN {base_python} -m venv /opt/discovery-client && "
        "/opt/discovery-client/bin/python -m pip install --no-cache-dir "
        "--index-url https://pypi.org/simple -r /opt/discovery-original-requirements.txt && "
        "/opt/discovery-client/bin/python -m pip check\n"
    )


def prepare(args):
    recipe = dockerfile(args.base_image, args.base_python)
    source = args.author_root.resolve(strict=True)
    raw = (source / "requirements.txt").read_bytes()
    if hashlib.sha256(raw).hexdigest() != REQUIREMENTS_SHA256:
        raise ValueError("unchanged pinned original requirements required")
    output = args.output.resolve()
    repository = Path(__file__).resolve().parents[1]
    if (
        output.exists()
        or output.is_relative_to(source)
        or source.is_relative_to(output)
        or output.is_relative_to(repository)
    ):
        raise ValueError("fresh context outside source and repository required")
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    (output / "requirements.txt").write_bytes(raw)
    (output / "Dockerfile").write_text(recipe, encoding="utf-8")
    receipt = {
        "source_id": "DISCOVERYBENCH",
        "status": "ORIGINAL_DEPENDENCY_BUILD_CONTEXT_PREPARED_NOT_BUILT",
        "author_revision": AUTHOR_REVISION,
        "original_requirements_sha256": REQUIREMENTS_SHA256,
        "base_image_id": args.base_image,
        "base_python": args.base_python,
        "base_is_official_discoverybench_image": False,
        "transitive_dependencies_hash_locked": False,
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }
    (output / "context-public.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base-image")
    parser.add_argument("--base-python", default="/opt/miniconda3/bin/python")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "PLAN_ONLY", "model_calls": 0}))
        return
    if any(
        getattr(args, key) is None for key in ("author_root", "output", "base_image")
    ):
        parser.error(
            "explicit original source, immutable base and fresh output required"
        )
    os.umask(0o077)
    prepare(args)


if __name__ == "__main__":
    main()
